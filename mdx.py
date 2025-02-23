import gc
import hashlib
import os
import queue
import threading
import warnings
import logging

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm

# Configure logging: change level to DEBUG for verbose output
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

warnings.filterwarnings("ignore")
stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}


class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        logging.info("Initializing MDXModel")
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        logging.debug(f"Window shape: {self.window.shape}")

        out_c = self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)
        logging.debug(f"Frequency pad shape: {self.freq_pad.shape}")

    def stft(self, x):
        logging.debug("Performing STFT")
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        result = x[:, :, :self.dim_f]
        logging.debug(f"STFT result shape: {result.shape}")
        return result

    def istft(self, x, freq_pad=None):
        logging.debug("Performing inverse STFT")
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        result = x.reshape([-1, 2, self.chunk_size])
        logging.debug(f"iSTFT result shape: {result.shape}")
        return result


class MDX:
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR
    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path: str, params: MDXModel, processor=DEFAULT_PROCESSOR):
        logging.info("Initializing MDX session")
        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']
        logging.debug(f"Device: {self.device}, Providers: {self.provider}")

        self.model = params

        # Load the ONNX model using ONNX Runtime
        logging.info(f"Loading ONNX model from {model_path}")
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(None, {'input': torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        logging.debug("ONNX model preloaded")
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]

        self.prog = None

    @staticmethod
    def get_hash(model_path):
        logging.info(f"Computing hash for model at {model_path}")
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
                logging.debug("Computed hash using seek method")
        except Exception as e:
            logging.warning(f"Seek method failed: {e}. Using full file read.")
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        logging.debug(f"Model hash: {model_hash}")
        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        logging.info("Segmenting wave array")
        if combine:
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:
                    processed_wave = segment[:, start:end]
                else:
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)
                logging.debug(f"Combined segment {segment_count}: shape {segment[:, start:end].shape}")
        else:
            processed_wave = []
            sample_count = wave.shape[-1]
            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count
            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin
                cut = wave[:, start:end].copy()
                processed_wave.append(cut)
                logging.debug(f"Segment {segment_count}: start {start}, end {end}, shape {cut.shape}")
                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        logging.info("Padding wave array")
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size
        logging.debug(f"n_sample: {n_sample}, trim: {trim}, gen_size: {gen_size}, pad: {pad}")

        # Padded wave
        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)
        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)
            logging.debug(f"Padded segment starting at {i}: shape {waves.shape}")

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        logging.info(f"Total padded segments: {mix_waves.shape[0]}")
        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        logging.info(f"Thread {_id}: Starting processing wave segments")
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for i, mix_wave in enumerate(mix_waves):
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
                logging.debug(f"Thread {_id}: Processed segment {i}, shape {processed_wav.shape}")
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        logging.info(f"Thread {_id}: Finished processing wave segments")
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        logging.info("Starting multi-threaded wave processing")
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)
        logging.debug(f"Wave segmented into {len(waves)} parts")

        # Create a queue to hold the processed wave segments
        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(target=self._process_wave, args=(mix_waves, trim, pad, q, c))
            thread.start()
            threads.append(thread)
            logging.info(f"Started thread {c} for batch processing")
        for thread in threads:
            thread.join()
            logging.info("Thread joined")
        self.prog.close()

        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [list(wave.values())[0] for wave in
                             sorted(processed_batches, key=lambda d: list(d.keys())[0])]
        if len(processed_batches) != len(waves):
            logging.error("Incomplete processed batches, please reduce batch size!")
            raise AssertionError('Incomplete processed batches, please reduce batch size!')
        logging.info("Multi-threaded processing complete")
        return self.segment(processed_batches, True, chunk)


def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    logging.info("Running MDX separation process")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logging.debug(f"Selected device: {device}")

    device_properties = torch.cuda.get_device_properties(device) if torch.cuda.is_available() else None
    if device_properties:
        vram_gb = device_properties.total_memory / 1024**3
        logging.debug(f"GPU VRAM: {vram_gb:.2f} GB")
        m_threads = 1 if vram_gb < 8 else 2
    else:
        logging.debug("Using CPU, setting m_threads=2 by default")

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    logging.info(f"Model hash: {model_hash}")
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"]
    )

    mdx_sess = MDX(model_path, model)
    logging.info(f"Loading audio file: {filename}")
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    logging.debug(f"Loaded wave shape: {wave.shape}, sample rate: {sr}")

    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    logging.debug(f"Audio peak before normalization: {peak}")
    wave /= peak

    if denoise:
        logging.info("Performing denoising process")
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak

    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        logging.info(f"Writing main output file to {main_filepath}")
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        logging.info(f"Writing inversion output file to {invert_filepath}")
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        logging.info(f"Removing original file: {filename}")
        os.remove(filename)

    # Clean up
    del mdx_sess, wave_processed, wave
    gc.collect()
    logging.info("MDX separation process complete")
    return main_filepath, invert_filepath
