import os, sys
import gradio as gr
import yt_dlp
from audio_separator.separator import Separator

now_dir = os.getcwd()
sys.path.append(now_dir)

download_dir = os.path.join(now_dir, "assets", "audios-others")

audio_root = os.path.join(now_dir, "assets", "audios")



def download_audio(url):
    # Create downloads folder if it doesn't exist.
    
    
    # Extract video info to obtain a unique video id.
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info_dict = ydl.extract_info(url, download=False)
    video_id = info_dict.get("id")
    if not video_id:
        raise ValueError("Could not extract video id from URL")
    
    # Determine the output file name (we save as mp3).
    output_file = os.path.join(download_dir, f"{video_id}.mp3")
    if os.path.exists(output_file):
        print("Audio already downloaded. Skipping download.")
        return output_file
    
    # Download audio with yt-dlp.
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(download_dir, f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_file

def separate_audio_from_url(url):
    # Download the audio if needed.
    input_audio = download_audio(url)
    
    # Create output directory if it doesn't exist.
    
    # Initialize the separator with the output directory.
    separator = Separator(output_dir=audio_root)
    
    # Define output paths.
    vocals_path = os.path.join(output_dir, 'Vocals.wav')
    instrumental_path = os.path.join(download_dir, 'Instrumental.wav')
    vocals_reverb_path = os.path.join(output_dir, 'Vocals (Reverb).wav')
    vocals_no_reverb_path = os.path.join(output_dir, 'Vocals (No Reverb).wav')
    lead_vocals_path = os.path.join(output_dir, 'Lead Vocals.wav')
    backing_vocals_path = os.path.join(output_dir, 'Backing Vocals.wav')
    
    # --- Stage 1: Split into Vocals and Instrumental ---
    separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    voc_inst = separator.separate(input_audio)
    os.rename(os.path.join(output_dir, voc_inst[0]), instrumental_path)
    os.rename(os.path.join(output_dir, voc_inst[1]), vocals_path)
    
    # --- Stage 2: Process Vocals (DeEcho-DeReverb) ---
    separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
    voc_no_reverb_output = separator.separate(vocals_path)
    os.rename(os.path.join(output_dir, voc_no_reverb_output[0]), vocals_no_reverb_path)
    os.rename(os.path.join(output_dir, voc_no_reverb_output[1]), vocals_reverb_path)
    
    # --- Stage 3: Separate Backing and Lead Vocals ---
    separator.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    backing_voc = separator.separate(vocals_no_reverb_path)
    os.rename(os.path.join(output_dir, backing_voc[0]), backing_vocals_path)
    os.rename(os.path.join(output_dir, backing_voc[1]), lead_vocals_path)
    
    return (instrumental_path, vocals_path, vocals_reverb_path, 
            vocals_no_reverb_path, lead_vocals_path, backing_vocals_path)

def uvr_tabs():
  with gr.Row():
    url_input = gr.Text(label="Enter Audio/Video URL (supported by yt-dlp)")
    
  with gr.Row():
    output_instrumental = gr.Audio(label="Instrumental", type="filepath")
    output_vocals = gr.Audio(label="Vocals", type="filepath")
  with gr.Row():
    output_vocals_reverb = gr.Audio(label="Vocals (Reverb)", type="filepath")
    output_vocals_no_reverb = gr.Audio(label="Vocals (No Reverb)", type="filepath")
  with gr.Row():
    output_lead_vocals = gr.Audio(label="Lead Vocals", type="filepath")
    output_backing_vocals = gr.Audio(label="Backing Vocals", type="filepath")
  
  run_button = gr.Button("Separate Audio")
  run_button.click(
    fn=separate_audio_from_url,
    inputs=url_input,
    outputs=[output_instrumental, output_vocals, output_vocals_reverb, 
             output_vocals_no_reverb, output_lead_vocals, output_backing_vocals]
  )
    
