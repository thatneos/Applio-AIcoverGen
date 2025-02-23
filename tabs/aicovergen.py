import json
import os
from argparse import ArgumentParser

import gradio as gr
from main import song_cover_pipeline

# Define directories (adjust as needed)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'logs')

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['mute', 'reference']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)

def aicovergen():
    """
    Returns a Gradio TabbedInterface for generating AI cover songs.
    This implementation uses gr.Interface (one per tab) and gr.TabbedInterface,
    which is available in Gradio 5.x. (No gr.Blocks is used here.)
    """

    # Load available voice models
    voice_models = get_current_models(rvc_models_dir)
    # A hidden component used to flag the webUI context
    is_webui = gr.Number(value=1, visible=False)

    # Define input components for the Generate tab.
    # (Note: for clarity, some of the original layout components have been omitted.)
    song_input = gr.Text(label='Song input', info='Link to a YouTube song or full path to a local file.')
    rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Select one of the available voice models.')
    pitch = gr.Slider(-3, 3, value=0, step=1, label='Pitch Change (Vocals ONLY)', info='Use 1 for male→female conversion and -1 for vice-versa (in octaves).')
    keep_files = gr.Checkbox(label='Keep intermediate files', info='Keep audio files generated in the output directory.')
    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocals Gain')
    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocals Gain')
    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music Gain')
    index_rate = gr.Slider(0, 1, value=0.5, label='Index Rate', info="Controls how much of the AI voice's accent to retain.")
    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius', info='Apply median filtering if value ≥ 3.')
    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate', info="Blends original vocal loudness (0) with fixed loudness (1).")
    f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Pitch Detection Algorithm', info='rmvpe is recommended for clarity; mangio-crepe gives smoother vocals.')
    crepe_hop_length = gr.Slider(32, 320, value=128, step=1, label='Crepe Hop Length', info='Lower values yield better pitch accuracy at the cost of speed.')
    protect = gr.Slider(0, 0.5, value=0.33, label='Protect Rate', info='Protect voiceless consonants and breaths (0.5 disables protection).')
    pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Overall Pitch Change', info='Alters both vocals and instrumentals (in semitones).')
    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Reverb Room Size', info='Larger values simulate a bigger room.')
    reverb_wet = gr.Slider(0, 1, value=0.2, label='Reverb Wet Level', info='Amount of reverb effect on AI vocals.')
    reverb_dry = gr.Slider(0, 1, value=0.8, label='Reverb Dry Level', info='Amount of dry (unprocessed) signal.')
    reverb_damping = gr.Slider(0, 1, value=0.7, label='Reverb Damping', info='High damping absorbs more high frequencies.')
    output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Output Format', info='mp3 offers smaller file size; wav is higher quality.')
    
    ai_cover = gr.Audio(label='AI Cover', show_share_button=False)

    # Define the function to be wrapped by the interface.
    # (This calls the song_cover_pipeline with the provided inputs.)
    def generate_fn(song_input, rvc_model, pitch, keep_files, is_webui, main_gain,
                    backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                    f0_method, crepe_hop_length, protect, pitch_all, reverb_rm_size,
                    reverb_wet, reverb_dry, reverb_damping, output_format):
        return song_cover_pipeline(song_input, rvc_model, pitch, keep_files, is_webui,
                                   main_gain, backup_gain, inst_gain, index_rate, filter_radius,
                                   rms_mix_rate, f0_method, crepe_hop_length, protect, pitch_all,
                                   reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, output_format)

    # Build the "Generate" interface (a single tab).
    
