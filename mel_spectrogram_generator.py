import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from multiprocessing import Pool

def audio_to_spectrogram(file_path, output_file):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Create a mel-scaled spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Create the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    # Save the spectrogram
    plt.savefig(output_file)
    plt.close()

def process_file(file_name):
    if file_name.endswith('.flac'):
        full_path = os.path.join(folder_path, file_name)
        output_name = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.png')
        audio_to_spectrogram(full_path, output_name)
        print(f"Processed: {file_name}")

folder_path = '/Users/taehmotheus/Desktop/AI-voice-detection/test_data/example/spoof'

# List of .flac files
flac_files = [f for f in os.listdir(folder_path) if f.endswith('.flac')]

# Using multiprocessing
if __name__ == '__main__':
    with Pool() as pool:
        pool.map(process_file, flac_files)
