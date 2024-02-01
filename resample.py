import os
import torchaudio
from tqdm import tqdm

def resample_torchaudio(input_folder, output_folder, output_sample_rate=48000):
    """
    Resample all audio files in the input folder and save them in the output folder.
    """
    # wav files are in subfolders
    for subfolder in tqdm(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(output_folder, subfolder)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            
            for file in os.listdir(subfolder_path):
                if file.endswith(".wav"):
                    audio, sample_rate = torchaudio.load(os.path.join(subfolder_path, file))
                    resampler = torchaudio.transforms.Resample(sample_rate, output_sample_rate)
                    audio = resampler(audio)
                    output_filename = f"{os.path.splitext(file)[0]}.wav"
                    torchaudio.save(os.path.join(output_subfolder, output_filename), audio, output_sample_rate)

if __name__ == "__main__":
    input_folder = "/storageNVME/kcriss/gtr_picked_vad"
    output_folder = "/storageNVME/kcriss/gtr_picked_vad_resampled_torchaudio"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    resample_torchaudio(input_folder, output_folder)
