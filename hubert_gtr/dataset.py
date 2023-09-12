from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,)
import soundfile as sf

class AudioDataset(Dataset):
    def __init__(self, file_paths, feature_extractor):
        self.file_paths = file_paths
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        wav, sr = sf.read(wav_path)
        input_values = self.feature_extractor(wav, return_tensors="pt").input_values[0]
        return input_values

