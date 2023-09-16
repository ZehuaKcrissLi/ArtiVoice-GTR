from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor
import soundfile as sf

from torch.utils.data import Dataset
import soundfile as sf
from transformers import (
    Wav2Vec2FeatureExtractor, 
    HubertModel
)

from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, file_paths, model_path):
        self.file_paths = file_paths
        
                
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        wav, sr = sf.read(wav_path)
        input_values = self.feature_extractor(wav, return_tensors="pt").input_values
        input_values = input_values.half()
        input_values = input_values.to(self.device)
        
        return input_values


    
# model_path = "TencentGameMate/chinese-hubert-large"
# wav_dir = Path("/storageNVME/kcriss/picked_sliced")
# wav_paths = [str(path) for path in wav_dir.glob("*.wav")]
# dataset = AudioDataset(file_paths=wav_paths, model_path=model_path)
