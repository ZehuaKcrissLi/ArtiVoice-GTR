
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
import soundfile as sf
from pathlib import Path
import numpy as np
from model import LinearClassifier
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
# set model
model = LinearClassifier(input_dim=1024, num_classes=5)
# get wav paths
wav_dir = Path("/storageNVME/kcriss/picked_sliced")
wav_paths = [str(path) for path in wav_dir.glob("*.wav")]

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2) 

# initialize feature extractor and dataset and model
model_path="TencentGameMate/chinese-hubert-large"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
dataset = AudioDataset(file_paths=wav_paths, feature_extractor=feature_extractor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# set model to device and half precision
model = HubertModel.from_pretrained(model_path)
model = model.to(device)
model = model.half()
model.eval()

# get hidden states
for batch in dataloader:
    batch = batch.to(device).half()

    with torch.no_grad():
        outputs = model(batch, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

    for idx, tensor in enumerate(last_hidden_state):
        print(f"Batch {idx}, Tensor shape: {tensor.shape}")