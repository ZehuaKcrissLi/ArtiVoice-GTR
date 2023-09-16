import torch
import torch.nn.functional as F
import soundfile as sf

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(2)
model_path="TencentGameMate/chinese-hubert-large"
wav_path="/home/kcriss/artivoice/hubert_gtr/test.wav"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = HubertModel.from_pretrained(model_path)

model = model.to(device)
model = model.half()
model.eval()

wav, sr = sf.read(wav_path)
input_values = feature_extractor(wav, return_tensors="pt").input_values
input_values = input_values.half()
input_values = input_values.to(device)

with torch.no_grad():
    outputs = model(input_values, output_hidden_states=True).hidden_states
    
# for idx, tensor in enumerate(outputs):
#     print(f"Tensor {idx} shape: {tensor.shape}")
    

# hidden_states_tensor = torch.stack(outputs)
# print(f"Hidden states tensor shape: {hidden_states_tensor.size()}")
print(outputs)