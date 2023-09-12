# Load model directly
from transformers import AutoProcessor, SpeechT5ForTextToSpeech
from IPython.display import Audio
from transformers import SpeechT5HifiGan
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

import torch
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

text = '我会说中文吗？，我不会说中文。，我会说中文吗？，我不会说中文。'
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)


print(torch.min(speech), torch.max(speech), torch.mean(speech))

Audio(speech.numpy(), rate=16000)


import soundfile as sf
sf.write("output.wav", speech.numpy(), samplerate=16000)
