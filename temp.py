import torch
import numpy as np

import librosa
import soundfile as sf

import pickle
from pathlib import Path

from models import StyleEncoder
from train_teacher_forcing import preprocess

device = "cuda:3"

def extact_s():
    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    state = torch.load("/storageNVME/melissa/ckpts/stylettsCN/pretrained/Models/libritts/epoch_2nd_00050.pth", map_location='cpu')
    style_encoder.load_state_dict(state['net']["style_encoder"])
    style_encoder = style_encoder.to(device)

    S = {}
    indir = Path("/storageNVME/melissa/rand_50_no01")

    for file in indir.glob("*.wav"):
        wave, sr = sf.read(file)
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            s = style_encoder(mel_tensor.unsqueeze(1))
        S[file.stem] = s.squeeze(1).cpu().numpy()
    
    with open("/home/melissa/ArtiVoice-GTR/Data/rand_50_no01_style_embeddings.pkl", "wb") as f:
        pickle.dump(S, f)


def match_s():
    # pretrained = np.load("/home/melissa/ArtiVoice-GTR/Data/pretrained_style_embeddings_ave.pkl", allow_pickle=True)
    pretrained = np.load("/home/melissa/ArtiVoice-GTR/Data/pretrained_style_embeddings.pkl", allow_pickle=True)
    extracted = np.load("/home/melissa/ArtiVoice-GTR/Data/rand_50_no01_style_embeddings.pkl", allow_pickle=True)
    
    mse_results = {}
    for k, v in extracted.items():
        mse_list = {}
        for k1, v1 in pretrained.items():

            # for utt level
            k1_s = k1.split("_")
            if k1_s[1] != "01":
                continue
            k1 = int(k1_s[0])

            mse =  ((v - v1)**2).mean()
            mse_list[k1] = mse
        sorted_keys = sorted(mse_list, key=mse_list.get, reverse=True)
        sorted_mse_list = {s: mse_list[s] for s in sorted_keys}
        mse_results[k] = sorted_mse_list
        
        import pdb; pdb.set_trace()
        print(k, mse_results[k])
    
    with open("a", "wb") as f:
        pickle.dump(mse_results, f)

if __name__=="__main__":
    # extact_s()
    match_s()
    pass 
