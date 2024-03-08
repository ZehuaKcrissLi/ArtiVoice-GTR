import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pickle
from munch import Munch
import yaml
from utils import *
from models import build_model, GTRStyleEncoder, StyleEncoder, GTRStyleEncoder3
from meldataset import build_dataloader
from tqdm import tqdm

import wandb
wandb.login()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_pretrained_style(ckpt_path="/storageNVME/melissa/ckpts/stylettsCN/pretrained/Models/libritts/epoch_2nd_00050.pth", 
                                   device="cpu", 
                                   filelist="Data/gtr_train.txt"):
    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    state = torch.load(ckpt_path, map_location='cpu')
    style_encoder.load_state_dict(state['net']["style_encoder"])
    style_encoder = style_encoder.to(device)

    reference_embeddings = {}

    with open(filelist, "r") as f:
        for _, path in tqdm(enumerate(f), total=2500):
            path = path.strip().split("|")[0]
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            mel_tensor = preprocess(audio).to(device)
            try:
                with torch.no_grad():
                    ref = style_encoder(mel_tensor.unsqueeze(1))
                path_s = path.split("/")
                utt_id = "_".join([path_s[-2], path_s[-1].split(".")[0]])
                reference_embeddings[utt_id] = ref.squeeze(1).cpu().numpy()
            except Exception as e:
                print(e)
        return reference_embeddings


def averge_embedding(infile="/home/melissa/ArtiVoice-GTR/Data/pretrained_style_embeddings.pkl", 
                     outfile="/home/melissa/ArtiVoice-GTR/Data/pretrained_style_embeddings_ave.pkl"
                     ):
    embeds = pickle.load(open(infile, "rb"))
    embeds_ave = {}
    for k, v in embeds.items():
        label = k.split("_")[0]
        embeds_ave[int(label)] = embeds_ave.get(label, []) + [v]
    
    for k, v in embeds_ave.items():
        ave = np.stack(embeds_ave[k]).mean(0)
        embeds_ave[k] = ave

    with open(outfile, "wb") as f:
        pickle.dump(embeds_ave, f)


def main():
    # averge embedding from 20 sentences
    train_path = "Data/gtr_train.txt"
    val_path = "Data/gtr_test.txt"
    device = "cuda:2"
    version = "v0_ave_embedding"
    batch_size = 64
    num_epochs = 1000
    style_embed_file = "/home/melissa/ArtiVoice-GTR/Data/pretrained_style_embeddings_ave.pkl"

    wandb.init(
        project="gtr_style_encoder",
        name=version,
        mode="disabled", 
    )

    if os.path.exists(style_embed_file):
        pretrained_outputs_map = pickle.load(open(style_embed_file, "rb"))
    else:
        # pretrained_outputs_map = compute_pretrained_style(device=device, filelist=train_path)
        # with open(style_embed_file, "wb") as f:
        #     pickle.dump(pretrained_outputs_map, f)
        raise notImplementedError

    gtr_encoder = GTRStyleEncoder(out_dim=128, style_dim=4).to(device)
    print("training version", version)
    print(gtr_encoder)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gtr_encoder.parameters(), lr=0.001)

    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config={},
                                        collate_config={"return_wave": True},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      collate_config={"return_wave": True},
                                      dataset_config={})


    for epoch in range(num_epochs):
        gtr_encoder.train()
        running_loss = 0.0
        num_utts = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            paths, texts, input_lengths, mels, mel_input_length, tones = batch
            mels = mels.to(device)
            mel_input_length = mel_input_length.to(device)

            optimizer.zero_grad()

            pretrained_outputs = []
            paths_ints = torch.zeros(batch_size).long()
            for i, path in enumerate(paths):
                # path_s = path.split("/")
                # paths_ints[i] = int(path_s[-2])
                # utt_id = "_".join([path_s[-2], path_s[-1].split(".")[0]])
                # pretrained_outputs.append(pretrained_outputs_map[utt_id])
                pretrained_outputs.append(pretrained_outputs_map[int(path.cpu().numpy())])
            
            pretrained_outputs = torch.from_numpy(np.stack(pretrained_outputs)).to(device)
            paths_ints = paths_ints.to(device)
            gtr_output = gtr_encoder(paths_ints)
            
            loss = criterion(gtr_output, pretrained_outputs.squeeze(1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * paths_ints.size(0)
            num_utts += paths_ints.size(0)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / num_utts:.4f}')
        wandb.log({"train/loss": running_loss / num_utts})

    print('Training finished')



def main0():  # faster...
    # on the fly embedding extraction
    train_path = "Data/gtr_train.txt"
    val_path = "Data/gtr_test.txt"
    output_dir = "/storageNVME/melissa/ckpts/stylettsCN/pretrained"
    resume_training = "/storageNVME/melissa/ckpts/stylettsCN/pretrained/gtr_encoder_v1_500.pth"
    version = "test"
    device = "cuda:3"
    batch_size = 64
    num_epochs = 1000

    wandb.init(
        project="gtr_style_encoder",
        name=version,
        mode="disabled", 
    )

    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    state = torch.load("/storageNVME/melissa/ckpts/stylettsCN/pretrained/Models/libritts/epoch_2nd_00050.pth", map_location='cpu')
    style_encoder.load_state_dict(state['net']["style_encoder"])
    style_encoder = style_encoder.to(device)
    # gtr_encoder = GTRStyleEncoder(out_dim=128, style_dim=4).to(device)
    gtr_encoder = GTRStyleEncoder(out_dim=128, style_dim=4).to(device)
    if resume_training != "":
        state = torch.load(resume_training, map_location='cpu')
        gtr_encoder.load_state_dict(state)
    print("training version", version)
    print(gtr_encoder)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gtr_encoder.parameters(), lr=0.001)

    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config={},
                                        collate_config={"return_wave": True},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      collate_config={"return_wave": True},
                                      dataset_config={})


    for epoch in range(num_epochs):
        gtr_encoder.train()
        running_loss = 0.0
        num_utts = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            paths, texts, input_lengths, mels, mel_input_length, tones = batch
            paths = paths.to(device)
            mels = mels.to(device)
            mel_input_length = mel_input_length.to(device)

            with torch.no_grad():
                pretrained_outputs = style_encoder(mels.unsqueeze(1))

            optimizer.zero_grad()
            gtr_output = gtr_encoder(paths)
            
            loss = criterion(gtr_output, pretrained_outputs)

            # print((pretrained_outputs**2).mean()) # 0.03
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * paths.size(0)
            num_utts += paths.size(0)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / num_utts:.4f}')
        wandb.log({"train/loss": running_loss / num_utts})

        if epoch % 50 == 0:
            torch.save(gtr_encoder.state_dict(), os.path.join(output_dir, f"gtr_encoder_{version}_{epoch}.pth"))
            print("Model saved")

    print('Training finished')

if __name__ == "__main__":
    main0()
    # main()
    # averge_embedding()