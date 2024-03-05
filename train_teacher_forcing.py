import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from munch import Munch
import yaml
from utils import *
from models import build_model, GTRStyleEncoder, StyleEncoder
from meldataset import build_dataloader
from tqdm import tqdm


def main():
    ckpt_path = "/storageNVME/melissa/ckpts/stylettsCN/pretrained/Models/libritts/epoch_2nd_00050.pth"
    train_path = "Data/gtr_train.txt"
    val_path = "Data/gtr_test.txt"
    device = "cuda:0"
    batch_size = 16
    num_epochs = 1000
    # config_path = "/home/melissa/ArtiVoice-GTR/Configs/config_1st.yml"
    # config = yaml.safe_load(open(config_path))

    # model = build_model(Munch(config['model_params']), text_aligner=None, pitch_extractor=None)
    # style_encoder = model.style_encoder
    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    gtr_encoder = GTRStyleEncoder(out_dim=128, style_dim=7)

    state = torch.load(ckpt_path, map_location='cpu')
    style_encoder.load_state_dict(state['net']["style_encoder"])

    style_encoder = style_encoder.to(device)
    gtr_encoder = gtr_encoder.to(device)
    
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(gtr_encoder.parameters(), lr=0.001)

    # Define your dataset and dataloader
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

    # Training loop
    for epoch in range(num_epochs):
        gtr_encoder.train()
        running_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader)):
            paths, texts, input_lengths, mels, mel_input_length, tones = batch
            paths = paths.to(device)
            mels = mels.to(device)
            mel_input_length = mel_input_length.to(device)

            optimizer.zero_grad()
            
            with torch.no_grad():
                gt = []
                mel_len = int(mel_input_length.min().item() / 2 - 1)
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                gt = torch.stack(gt).detach()
                pretrained_outputs = style_encoder(gt.unsqueeze(1))
            
            second_outputs = gtr_encoder(paths)
            
            loss = criterion(second_outputs, pretrained_outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * paths.size(0)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')

    print('Training finished')

if __name__ == "__main__":
    main()