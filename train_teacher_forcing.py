import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from munch import Munch
import yaml
from models import build_model, GTRStyleEncoder
from meldataset import build_dataloader


def main():
    ckpt_path = "/storageNVME/melissa/ckpts/stylettsCN/libritts_aishell3_s2/libritts_aishell3_s1-epoch_1st_00016.pth"
    train_list = "Data/gtr_train.txt"
    val_list = "Data/gtr_test.txt"
    device = "cuda:0"
    batch_size = 16
    num_epochs = 1000
    config_path = "/home/melissa/ArtiVoice-GTR/Configs/config_1st.yml"
    config = yaml.safe_load(open(config_path))

    model = build_model(Munch(config['model_params']), text_aligner=None, pitch_extractor=None)
    style_encoder = model.style_encoder
    gtr_encoder = GTRStyleEncoder(out_dim=128, style_dim=128)

    state = torch.load(ckpt_path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params:
            if key == "style_encoder":
                style_encoder[key].load_state_dict(params[key])
    
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(gtr_encoder.parameters(), lr=0.001)

    # Define your dataset and dataloader
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
        for i, batch in enumerate(train_dataloader):
            paths, texts, input_lengths, mels, mel_input_length, tones = batch
            paths = paths.to(device)

            optimizer.zero_grad()
            
            with torch.no_grad():
                pretrained_outputs = style_encoder(paths)
            
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