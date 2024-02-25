import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
import soundfile as sf
import librosa
from pathlib import Path
import numpy as np
import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
from natsort import natsorted


def extract_features(model_path, data_path, feature_path, feature_extractor):
    # 检查特征路径是否存在，不存在则创建
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    data_path = '/storageNVME/kcriss/gtr-145'

    features_folder_path = f'{data_path}-features'
    if not os.path.exists(features_folder_path):
        os.makedirs(features_folder_path)

    # 使用glob和natsort获取所有wav文件的路径
    wav_paths = natsorted(glob.glob(os.path.join(data_path, '**/*.wav'), recursive=True))

    # 遍历每个wav文件
    for wav_path in wav_paths:
        # 加载wav文件
        wav, sr = librosa.load(wav_path)
        
        # 提取特征
        features = feature_extractor.extract_features(wav)
        
        # 分解路径以获取父目录名称和文件名称
        parent_dir_name = os.path.basename(os.path.dirname(wav_path))
        wav_file_name = os.path.basename(wav_path)
        
        # 从父目录名称提取最后三位数字
        parent_dir_last_three_digits = parent_dir_name[-3:]
        
        # 构建特征文件名，将wav扩展名改为.npy
        feature_file_name = parent_dir_last_three_digits + '_' + os.path.splitext(wav_file_name)[0] + '.npy'
        
        # 构建完整的特征文件路径
        feature_file_path = os.path.join(features_folder_path, feature_file_name)
        
        # 保存特征
        np.save(feature_file_path, features)
    # # convert pt to np
    # input_values = input_values.numpy()


class AudioDataset(Dataset):
    def __init__(self, file_path, GTR='G', mode='train'):
        self.GTR = GTR
        self.files = []
        self.labels = []
        
        # find all the folders under the file path
        for folder in os.listdir(file_path):
            folder_path = os.path.join(file_path, folder)
            if os.path.isdir(folder_path):
                # find all the files under the folder
                file_lists = os.listdir(folder_path)

                # split the file lists into train and test
                if mode == 'train':
                    file_lists = file_lists[:int(len(file_lists) * 0.8)]
                else:
                    file_lists = file_lists[int(len(file_lists) * 0.8):]

                for file in file_lists:
                    # get the folder name
                    folder_name = folder_path.split('/')[-1]
                    digits = re.findall(r'\d+', folder_name)  # extract digits

                    if not digits:
                        raise ValueError(
                            "The folder name must contain digits!")

                    # get the label
                    if self.GTR == 'G':
                        cur_label = int(digits[1][-3])  # 第一个数字的倒数第三个字符
                    elif self.GTR == 'T':
                        cur_label = int(digits[1][-2])  # 第一个数字的倒数第二个字符
                    elif self.GTR == 'R':
                        cur_label = int(digits[1][-1])  # 第一个数字的最后一个字符
                    else:
                        raise ValueError("GTR must be one of G, T, R!")

                    self.labels.append(cur_label)
                    self.files.append(os.path.join(folder_path, file))

        # define the labels and make them to be from 0 to num_classes
        self.labels = np.array(self.labels)
        self.labels = self.labels - np.min(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        # read the npy file to get the feature
        feature = np.load(wav_path)
        # convert to tensor
        feature = torch.from_numpy(feature)
        # get the label
        label = self.labels[idx]
        # conver to tensor
        label = torch.tensor(label)

        return feature, label

    def collation_fn(batch):
        # Unzip the batch
        features, labels = zip(*batch)

        # Find the max length in this batch
        max_length = max([feature.shape[-1] for feature in features])

        # Initialize tensors for padded features and labels
        padded_features = torch.zeros(
            len(batch), features[0].shape[0], max_length)
        labels_tensor = torch.zeros(len(batch), dtype=torch.long)

        # Pad each feature and copy into the padded tensor
        for i, feature in enumerate(features):
            length = feature.shape[-1]
            padded_features[i, :, :length] = feature

        # Copy labels to the label tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return padded_features, labels_tensor

def train(model, train_loader, test_loader, num_epochs, optimizer, scheduler, device, model_save_path, GTR):
    # set up loss function with weighted loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # create a SummaryWriter for TensorBoard logging
    summary_writer = SummaryWriter(log_dir=f'logs/{GTR}')

    # train the model
    for epoch in range(num_epochs):
        # train the model
        model.set_training(True)
        for wave_feat, labels in train_loader:
            # set to device
            wave_feat = wave_feat.to(device)
            labels = labels.to(device)

            # get the output
            outputs = model(wave_feat)

            # calculate the loss
            loss = loss_fn(outputs, labels)

            # backprop
            loss.backward()

            # update the parameters
            optimizer.step()

            # zero grad
            optimizer.zero_grad()

            # Write loss to TensorBoard
            iteration = epoch * len(train_loader) + len(train_loader)
            summary_writer.add_scalar('Loss/train', loss.item(), iteration)

        # validate the model
        model.set_training(False)
        with torch.no_grad():
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            for wave_feat, labels in test_loader:
                # set to device
                wave_feat = wave_feat.to(device)
                labels = labels.to(device)

                # get the output
                outputs = model(wave_feat)

                # get the predicted labels
                _, predicted = torch.max(outputs.data, 1)

                # get the accuracy
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            # calculate the f1 score
            f1 = f1_score(all_labels, all_preds, average='macro')
            # calculate the precision
            precision = precision_score(all_labels, all_preds, average='macro')
            # calculate the recall
            recall = recall_score(all_labels, all_preds, average='macro')
            # Write f1 score to TensorBoard
            summary_writer.add_scalar('F1/test', f1, epoch)
            # Write precision to TensorBoard
            summary_writer.add_scalar('Precision/test', precision, epoch)
            # Write recall to TensorBoard
            summary_writer.add_scalar('Recall/test', recall, epoch)
            
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
                


            # Write accuracy to TensorBoard
            summary_writer.add_scalar('Accuracy/test', accuracy, epoch)

        # save the model every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(
                model_save_path, f"model_{epoch}.pth"))

        # update the learning rate
        scheduler.step()

    # Close the SummaryWriter
    summary_writer.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/storageNVME/kcriss/gtr-145")
    parser.add_argument("--model_path", type=str,
                        default="TencentGameMate/chinese-hubert-large")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--model_save_path", type=str,
                        default="/home/kcriss/artivoice/hubert_gtr/models")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--extract_features", type=bool, default=True)
    parser.add_argument("--feature_path", type=str,
                        default="/storageNVME/kcriss/gtr-145-features")
    parser.add_argument("--gtr", type=str, default="T")
    parser.add_argument('--device', type=int, choices=[0,1,2,3], default=0)
    args = parser.parse_args()

    feature_extractor = FeatureExtractor()
    
    if args.extract_features:
        extract_features(args.model_path, args.data_path, args.feature_path, feature_extractor)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if args.gtr == 'G':
        num_classes = 5
    elif args.gtr == 'T':
        num_classes = 5
    elif args.gtr == 'R':
        num_classes = 8
    # get the feature paths
    # check the existence of the feature path
    if not os.path.exists(args.feature_path):
        raise ValueError(
            "The feature path does not exist! Need to first do extract features!")

    train_dataset = AudioDataset(
        file_path=args.feature_path, GTR=args.gtr, mode='train')
    test_dataset = AudioDataset(
        file_path=args.feature_path, GTR=args.gtr, mode='test')

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=AudioDataset.collation_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=AudioDataset.collation_fn)

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    device = torch.device(f"cuda:{args.device}")
    # set model
    # model = GTRClassifier(model_path=args.model_path,
    #                         num_classes=args.num_classes)
    model = GTRRegressor(model_path=args.model_path)

    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)

    # set model to device
    model = model.to(device)

    # train the model
    train_reg(model, train_dataloader, test_dataloader, args.num_epochs,
          optimizer, scheduler, device, args.model_save_path, GTR=args.gtr)
