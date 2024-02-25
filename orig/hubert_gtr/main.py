
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
import soundfile as sf
from pathlib import Path
import numpy as np
import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from model import GTRClassifier, GTRRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)


def extract_features(model_path, data_path, feature_path, feature_extractor=None):

    # check the existence of the feature path
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    if feature_extractor is None:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_path)

    # list all files under the data path
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            # 遍历子文件夹中的所有文件
            for filename in os.listdir(folder_path):
                if not filename.endswith('.wav'):
                    continue
                # 获取完整的文件路径
                wav_path = os.path.join(folder_path, filename)

                # 使用torchaudio读取wav文件并降采样到16kHz
                wav, sr = torchaudio.load(wav_path)
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        # get the input values
        input_values = feature_extractor(
            wav, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0)

        # convert pt to np
        input_values = input_values.numpy()

        # save the input values
        np.save(os.path.join(feature_path, filename), input_values)


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

def train_reg(model, train_loader, test_loader, num_epochs, optimizer, scheduler, device, model_save_path, GTR):
    # set up loss function as regression
    loss_fn = torch.nn.MSELoss()

    # create a SummaryWriter for TensorBoard logging
    summary_writer = SummaryWriter(log_dir=f'logs/{GTR}')

    # train the model
    for epoch in range(num_epochs):
        # train the model
        model.set_training(True)
        for wave_feat, labels in train_loader:
            # set to device
            wave_feat = wave_feat.to(device)
            labels = labels.to(device).float()

            # get the output
            outputs = model(wave_feat).float()

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
                outputs = model(wave_feat).squeeze(1)

                # get predicted output and find nearest integer
                predicted = torch.round(outputs)

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

            # get the confusion matrix of the predictions
            confusion_matrix = pd.crosstab(pd.Series(all_labels), pd.Series(all_preds), rownames=['True'], colnames=['Predicted'], margins=True)

            plt.figure(figsize=(10, 7))
            sns.heatmap(confusion_matrix, annot=True, fmt='g')
            plt.title('Confusion Matrix')
            fig = plt.gcf()

            # 使用TensorBoard记录图表
            summary_writer.add_figure('Confusion Matrix', fig, epoch)

            # 清除当前图表，避免重叠绘制
            plt.clf()

            
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

    if args.extract_features:
        extract_features(args.model_path, args.data_path, args.feature_path)
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
