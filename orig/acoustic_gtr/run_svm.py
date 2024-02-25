import pdb

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import re
from tqdm import tqdm
import glob
import multiprocess

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def process_file(wav_file):
    # 从文件路径中提取数字
    digits = re.findall(r'\d+', wav_file)[-2]
    try:
        G = int(digits[-3])
        T = int(digits[-2])
        R = int(digits[-1])

        # 提取特征
        extractor = FeatureExtractor()
        feature = extractor.extract_features(audio_path=wav_file)

        # feature_file_path = os.path.join(features_folder_path, os.path.basename(wav_file) + '.npy')
        # np.save(feature_file_path, feature)  # 保存特征到文件

        return (G, T, R, feature, f'Processed {wav_file}_G{G}_T{T}_R{R}')
    except Exception as e:
        return None, None, None, None, f"Error processing {wav_file}: {e}"


class FeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                     fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'),
                                                     fill_na=0.0)
        f0 = f0[np.newaxis, :]
        voiced_flag = voiced_flag[np.newaxis, :]
        voiced_probs = voiced_probs[np.newaxis, :]


        features = np.concatenate((f0, voiced_probs), axis=0)
        # Aggregate features
        # features = np.nan_to_num(features, nan=0.0)
        # features[~np.isfinite(features)] = 0
        features = np.vstack((np.mean(features, axis=1))).flatten()

        return features





if __name__ == "__main__":
    # Set path
    base_folder_path = '/Users/yizhong/Documents/projects/artivoice/splits_batch1/'
    # output_dir = 'acoustic_gtr/features'
    # os.makedirs(output_dir, exist_ok=True)
    # Get audio files
    wav_files = glob.glob(os.path.join(base_folder_path, '**/*.wav'), recursive=True)
    # Extract features

    labels_G = []
    labels_T = []
    labels_R = []
    features = []

    with multiprocess.Pool() as pool:
        results = list(tqdm(pool.imap(process_file, wav_files), total=len(wav_files)))


    for G, T, R, feature, message in results:
        if feature is not None:
            labels_G.append(G)
            labels_T.append(T)
            labels_R.append(R)
            features.append(feature)
        #     print(message)
        # else:
        #     print(message)

    features = np.array(features)
    labels_G = np.array(labels_G)
    labels_T = np.array(labels_T)
    labels_R = np.array(labels_R)

    X_train, X_test, y_train, y_test = train_test_split(features, labels_R, test_size=0.2, random_state=32)

    clf = make_pipeline(SVC(C=0.1, gamma=0.001, kernel='rbf'))
    # clf = make_pipeline(SVC())
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    print("Train set report:")
    print(classification_report(y_train, y_train_pred))

    # 在测试集上进行预测
    y_test_pred = clf.predict(X_test)

    # 测试集评估
    print("Test set report:")
    print(classification_report(y_test, y_test_pred))

