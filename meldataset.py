#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import os
import os.path as osp
import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_tones = "12345 X"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_tones)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print("Key Error: ", text)
        return indexes

class TextCleaner4Tones:
    def __init__(self, dummy=None):
        self.word_index_dictionary = {
            "X": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            " ": 6,
        }
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print("Key Error: ", text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        #_data_list = [l[:-1].split('|') for l in data_list]
        _data_list = [l.split('|') for l in data_list]
        self.data_list = [data if len(data) == 4 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.text_cleaner4tones = TextCleaner4Tones()
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

#         self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, tone_tensor, speaker_id = self._load_tensor(data)
        # wave, text_tensor, speaker_id = self._load_tensor(data)

        mel_tensor = preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        # print(text_tensor.shape, text_tensor)
        # print(tone_tensor.shape, tone_tensor)
        return speaker_id, acoustic_feature, text_tensor, tone_tensor, path

    def _load_tensor(self, data):
        wave_path, text, tone, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            # print(wave_path, sr)

        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        text = self.text_cleaner(text)
        # tone = self.text_cleaner4tones(tone)
        tone = self.text_cleaner(tone)

        text.insert(0, 0)
        text.append(0)

        tone.insert(0, 0)
        tone.append(0)

        text = torch.LongTensor(text)
        tone = torch.LongTensor(tone)

        return wave, text, tone, speaker_id
        # return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, tone, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # speaker_id, acoustic_feature, text_tensor, tone_tensor, path
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        tones = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        # paths = ['' for _ in range(batch_size)]
        paths = torch.zeros(batch_size).long()  # gtr conditioning as int

        
        for bid, (label, mel, text, tone, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            tones[bid, :text_size] = tone
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            
            if self.return_wave:
                paths[bid] = int(path.split("/")[-2])

            
        if self.return_wave:
            return paths, texts, input_lengths, mels, output_lengths, tones
        
        return texts, input_lengths, mels, output_lengths, tones



def build_dataloader(path_list,
                     validation=False,
                     num_workers=1,
                     device='cpu',
                     batch_size=8,
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'),)

    return data_loader
