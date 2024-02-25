# feature extractor class
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr

    def extract_features(self, y, sr=22050):
        # y, sr = librosa.load(self.audio_path, sr=self.sr)
        
        # extract mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # extract mel
        mel = librosa.feature.melspectrogram(y=y, sr=sr)

        # extract contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # extract spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        # extract spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # extract spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        spec_rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)

        # extract pitch(f0) from time series
        f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                     fmin=librosa.note_to_hz('C2'),
                                                     fmax=librosa.note_to_hz('C7'))
        f0 = f0[np.newaxis, :]
        voiced_flag = voiced_flag[np.newaxis, :]
        voiced_probs = voiced_probs[np.newaxis, :]
        
        # extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
 
        # extract flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        
        # concatenate all features
        features = np.concatenate((mfcc, mel, contrast, spec_cent, spec_bw, spec_rolloff, spec_rolloff_min, f0, voiced_flag, voiced_probs, zcr, flatness), axis=0)
        
        return features