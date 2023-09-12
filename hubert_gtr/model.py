from transformers import HubertModel
import torch.nn as nn
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.hubert_model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-large")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-large")
        
    def forward(self, x):
        input_values = self.feature_extractor(x, return_tensors="pt").input_values
        outputs = self.hubert_model(input_values).last_hidden_state
        return self.linear(outputs)

