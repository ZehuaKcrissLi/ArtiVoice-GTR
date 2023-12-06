from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch.nn as nn
import torch

class GTRClassifier(nn.Module):
    def __init__(self, model_path, num_classes):
        super(GTRClassifier, self).__init__()
        
        self.hubert_model = HubertModel.from_pretrained(model_path)
        
        # get the number of layers from the hubert model
        num_layers = len(self.hubert_model.encoder.layers)
        
        # Define weights for the weighted sum of the outputs of all layers
        # self.layer_weights = nn.Parameter(torch.rand(num_layers), requires_grad=True)
        
        # Define a series of linear layers with decreasing dimensions
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Define dropout layers for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        
        # put the classifer into series
        self.classifier = nn.Sequential(
            self.fc1,
            self.bn1,
            nn.ReLU(),
            self.dropout,
            self.fc2
        )
    
    def set_training(self, flag):
        if flag:
            self.hubert_model.eval()
            self.classifier.train()
        else:
            self.eval()

    def forward(self, x):
        # Get outputs of all layers from Hubert

        outputs = self.hubert_model(x.squeeze(1), output_hidden_states=True).last_hidden_state
        
        
        # Create a weighted sum of the outputs of all layers use tensor operation
        # weighted_sum = torch.stack([w * output for w, output in zip(self.layer_weights, outputs)], dim=0).sum(dim=0)
        
        # Average pool over the time dimension
        # avg_pooled = weighted_sum.mean(dim=1) # (b, dim)
        avg_pooled = outputs.mean(dim=1)
        
        # Pass through the linear layers with dropout and batch normalization
        x = self.classifier(avg_pooled)
        
        return x


class GTRRegressor(nn.Module):
    def __init__(self, model_path):
        super(GTRRegressor, self).__init__()
        
        self.hubert_model = HubertModel.from_pretrained(model_path)
        
        # Define a series of linear layers with decreasing dimensions
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)  # Output dimension is 1 for regression
        
        # Define dropout layers for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        
        # put the regressor into series
        self.regressor = nn.Sequential(
            self.fc1,
            self.bn1,
            nn.ReLU(),
            self.dropout,
            self.fc2
        )
    
    def set_training(self, flag):
        if flag:
            self.hubert_model.eval()
            self.regressor.train()
        else:
            self.eval()

    def forward(self, x):
        # Get outputs of all layers from Hubert
        outputs = self.hubert_model(x.squeeze(1), output_hidden_states=True).last_hidden_state
        
        # Average pool over the time dimension
        avg_pooled = outputs.mean(dim=1)
        
        # Pass through the linear layers with dropout and batch normalization
        x = self.regressor(avg_pooled)
        
        return x