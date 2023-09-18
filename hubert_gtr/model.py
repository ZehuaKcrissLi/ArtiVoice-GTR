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

class GTRClassifier_T(nn.Module):
    def __init__(self, num_classes):
        super(GTRClassifier_T, self).__init__()
        self.hubert_model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-large")
        
        # Define weights for the weighted sum of the outputs of all layers
        self.layer_weights = nn.Parameter(torch.rand(13), requires_grad=True)
        
        # Define a series of linear layers with decreasing dimensions
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Define dropout layers for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Get outputs of all layers from Hubert
        outputs = self.hubert_model(x, output_hidden_states=True).hidden_states
        
        # Create a weighted sum of the outputs of all layers
        weighted_sum = sum(w * output for w, output in zip(self.layer_weights, outputs))
        
        # Average pool over the time dimension
        avg_pooled = weighted_sum.mean(dim=1)
        
        # Pass through the linear layers with dropout and batch normalization
        x = self.fc1(avg_pooled)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x


class GTRClassifier_R(nn.Module):
    def __init__(self, num_classes):
        super(GTRClassifier_R, self).__init__()
        self.hubert_model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-large")
        
        # Define weights for the weighted sum of the outputs of all layers
        self.layer_weights = nn.Parameter(torch.rand(13), requires_grad=True)
        
        # Define a series of linear layers with decreasing dimensions
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Define dropout layers for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Get outputs of all layers from Hubert
        outputs = self.hubert_model(x, output_hidden_states=True).hidden_states
        
        # Create a weighted sum of the outputs of all layers
        weighted_sum = sum(w * output for w, output in zip(self.layer_weights, outputs))
        
        # Average pool over the time dimension
        avg_pooled = weighted_sum.mean(dim=1)
        
        # Pass through the linear layers with dropout and batch normalization
        x = self.fc1(avg_pooled)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x
