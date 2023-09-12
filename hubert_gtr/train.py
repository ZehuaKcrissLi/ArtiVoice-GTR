import torch

def train_model(train_loader, model, classifier, optimizer, criterion, device):
    model.eval() 
    classifier.train() 
    
    for batch in train_loader:
        inputs, _ = batch
        inputs = inputs.to(device).half()
        
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

