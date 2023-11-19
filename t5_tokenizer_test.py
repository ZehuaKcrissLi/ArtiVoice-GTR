from transformers import AutoTokenizer, LongT5Model, LongT5Config
import torch

class CustomLongT5Model(LongT5Model):
    def forward(self, *args, **kwargs):
        encoder_activations = []
        decoder_activations = []

        outputs = super().forward(
            input_ids=kwargs['input_ids'],
            decoder_input_ids=kwargs['decoder_input_ids'],
            output_hidden_states=True
        )

        encoder_activations = outputs.encoder_hidden_states
        decoder_activations = outputs.decoder_hidden_states

        return encoder_activations, decoder_activations

# Load the tokenizer and pre-trained model
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-large")
pretrained_model = LongT5Model.from_pretrained("google/long-t5-tglobal-large")

# Setup model config to output all hidden states
config = LongT5Config.from_pretrained("google/long-t5-tglobal-large")
config.output_hidden_states = True

# Create the custom model and load the pre-trained weights
model = CustomLongT5Model.from_pretrained("google/long-t5-tglobal-large", config=config)

# Encoder input 
inputs_ids = tokenizer("I like cheese", return_tensors="pt").input_ids

# Decoder input
decoder_input_ids = tokenizer("</s>", return_tensors="pt").input_ids

# Run the model
with torch.no_grad():
    model.eval()
    encoder_activations, decoder_activations = model(input_ids=inputs_ids, decoder_input_ids=decoder_input_ids)

# Output the shape of activations for each layer
for i, act in enumerate(encoder_activations):
    print(f"Encoder layer {i} activations shape: {act.shape}")
    print(f"Encoder layer {i} activations: {act}")
for i, act in enumerate(decoder_activations):
    print(f"Decoder layer {i} activations shape: {act.shape}")
    print(f"Decoder layer {i} activations: {act}")
