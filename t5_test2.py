from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch

class CustomT5Model(T5ForConditionalGeneration):
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
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Setup model config to output all hidden states
config = T5Config.from_pretrained("google/flan-t5-base")
config.output_hidden_states = True

# Create the custom model and load the pre-trained weights
model = CustomT5Model.from_pretrained("google/flan-t5-base", config=config).to("cuda")

# Prepare the input data
input_text = "I like cheese"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids'].to("cuda")
decoder_input_ids = torch.zeros_like(input_ids).to("cuda")  # Create an all-zero tensor for decoder_input_ids

# Run the model
with torch.no_grad():
    model.eval()
    encoder_activations, decoder_activations = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

# Output the shape of activations for each layer
for i, act in enumerate(encoder_activations):
    print(f"Encoder layer {i} activations shape: {act.shape}")

for i, act in enumerate(decoder_activations):
    print(f"Decoder layer {i} activations shape: {act.shape}")
