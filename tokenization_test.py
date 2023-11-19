from transformers import AutoTokenizer, LongT5Model
import torch


tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-large")
model = LongT5Model.from_pretrained("google/long-t5-tglobal-large")

encoder_input_ids = tokenizer("Translate English to French: Hello, my dog is cute", return_tensors="pt").input_ids
decoder_input_ids = tokenizer("<s>", return_tensors="pt").input_ids

with torch.no_grad():
    model.eval()
    outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)


last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)




