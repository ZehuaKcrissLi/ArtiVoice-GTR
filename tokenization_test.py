from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, AutoProcessor

# 加载 SpeechT5 的 tokenizer
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
processor = AutoProcessor.from_pretrained("GCYY/speecht5_finetuned_fleurs_zh_4000")
# model = SpeechT5ForTextToSpeech.from_pretrained("wuula/speecht5_tts_common_voice_zh")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("GCYY/speecht5_finetuned_fleurs_zh_4000")
# 尝试对一些中文文本进行 tokenization
tokenizer = processor.tokenizer
tokenizer_vocab = {k for k,_ in tokenizer.get_vocab().items()}

# print(tokenizer_vocab)




sample_text = "从错误信息来看，你遇到的问题是尝试使用一个不同类型的tokenizer加载一个预训练模型。"
tokens = tokenizer.tokenize(sample_text)

# 输出 tokenized 的结果
print(tokens)




