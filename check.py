from transformers import BertForMaskedLM
from transformers import BertTokenizer
from transformers import pipeline

tokenizer = BertTokenizer.from_pretrained('./tokenizer/')
model = BertForMaskedLM.from_pretrained('./pretrained_models/' + 'checkpoint-6200/')

fill_mask = pipeline("fill-mask", tokenizer=tokenizer, model=model)
print(fill_mask('12 314 162 [MASK] 62 276'))  # '12 314 162 315 62 276'
