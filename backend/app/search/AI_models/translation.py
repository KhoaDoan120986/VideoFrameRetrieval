from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

class Translator:
    def __init__(self, model_name: str = "VietAI/envit5-translation", device: str = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _clean_prefix(self, text: str) -> str:
        return re.sub(r"^(en|vi)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    
    def translate(self, text: str, source_lang: str = "en", max_length: int = 128) -> str:
        content = f"{source_lang}: {text}"
        inputs = self.tokenizer(
            content, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return self._clean_prefix(decoded)
