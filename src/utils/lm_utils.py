from torchaudio.models.decoder import download_pretrained_files
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def get_kenlm_path(model_name):
    files = download_pretrained_files(model_name)
    return files.lm

def get_unigrams(model_name):
    files = download_pretrained_files(model_name)
    unigrams = open(files.lexicon).readlines()
    unigrams = [
        unigram.split('\t')[0]
        for unigram in unigrams
    ]
    return unigrams

class LanguageModel:
    def __init__(
        self,
        device=None,
        model_id = "openai-community/gpt2-large"
    ):
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    def __call__(self, text: str) -> float:
        tokens = self.tokenizer(text).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        with torch.no_grad():
            inputs = torch.LongTensor(tokens[:-1]).to(self.device)
            labels = torch.LongTensor(tokens[1:]).to(self.device)
            outputs = self.model(inputs, labels=labels)
        
        return -outputs.loss.item()
