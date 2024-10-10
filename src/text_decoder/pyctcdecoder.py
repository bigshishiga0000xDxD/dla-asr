from torch import Tensor
from src.text_decoder import BaseDecoder
from src.text_encoder import TextEncoder

from pyctcdecode import build_ctcdecoder

class PyCTCDecoder(BaseDecoder):
    def __init__(self, encoder: TextEncoder):
        super().__init__(encoder)

        self.decoder = build_ctcdecoder(
            labels=encoder.vocab
        )
    
    def decode(self, log_probs: Tensor, log_probs_length: Tensor, **batch) -> list[str]:
        output = []
        for log_probs, log_probs_length in zip(log_probs, log_probs_length):
            log_probs = log_probs[:log_probs_length]
            output.append(self.decoder.decode(log_probs.detach().cpu().numpy()))
        return output
