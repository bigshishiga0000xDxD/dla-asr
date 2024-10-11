from torch import Tensor
from src.text_decoder import BaseDecoder
from src.text_encoder import TextEncoder

from pyctcdecode import build_ctcdecoder


class PyCTCDecoder(BaseDecoder):
    def __init__(
        self,
        encoder: TextEncoder,
        beam_size=100,
        light_lm_path=None,
        unigrams=None,
        heavy_lm=None,
        alpha=0.5,
        beta=1.5,
        n_best=3
    ):
        super().__init__(encoder)

        self.decoder = build_ctcdecoder(
            labels=encoder.vocab,
            kenlm_model_path=light_lm_path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta
        )
        self.beam_size = beam_size
        self.heavy_lm = heavy_lm
        self.beta = beta
        self.n_best = n_best
    
    def decode(self, log_probs: Tensor, log_probs_length: Tensor, **batch) -> list[str]:
        if self.heavy_lm:
            return self._decode_beams(log_probs, log_probs_length)
        else:
            return self._decode(log_probs, log_probs_length)

    def _decode_beams(self, log_probs: Tensor, log_probs_length: Tensor) -> list[str]:
        output = []
        for log_probs, log_probs_length in zip(log_probs, log_probs_length):
            log_probs = log_probs[:log_probs_length]
            hyps = [
                (beam[0], beam[3])
                for beam in self.decoder.decode_beams(
                    log_probs.detach().cpu().numpy(),
                    beam_width=self.beam_size
                )[:self.n_best]
            ]

            hyps = [
                (hyp, score + self.beta * self.heavy_lm(hyp))
                for hyp, score in hyps
            ]

            output.append(max(hyps, key=lambda p: p[1])[0])
        return output

    def _decode(self, log_probs: Tensor, log_probs_length: Tensor) -> list[str]:
        output = []
        for log_probs, log_probs_length in zip(log_probs, log_probs_length):
            log_probs = log_probs[:log_probs_length]
            output.append(self.decoder.decode(
                log_probs.detach().cpu().numpy(),
                beam_width=self.beam_size
            ))
        return output
