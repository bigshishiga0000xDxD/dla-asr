from collections import defaultdict

import torch
from torch import Tensor
from scipy.special import logsumexp
from src.text_decoder import BaseDecoder
from src.text_encoder.text_encoder import TextEncoder


class BeamSearchDecoder(BaseDecoder):
    def __init__(self, encoder: TextEncoder, beam_size):
        super().__init__(encoder)
        self.beam_size = beam_size

    def decode(self, log_probs: Tensor, log_probs_length: Tensor, **batch) -> list[str]:
        output = []

        for log_probs, log_probs_length in zip(log_probs, log_probs_length):
            log_probs = log_probs[:log_probs_length].detach().cpu()
            # state of beam is decoded string and the index of last token
            beams = {
                (
                    self.encoder.EMPTY_TOK,
                    self.encoder.char2ind[self.encoder.EMPTY_TOK],
                ): 0
            }

            # append dummy empty token token to merge beams with the same
            # decoded strings and different last tokens to the one beam in the end
            id_vector = torch.FloatTensor(log_probs.shape[-1]).fill_(-1e9)
            id_vector[self.encoder.char2ind[self.encoder.EMPTY_TOK]] = 0
            log_probs = torch.vstack((log_probs, id_vector.unsqueeze(0)))

            for logits in log_probs:
                new_beams = defaultdict(list)
                for beam, log_prob in beams.items():
                    for i, x in enumerate(logits):
                        # appended token is equal to last one in sequence
                        if i == beam[1]:
                            # beam doesn't change, probs are multiplied
                            new_beams[beam].append(log_prob + x)
                        else:
                            # add meaningful character to decoded string,
                            # probs are multiplied
                            string = beam[0] + self.encoder.ind2char[i]
                            new_beams[(string, i)].append(log_prob + x)

                # take sum over all probs for every beam
                new_beams = {
                    beam: logsumexp(log_probs) for beam, log_probs in new_beams.items()
                }

                # take top k beams
                new_beams = dict(
                    sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[
                        : self.beam_size
                    ]
                )

                beams = new_beams

            best_score = -1e9
            for beam, score in beams.items():
                if score > best_score:
                    best_string, best_score = beam[0], score

            output.append(best_string)

        return output
