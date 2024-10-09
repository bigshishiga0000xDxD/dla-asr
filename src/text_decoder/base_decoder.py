from abc import abstractmethod

import torch

from src.text_encoder.text_encoder import TextEncoder

class BaseDecoder:
    def __init__(self, encoder: TextEncoder):
        """
        Args:
            encoder (TextEncoder): TextEncoder instance
        """
        self.encoder = encoder

    @abstractmethod
    def decode(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        **batch
    ) -> list[str]:
        """
        Base decoder method

        Args:
            log_probs (torch.Tensor): (B x L x C) tensor, where
                B is batch size,
                L is the maximum length of sequence,
                C is size of vocab
            log_probs_length (torch.Tensor): (B) tensor, where
                B is batch size
        Returns:
            texts (list[str]): list of length B with decoded texts
        """
        raise NotImplementedError()