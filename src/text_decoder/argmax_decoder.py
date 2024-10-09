from torch import Tensor
from src.text_decoder import BaseDecoder
from src.text_encoder.text_encoder import TextEncoder

class ArgmaxDecoder(BaseDecoder):
    def __init__(self, encoder: TextEncoder):
        super().__init__(encoder)
    
    def _ctc_decode(self, inds) -> str:
        stack = []
        for ind in inds:
            if not stack or stack[-1] != ind:
                stack.append(ind)
        return self.encoder.decode(stack)
    
    def decode(self, log_probs: Tensor, log_probs_length: Tensor, **batch) -> list[str]:
        batch_size = log_probs.shape[0]
        output = []

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995

            argmax_inds = log_probs[i, :log_probs_length[i]].clone().cpu().argmax(-1).numpy()
            argmax_text = self._ctc_decode(argmax_inds)

            output.append(argmax_text)
        
        return output
