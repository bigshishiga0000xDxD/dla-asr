from abc import abstractmethod
from torch import Tensor

class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


class ErrorRateMetric(BaseMetric):
    def __init__(self, text_encoder, text_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
    
    @abstractmethod
    def _error_fn(self, target_text, predictred_text):
        raise NotImplementedError()

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        errs = []
        pred_texts = self.text_decoder.decode(log_probs, log_probs_length)
        for pred_text, target_text in zip(pred_texts, text):
            target_text = self.text_encoder.normalize_text(target_text)
            errs.append(self._error_fn(target_text, pred_text))
        return sum(errs) / len(errs)