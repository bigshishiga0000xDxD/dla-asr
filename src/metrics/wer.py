from src.metrics.base_metric import ErrorRateMetric
from src.metrics.utils import calc_wer


class WERMetric(ErrorRateMetric):
    def _error_fn(self, *args, **kwargs):
        return calc_wer(*args, **kwargs)
