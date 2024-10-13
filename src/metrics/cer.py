from src.metrics.base_metric import ErrorRateMetric
from src.metrics.utils import calc_cer

class CERMetric(ErrorRateMetric):
    def _error_fn(self, *args, **kwargs):
        return calc_cer(*args, **kwargs)
