import numpy as np

import config as cfg
from utils.adaptive_moving_average import AdaptiveMovingAverage


class BiasEstimator:
    def __init__(self, points_between_updates: int, expected_value: float = 0.0, allow_nan=False,
                 use_moving_average: bool = False,
                 track_bias: bool = False):

        self.use_moving_average = use_moving_average
        # Current best estimate
        self.bias: float = 0

        if use_moving_average:
            self.adaptive_bias = AdaptiveMovingAverage(0.01, cfg.adaptive_alpha_max, cfg.adaptive_alpha_gain,
                                                       track_bias)
            self.adaptive_bias.update(self.bias)

        self.expected_value = expected_value
        self.points_since_update = 0
        self.points_between_updates = points_between_updates

        if allow_nan:
            self.mean = np.nanmean
        else:
            self.mean = np.mean

    def update_counter(self, n_new_measurements):
        self.points_since_update += n_new_measurements

    def should_update(self):
        return self.points_since_update >= self.points_between_updates

    def update(self, measurements: np.array):
        self.points_since_update = 0
        bias_sliding_window = self.mean(measurements, axis=0) - self.expected_value
        if self.use_moving_average:
            self.adaptive_bias.update(bias_sliding_window)
            self.bias = self.adaptive_bias.get_state()
        else:
            self.bias = bias_sliding_window

    def value(self):
        return self.bias
