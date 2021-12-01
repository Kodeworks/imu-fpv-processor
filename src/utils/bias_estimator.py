import numpy as np

import config as cfg
from utils.adaptive_moving_average import AdaptiveMovingAverage


# TODO: handle this
# def update_counters_on_buffer_reuse(self):

class BiasEstimator:
    def __init__(self, points_between_updates: int, expected_value: int = 0.0, use_moving_average: bool = False,
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

    # TODO: data sent in is ensured to be continuous
    # cyclic buffer is dealt with outside of the bias estimator
    def update(self, measurements: np.array):
        # Only update bias if enough data has arrived
        if self.points_since_update >= self.points_between_updates:
            self.points_since_update = 0
            bias_sliding_window = np.nanmean(measurements, axis=0) - self.expected_value
            if self.use_moving_average:
                self.adaptive_bias.update(bias_sliding_window)
                self.bias = self.adaptive_bias.get_state()
            else:
                self.bias = bias_sliding_window
        else:
            self.points_since_update += measurements.shape[0]

    def value(self):
        return self.bias
