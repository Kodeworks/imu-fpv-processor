import numpy as np
import src.config as cfg
from src.utils.adaptive_moving_average import AdaptiveMovingAverage
from src.utils.mem_map_utils import MemMapUtils


# TODO: gravitational constant will probably be included in this bias

# TODO: handle this
# def update_counters_on_buffer_reuse(self):

class BiasEstimator:
    def __init__(self, points_between_updates: int, expected_value: int = 0.0, use_moving_average: bool = False,
                 track_bias: bool = False, bias_tracking_length: int = 0):
        # Biases for each timestep are kept for examination
        self.track_bias = track_bias
        if track_bias:
            self.bias_array = np.zeros(shape=(bias_tracking_length,), dtype=float)
            self.number_of_biases = 0

        self.use_moving_average = use_moving_average
        # Current best estimate
        # TODO: should these be sent in so they can be tuned individually?
        if use_moving_average:
            self.adaptive_bias = AdaptiveMovingAverage(0.01, cfg.adaptive_alpha_max, cfg.adaptive_alpha_gain,
                                                       track_bias)
        self.bias = 0

        self.expected_value = expected_value
        self.last_bias_row = -1  # TODO: sometimes initialized to 0
        self.points_between_updates = points_between_updates

    # TODO: data sent in is ensured to be continuous
    # cyclic buffer is dealt with outside of the bias estimator
    def update(self, measurements: np.array, row_no: int):
        # Only update bias if enough data has arrived
        if row_no - self.last_bias_row >= self.points_between_updates:
            bias_sliding_window = np.nanmean(measurements) - self.expected_value
            if self.use_moving_average:
                self.adaptive_bias.update(bias_sliding_window)
                self.bias = self.adaptive_bias.get_state()
            else:
                self.bias = bias_sliding_window

            self.last_bias_row += row_no
            if self.track_bias:
                self.bias_array[self.number_of_biases] = 0
                self.number_of_biases += 1

    def value(self):
        return self.bias

    def update_counter(self, last_valid: int):
        self.last_bias_row = self.last_bias_row - last_valid - 1


"""
Everything below this point is the code to rewrite
"""
