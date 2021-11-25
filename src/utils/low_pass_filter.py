import numpy as np
from scipy.signal import filtfilt


class LowPassFilter:
    """
    usage:
    array_to_filter = LowPassFilter.get_interval_with_min_size(self.processed_input, self.start, self.end, cfg.min_filter_size, self.last_valid)
    filtered_array = LowPassFilter.process(array_to_filter, self.low_a, self.low_b, self.end-self.start)
    """
    @staticmethod
    def process(arr: np.array, low_a, low_b, n_return_vals):
        """
                Filters each of the 6 DOFs using a low-pass filter.
                :param arr: array to be lowpass filtered
                :param low_a: butterworth value?
                :param low_b: butterworth value?
                :n_return_vals: slice of the array we wish to use
                """
        if arr.ndim == 1:
            arr = np.array(filtfilt(low_b, low_a, arr))
        else:
            for dof_i in range(arr.shape[1]):
                arr[:, dof_i] = np.array(filtfilt(low_b,  low_a, arr[:, dof_i]))

        return arr[-n_return_vals:]



