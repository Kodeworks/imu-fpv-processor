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

    # TODO: move to another class
    @staticmethod
    def get_interval_with_min_size(arr: np.array, start: int, end: int, min_size: int, end_index: int):
        """
               Hides the complexity of accessing our cyclic buffer and returns an array of the requested size at the requested location.
               :param arr: the array to access
               :param start: Index of first row to get slice from
               :param end: Index of last row to get slice from
               :param min_size: minimum size of array returned, goes backwards from end
               :param end_index: the array hasn't necessarily used the space all the way until the end
               """
        if end < min_size:
            # this means that start is between 0 and end
            end_arr = arr[end_index-(min_size-end):end_index]
            start_arr = arr[:end]
            return np.concatenate([end_arr, start_arr])
        else:
            adjusted_start = min(start, end-min_size)
            return arr[adjusted_start:end]



