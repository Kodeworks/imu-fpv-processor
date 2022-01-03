import numpy as np
import src.config as cfg


class NanHandling:
    @staticmethod
    def should_discard_burst(arr: np.array):
        return np.count_nonzero(np.isnan(arr[:, 0])) > int(arr.shape[0] * cfg.discard_burst_nan_threshold)

    @staticmethod
    def burst_contains_nan(arr: np.array):
        return np.any(np.isnan(arr[:, 0]))

    @staticmethod
    def nan_handling(arr: np.array, start: int, end: int):
        # Method uses assumption 1
        should_discard_burst = False
        burst_contains_nan = False
        if np.any(np.isnan(arr[start:end, 0])):
            burst_contains_nan = True
            # If in fact all values of any DoF are NaN, the entire processing of this burst should be handled in a
            # separate method
            nan_in_burst = np.count_nonzero(np.isnan(arr[start:end, 0]))
            if nan_in_burst > int((end - start) * cfg.discard_burst_nan_threshold):
                should_discard_burst = True

        return burst_contains_nan, should_discard_burst

    @staticmethod
    def interpolate_missing_values(arr: np.array):
        print("Interpolating for NaN values")
        # Initialize array with all NaN and non NaN values
        interpolated_arr = np.copy(arr)

        # get indices of all values that are not NaN
        indices = np.array(list([index for index, element in enumerate(arr) if
                                 not np.any(np.any(np.isnan(element)))]))

        # Handle NaN values before first value
        if indices[0] != 0:
            initial_value = np.array([0.0, 0.0, -1.0, 0.0, 0.0])  # Generic set of [accX, accY, accZ, gyroX, gyroY]
            interpolated_arr[0:indices[0]] = NanHandling.get_interpolated_values(initial_value, arr[indices[0]],
                                                                                 indices[0] + 1)
        # Copy the last valid value to the remaining part of the array
        if indices[-1] != arr.shape[0] - 1:
            interpolated_arr[indices[-1] + 1:] = arr[indices[-1]]

        # Fill all gaps between indices
        for start_index, end_index in np.dstack(indices[1:], indices[:-1]):
            if end_index == start_index + 1:
                continue
            # Linearly interpolate the values
            interpolated_arr[start_index + 1:end_index] = NanHandling.get_interpolated_values(arr[start_index],
                                                                                              arr[end_index],
                                                                                              end_index - start_index)

        return interpolated_arr

    @staticmethod
    def get_interpolated_values(start_value, end_value, steps):
        increment = (end_value - start_value) / steps
        return np.linspace(increment, increment * (steps - 1), steps - 1)
