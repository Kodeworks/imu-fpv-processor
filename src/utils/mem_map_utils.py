import numpy as np


class MemMapUtils:
    @staticmethod
    def historic_data_is_contiguous(request_size: int, end_row_of_data: int):
        """
        Used for knowing whether or not an entire array of recent data can be indexed by
        [self.last_row+1 - request_size, self.last_row+1] or if some part of the array needs to be collected from
        the end of the buffer.
        :param request_size: Length of historic data that needs to be accessed
        :param end_row_of_data: Index of first data row to not be included
        """
        if request_size <= end_row_of_data:
            return True
        return False

    @staticmethod
    def patched_buffer_indices(request_size: int, current_row: int, prev_last_index: int):
        """
        Returns two index pairs of the parts of the buffer that make up the most recent historic data of size
        request_size, by slicing the buffer.
        Historic data does not include the current burst.
        Assumes that the buffer is circular in the way that recent data stops at [self.last_valid] and picks up at
        [0], and has last data point at [self.last_row].

        :param request_size: Number of historic data points being requested.
        :param current_row: Index that points to the start of the current burst.
        :param prev_last_index: Index with the last valid value in the array

        :returns: Two pairs of indices that can be used directly to access the most recent data,
        ie. buffer[first_pair[0]:first_pair[1]]
        """
        # buffer_beginning is analogous to some variable request_end
        buffer_beginning = (0, current_row)
        # buffer_end is analogous to some variable request_beginning
        buffer_end = (prev_last_index + 1 - (request_size - current_row), prev_last_index + 1)

        # indices are returned in the order that data was written to the buffer
        return buffer_end, buffer_beginning

    @staticmethod
    def get_contiguous_array_from_buffer(buffer: np.array, start_row: int, end_row: int, requested_size: int,
                                         end_index: int):
        if MemMapUtils.historic_data_is_contiguous(requested_size, end_row):
            buffer_end_range, buffer_start_range = MemMapUtils.patched_buffer_indices(requested_size, start_row,
                                                                                      end_index)
            return np.concatenate(
                buffer[buffer_end_range[0]:buffer_end_range[1], buffer[buffer_start_range[0]:buffer_start_range[1]]])

    @staticmethod
    # TODO: investigate if these are the same lol
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