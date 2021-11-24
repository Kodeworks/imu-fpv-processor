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
