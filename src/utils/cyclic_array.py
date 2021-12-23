import numpy as np


class CyclicArray:
    def __init__(self, length: int, dimensions: int = 1, initial_array: np.array = None):
        # allocate the memory we need ahead of time
        self.max_length: int = length
        self.tail: int = length - 1
        self.wrapped = False
        self.dimensions = dimensions
        if not initial_array:
            if dimensions == 1:
                shape = (length,)
            else:
                shape = (length, dimensions)

            self.queue = np.zeros(shape=shape, dtype=float)
            return

        initial_array_length = initial_array.shape[0]
        if initial_array_length == length:
            self.queue = np.array(array=initial_array, dtype=float)
        elif initial_array_length > length:
            self.queue = np.array(array=initial_array[initial_array_length - length:], dtype=float)
        else:
            if dimensions == 1:
                zeros_shape = (length - initial_array_length,)
            else:
                zeros_shape = (length - initial_array_length, dimensions)
            self.queue = np.append(np.array(array=initial_array, dtype=float),
                                   np.zeros(shape=zeros_shape, dtype=float))
            self.tail = initial_array_length - 1

    def size(self) -> int:
        return self.max_length

    def get_latest_n(self, n) -> np.array:
        n = min(n, self.max_length)
        if not self.wrapped:
            head = max(self.tail - n, 0)
            return self.queue[head:self.tail + 1]

        if self.tail < n:
            start_arr = self.queue[-(n - self.tail):]
            end_arr = self.queue[:self.tail]
            return np.concatenate([end_arr, start_arr])
        else:
            return self.queue[self.tail - n:self.tail]

    def enqueue(self, new_data: np.array) -> None:
        if self.tail + 1 == self.max_length:
            self.wrapped = True
        self.tail = (self.tail + 1) % self.max_length
        self.queue[self.tail] = new_data

    def enqueue_n(self, new_data: np.array) -> None:
        # move tail pointer forward then insert at the tail of the queue
        # to enforce max length of recording
        new_data = new_data[-self.max_length:]
        enqueue_length = new_data.shape[0]
        new_tail = (self.tail + enqueue_length) % self.max_length
        if new_tail == self.tail:
            # It can replace the whole queue
            self.queue = new_data
            self.tail = self.max_length - 1
            self.wrapped = False
        elif new_tail > self.tail or (self.tail + 1) % self.max_length == 0:
            # We haven't wrapped around or we start at 0
            if (self.tail + 1) % self.max_length == 0:
                self.wrapped = True

            self.queue[(self.tail + 1) % self.max_length: new_tail + 1] = new_data
            self.tail = new_tail
        else:
            # Handle wrapping
            self.wrapped = True
            numb_append_to_end = self.max_length - (self.tail + 1)
            self.queue[self.tail + 1:] = new_data[:numb_append_to_end]
            self.queue[:new_tail + 1] = new_data[numb_append_to_end:]
            self.tail = new_tail

    def get_head(self):
        if self.wrapped:
            head = (self.tail + 1) % self.max_length
        else:
            head = 0
        return self.queue[head]

    def get_tail(self):
        return self.queue[self.tail]

    def replace_item_at(self, index: int, new_value: float):
        corrected_index = (self.tail + 1 + index) % self.max_length
        self.queue[corrected_index] = new_value

    def item_at(self, index: int) -> int:

        corrected_index = (self.tail + 1 + index) % self.max_length
        return self.queue[corrected_index]

    def __repr__(self):
        return "tail index: " + str(self.tail) + "\narray: " + str(self.queue)

    def __str__(self):
        return "tail index: " + str(self.tail) + "\narray: " + str(self.queue)
