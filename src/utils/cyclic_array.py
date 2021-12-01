import numpy as np

# TODO: rewrite to hold a max size so it doesnt need to rotate as often, still return arrays of the configured length
#  needed because np.roll takes forever
# all operations are O(1) and don't require copying the array
# except to_array which has to copy the array and is O(n)
class CyclicArray:
    def __init__(self, length: int, dimensions: int = 1, initial_array: np.array = None):
        # allocate the memory we need ahead of time
        self.max_length: int = length
        self.queue_tail: int = length - 1
        self.dimensions = dimensions
        if not initial_array:
            if dimensions == 1:
                shape = (length,)
            else:
                shape = (length, dimensions)

            self.queue = np.zeros(shape, dtype=float)
            return

        initial_array_length = initial_array.shape[0]
        if initial_array_length == length:
            self.queue = np.array(initial_array, dtype=float)
        elif initial_array_length > length:
            self.queue = np.array(initial_array[initial_array_length - length:], dtype=float)
        else:
            if dimensions == 1:
                zeros_shape = (length - initial_array_length,)
            else:
                zeros_shape = (length - initial_array_length, dimensions)
            self.queue = np.append(np.array(initial_array, dtype=np.int64),
                                   np.zeros(zeros_shape, dtype=float))
            self.queue_tail = initial_array_length - 1

    def size(self) -> int:
        return self.max_length

    def to_array(self) -> np.array:
        head = (self.queue_tail + 1) % self.max_length
        return np.roll(self.queue, -head)  # this will force a copy

    def enqueue(self, new_data: np.array) -> None:
        # move tail pointer forward then insert at the tail of the queue
        # to enforce max length of recording
        self.queue_tail = (self.queue_tail + 1) % self.max_length
        self.queue[self.queue_tail] = new_data

    def enqueue_n(self, new_data: np.array) -> None:
        # move tail pointer forward then insert at the tail of the queue
        # to enforce max length of recording
        new_data = new_data[-self.max_length:]
        enqueue_length = new_data.shape[0]
        new_tail = (self.queue_tail + enqueue_length) % self.max_length
        if new_tail == self.queue_tail:
            # It can replace the whole queue
            self.queue = new_data
            self.queue_tail = self.max_length - 1
        elif new_tail > self.queue_tail or (self.queue_tail + 1) % self.max_length == 0:
            # We haven't wrapped around or we start at 0
            self.queue[(self.queue_tail + 1) % self.max_length: new_tail + 1] = new_data
            self.queue_tail = new_tail
        else:
            # Handle wrapping
            numb_append_to_end = self.max_length-(self.queue_tail + 1)
            self.queue[self.queue_tail + 1:] = new_data[:numb_append_to_end]
            self.queue[:new_tail + 1] = new_data[numb_append_to_end:]
            self.queue_tail = new_tail

        # self.queue[self.queue_tail] = new_data

    def head(self):
        queue_head = (self.queue_tail + 1) % self.max_length
        return self.queue[queue_head]

    def tail(self):
        return self.queue[self.queue_tail]

    def replace_item_at(self, index: int, new_value: float):
        loc = (self.queue_tail + 1 + index) % self.max_length
        self.queue[loc] = new_value

    def item_at(self, index: int) -> int:
        # the item we want will be at head + index
        loc = (self.queue_tail + 1 + index) % self.max_length
        return self.queue[loc]

    def __repr__(self):
        return "tail: " + str(self.queue_tail) + "\narray: " + str(self.queue)

    def __str__(self):
        return "tail: " + str(self.queue_tail) + "\narray: " + str(self.queue)


# Testing
# hello = CyclicArray(6, dimensions=2)
# hello.enqueue([1, 5])
# hello.enqueue([2, 3])
# hello.enqueue_n(np.array([[1, 2], [4, 5]]))
# hello.enqueue_n(np.array([[11, 22], [44, 55]]))
# hello.enqueue_n(np.array([[111, 222], [444, 555]]))
#
# hello.enqueue_n(np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]))
# hello.enqueue_n(np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]))
# hello.enqueue_n(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
# print(hello)
# # Test wrapping:
# hello.enqueue_n(np.array([[11, 11], [22, 22], [33, 33], [44, 44]]))
#
# print(hello)
