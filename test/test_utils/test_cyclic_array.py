import numpy as np

from src.utils.cyclic_array import CyclicArray


def test_init_with_same_length():
    initial_array = np.ones(shape=(100,), dtype=float)
    print(np.shape(initial_array))
    length = np.shape(initial_array)[0]
    dimensions = 1

    cyclic_array = CyclicArray(length=length, dimensions=dimensions, initial_array=initial_array)

    assert cyclic_array.size() == np.shape(cyclic_array.queue)[0]
    assert cyclic_array.tail == np.shape(cyclic_array.queue)[0] - 1
    assert cyclic_array.dimensions == dimensions

