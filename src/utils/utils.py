import numpy as np


class Utils:
    @staticmethod
    def degrees_to_radians(arr: np.array):
        """
        Converts data of angular velocity from deg/s to rad/s.
        :param arr: numpy array of arbitrary size to convert
        """
        return arr * np.pi / 180

    @staticmethod
    def radians_to_degrees(arr: np.array):
        """
        Converts data of angular velocity from rad/s to deg/s.
        :param arr: numpy array of arbitrary size to convert
        """
        return arr * 180.0 / np.pi
