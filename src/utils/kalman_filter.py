import numpy as np


class KalmanFilter:
    def __init__(self, sampling_period, rows_per_kalman_use,
                 dev_acc_state=None, dev_gyro_state=None, dev_mode: bool = False):
        # Kalman filter variables
        self.rows_per_kalman_use = rows_per_kalman_use

        self.state_priori = np.array([0.0, 0.0])
        self.state_posteriori = np.array([0.0, 0.0])

        self.p_priori = np.array([[0.5, 0.0],
                                  [0.0, 0.5]])
        self.p_posteriori = np.array([[0.5, 0.0],
                                      [0.0, 0.5]])
        _Q = 0.001 * np.pi  # Prone to change following testing but works fine
        _R = 0.001 * np.pi  # Prone to change following testing but works fine
        self.Q = np.array([[_Q, 0.0],
                           [0.0, _Q]])
        self.R = np.array([[_R, 0.0],
                           [0.0, _R]])
        self.K = np.array([[0.0, 0.0],
                           [0.0, 0.0]])

        # Input/output variables
        self.sampling_period = sampling_period

        self.dev_mode = dev_mode
        if self.dev_mode:
            self.dev_acc_state = dev_acc_state
            self.dev_gyro_state = dev_gyro_state
            self.row_number = 0

    # TODO: should take in a row instead of relying on row_no
    def iterate(self, processed_input: np.array):
        """
        An iteration of the Kalman filter.

        :param processed_input: Current measurements
        """
        # A priori state projection
        self.project_state(processed_input)
        # A priori state uncertainty projection
        self.apriori_uncertainty()
        # Kalman gain calculation
        self.set_gain()
        # A posteriori state correction
        self.correct_state_projection(processed_input)
        # A posteriori uncertainty correction
        self.set_aposteriori_uncertainty()

        if self.dev_mode:
            self.row_number += 1

    def project_state(self, processed_input: np.array):
        # Data merge row number

        # State is projected by adding angle changes from current time step
        # The usage of the np.flip(np.cos(state)) is required to move from sensed to global angular velocity
        self.state_priori = \
            self.state_posteriori + \
            self.sampling_period * processed_input[3:5] \
            * np.flip(np.cos(self.state_posteriori))

        self.state_priori[0] = self.get_corrected_angle(angle=self.state_priori[0])
        self.state_priori[1] = self.get_corrected_angle(angle=self.state_priori[1])
        # In development mode, store information on pure gyro-calculated angles
        if self.dev_mode:
            self.dev_gyro_state[self.row_number] = self.dev_gyro_state[self.row_number - 1] + \
                                                   self.sampling_period * processed_input[3:5] \
                                                   * np.flip(np.cos(self.dev_gyro_state[self.row_number - 1]))

    def apriori_uncertainty(self):
        self.p_priori = self.p_posteriori + self.Q

    def set_gain(self):
        self.K = np.matmul(self.p_priori, (self.p_priori + self.R).transpose())

    def correct_state_projection(self, processed_input: np.array):
        measurements = self.get_measurements(processed_input)
        self.state_posteriori = self.state_priori + np.matmul(self.K, measurements - self.state_priori)

    def set_aposteriori_uncertainty(self):
        self.p_posteriori = np.matmul(np.identity(2, dtype=float) - self.K, self.p_priori)

    def get_measurements(self, processed_input: np.array):
        """
        Observation function. Observes acceleration data, calculates x- and y-angles.
        :return: np.ndarray of shape=(2,), containing calculates x- and y-angles.
        """
        # Data merge row number
        a_abs = np.sqrt(np.sum(np.power(processed_input[0:3], 2)))
        # Special case where nan-values might cause [0.0, 0.0, 0.0] as an acc measurement
        z = np.array([0.0, 0.0])

        if a_abs == 0.0:
            if self.dev_mode:
                self.dev_acc_state[self.row_number] = z
            return self.output[self.row_number, 0:2]

        z[0] = -np.arcsin(processed_input[1] / a_abs)
        z[1] = 0.5 * np.pi - np.arccos(processed_input[0] / a_abs)

        if self.dev_mode:
            self.dev_acc_state[
                self.row_number:min(len(self.dev_acc_state), self.row_number + self.rows_per_kalman_use)] = z

        return z

    def get_state_estimate(self):
        return self.state_posteriori

    @staticmethod
    def get_corrected_angle(angle: float):
        """
        This method exists because angles can be corrected and pushed beyond -+pi, which is an unwanted effect.
        """
        if abs(angle) > np.pi:
            if angle > 0.0:
                if (angle % (2 * np.pi)) > np.pi:
                    angle = (angle % np.pi) - np.pi
                else:
                    angle = angle % np.pi
            else:
                angle = -angle
                if (angle % (2 * np.pi)) > np.pi:
                    angle = (angle % np.pi) - np.pi
                else:
                    angle = angle % np.pi
                angle = -angle
        return angle
