import numpy as np


class KalmanFilter:
    def __init__(self, processed_input, output, sampling_period, rows_per_kalman_use,
                 dev_acc_state=None, dev_gyro_state=None, dev_mode: bool = False):
        # Kalman filter variables
        self.rows_per_kalman_use = rows_per_kalman_use
        self.kal_state_pri = np.array([0.0, 0.0])
        self.kal_state_post = np.array([0.0, 0.0])
        self.kal_p_pri = np.array([[0.5, 0.0],
                                   [0.0, 0.5]])
        self.kal_p_post = np.array([[0.5, 0.0],
                                    [0.0, 0.5]])
        _Q = 0.001 * np.pi  # Prone to change following testing but works fine
        _R = 0.001 * np.pi  # Prone to change following testing but works fine
        self.kal_Q = np.array([[_Q, 0.0],
                               [0.0, _Q]])
        self.kal_R = np.array([[_R, 0.0],
                               [0.0, _R]])
        self.kal_K = np.array([[0.0, 0.0],
                               [0.0, 0.0]])

        # Input/output variables
        self.processed_input = processed_input
        self.output = output
        self.sampling_period = sampling_period

        self.dev_mode = dev_mode
        if self.dev_mode:
            self.dev_acc_state = dev_acc_state
            self.dev_gyro_state = dev_gyro_state

    def kalman_iteration(self, row_no: int):
        """
        An iteration of the Kalman filter.

        :param row_no: Current buffer row index
        """
        # A priori state projection
        self.kalman_project_state(row_no=row_no)
        # A priori state uncertainty projection
        self.kalman_apriori_uncertainty()
        # Kalman gain calculation
        self.kalman_gain()
        # A posteriori state correction
        self.kalman_correct_state_projection(row_no=row_no)
        # A posteriori uncertainty correction
        self.kalman_aposteriori_uncertainty()

    def kalman_project_state(self, row_no: int):
        # Data merge row number

        # State is projected by adding angle changes from current time step
        # The usage of the np.flip(np.cos(state)) is required to move from sensed to global angular velocity
        self.kal_state_pri = \
            self.kal_state_post + \
            self.sampling_period * self.processed_input[row_no, 3:5]\
            * np.flip(np.cos(self.kal_state_post))

        self.kal_state_pri[0] = self.get_corrected_angle(angle=self.kal_state_pri[0])
        self.kal_state_pri[1] = self.get_corrected_angle(angle=self.kal_state_pri[1])
        # In development mode, store information on pure gyro-calculated angles
        if self.dev_mode:
            self.dev_gyro_state[row_no] = self.dev_gyro_state[row_no - 1] + \
                                                 self.sampling_period * self.processed_input[row_no, 3:5]\
                                                 * np.flip(np.cos(self.dev_gyro_state[row_no-1]))

    def kalman_apriori_uncertainty(self):
        self.kal_p_pri = self.kal_p_post + self.kal_Q

    def kalman_gain(self):
        self.kal_K = np.matmul(self.kal_p_pri, (self.kal_p_pri + self.kal_R).transpose())

    def kalman_correct_state_projection(self, row_no: int):
        z = self.kalman_z(row_no=row_no)
        self.kal_state_post = self.kal_state_pri + np.matmul(self.kal_K, z - self.kal_state_pri)

    def kalman_aposteriori_uncertainty(self):
        self.kal_p_post = np.matmul(np.identity(2, dtype=float) - self.kal_K, self.kal_p_pri)

    def kalman_z(self, row_no: int):
        """
        Observation function. Observes acceleration data, calculates x- and y-angles.
        :return: np.ndarray of shape=(2,), containing calculates x- and y-angles.
        """
        # Data merge row number
        a_abs = np.sqrt(np.sum(np.power(self.processed_input[row_no, 0:3], 2)))
        # Special case where nan-values might cause [0.0, 0.0, 0.0] as an acc measurement
        z = np.array([0.0, 0.0])

        if a_abs == 0.0:
            if self.dev_mode:
                self.dev_acc_state[row_no] = z
            return self.output[row_no, 0:2]
        z[0] = -np.arcsin(self.processed_input[row_no, 1] / a_abs)
        z[1] = 0.5 * np.pi - np.arccos(self.processed_input[row_no, 0] / a_abs)

        if self.dev_mode:
            self.dev_acc_state[row_no:min(len(self.dev_acc_state), row_no+self.rows_per_kalman_use)] = z

        return z

    @staticmethod
    def get_corrected_angle(angle: float):
        """
        This method exists because angles can be corrected and pushed beyond -+pi, which is an unwanted effect.
        """
        if abs(angle) > np.pi:
            if angle > 0.0:
                if (angle % (2*np.pi)) > np.pi:
                    angle = (angle % np.pi) - np.pi
                else:
                    angle = angle % np.pi
            else:
                angle = -angle
                if (angle % (2*np.pi)) > np.pi:
                    angle = (angle % np.pi) - np.pi
                else:
                    angle = angle % np.pi
                angle = -angle
        return angle