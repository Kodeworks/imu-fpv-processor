"""
Assumptions:
1. If any single value in a row of 5 values is NaN, then the entire row is NaN

Explainations:
Pose - Both position and orientation of an object
DOF/dof - Degree of freedom, here used to describe data from one of the six sensor readings (acc x-y-z, gyro x-y-z)
"""
from src import globals as g

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft

import warnings
import math


class FloatService:
    """
    FloatService serves the purpose of estimating the height and the two angles of the x- and y-axis to the horizontal
    plane, of an IMU sensor. The FloatService process follows the steps of preprocessing, pose estimation and
    post processing.
    """
    def __init__(self, name: str, input, output, dev_mode: bool = False):
        """
        :param name: str, unique ID of float
        :param input: np.memmap/np.ndarray, input buffer
        :param output: np.memmap/np.ndarray, output buffer
        :param rows:
        :param dev_mode: bool, dev_mode enables features only usable in development
        """
        self.name = name
        self.input = input      # Input = [acc_x, acc_y, acc_z, gyro_x, gyro_y]
        self.output = output    # Output = [x_angle, y_angle, vertical_pos]
        self.input_len = np.shape(input)[0]
        self.last_row = -1

        self.gravitational_constant = 9.81

        # Index of highest valid index from last time the buffer counter was reset
        self.last_valid = self.input_len - 1
        # Miniburst control
        self.use_minibursts = True
        self.miniburst_size = 128

        # The following are variables to control how the burst is handled due to NaN values
        self.burst_contains_nan = False
        self.burst_is_discarded = False
        self.nan_in_burst = 0
        # TODO: The claim below was accurate when assumption 1 wasn't true. The current threshold is lower and
        #       should be examined
        # Tests show that the module handles 70% randomly placed NaN-values. Threshold is set to 50% to accomodate
        # for groupings of several NaN-values, which are more likely than a uniform distribution
        self.discard_burst_nan_threshold = 0.5

        # Internal storage
        self.processed_input = np.zeros(shape=[self.input_len, 5], dtype=float)
        self.actual_vertical_acceleration = np.zeros(shape=[self.input_len], dtype=float)
        self.proper_vertical_acceleration = np.zeros(shape=[self.input_len], dtype=float)
        self.dampened_vertical_velocity = np.zeros(shape=[self.input_len], dtype=float)
        self.dampened_vertical_position = np.zeros(shape=[self.input_len], dtype=float)
        self.vertical_acceleration = 0.0
        self.vertical_velocity = 0.0

        # Number of data points for calculating different means (prone to change)
        self.n_points_for_acc_mean = 4096
        self.n_points_for_gyro_mean = 4096
        self.n_points_for_proper_vert_acc_mean = 4096
        self.n_points_for_vel_mean = 512
        self.n_points_for_pos_mean = 512

        # Information on sensor bias is kept
        self.acc_bias_sliding_window = np.zeros(shape=3, dtype=float)
        self.gyro_bias_sliding_window = np.zeros(shape=2, dtype=float)
        # The _final-variables are the results from adaptive averaging
        self.acc_bias_final = np.zeros(shape=3, dtype=float)
        self.gyro_bias_final = np.zeros(shape=2, dtype=float)

        # Other bias that may emerge from the estimation process
        self.proper_vert_acc_bias = 0.0
        self.vert_vel_bias = 0.0
        self.vert_pos_bias = 0.0

        # Number of data rows to be processed before next bias update
        self.points_between_acc_bias_update = 256
        self.points_between_gyro_bias_update = 256
        self.points_between_proper_vert_acc_bias_update = 256
        self.points_between_vert_vel_bias_update = 32
        self.points_between_vert_pos_bias_update = 8

        # Row numbers of last bias updates
        self.last_acc_bias_update = -1
        self.last_gyro_bias_update = -1
        self.last_proper_vert_acc_bias_update = 0
        self.last_vert_vel_bias_update = 0
        self.last_vert_pos_bias_update = 0

        adav_alpha_max = 0.5
        adav_alpha_gain = 0.01
        # Adaptive averages used in extended sensor calibration
        self.adav_acc_x = AdaptiveMovingAverage(
            alpha_min=0.01, alpha_max=adav_alpha_max, alpha_gain=adav_alpha_gain, dev_mode=dev_mode
        )

        self.adav_acc_y = AdaptiveMovingAverage(
            alpha_min=0.01, alpha_max=adav_alpha_max, alpha_gain=adav_alpha_gain, dev_mode=dev_mode
        )

        self.adav_acc_z = AdaptiveMovingAverage(
            alpha_min=0.01, alpha_max=adav_alpha_max, alpha_gain=adav_alpha_gain, dev_mode=dev_mode
        )

        self.adav_gyro_x = AdaptiveMovingAverage(
            alpha_min=0.01, alpha_max=adav_alpha_max, alpha_gain=adav_alpha_gain, dev_mode=dev_mode
        )

        self.adav_gyro_y = AdaptiveMovingAverage(
            alpha_min=0.01, alpha_max=adav_alpha_max, alpha_gain=adav_alpha_gain, dev_mode=dev_mode
        )

        self.adav_gyro_z = AdaptiveMovingAverage(
            alpha_min=0.01, alpha_max=adav_alpha_max, alpha_gain=adav_alpha_gain, dev_mode=dev_mode
        )

        # Weights for weighted averages
        self.vert_pos_average_weights = None
        self.set_position_average_weights()

        # Dampening factors to counteract integration drift
        self.vel_dampening_factor = 0.001
        self.pos_dampening_factor = 0.001
        # Dampening factors right after a discarded package
        self.vel_dampening_factor_big = 0.05
        self.pos_dampening_factor_big = 0.05
        # Dampening factor end points
        self.vel_dampening_factor_end = 0.001
        self.pos_dampening_factor_end = 0.001
        # Finally, a dampening factor to dampen the dampening factors
        self.dampening_factor_dampener = 0.05

        # Some constants for reversing input to match mathematical convention
        self.perform_axis_reversal = True
        self.gyro_reversal_coefficient = -1.0
        self.axis_reversal_coefficient = -1.0
        # With the current physical implementation, the y-axis (index 1) is reversed
        self.axis_reversal_index = 1

        # Sensor sampling frequency and period
        self.sampling_rate = 104.0
        self.sampling_period = 1.0 / self.sampling_rate

        # Low-pass filtering coefficients
        nyquist_freq = self.sampling_rate / 2
        cutoff_rate = 0.1
        self.low_b, self.low_a = butter(int(1 / (cutoff_rate * self.sampling_rate) * nyquist_freq + 0.5),
                                        cutoff_rate,
                                        btype='lowpass',
                                        output='ba')

        # Kalman filter variables
        self.rows_per_kalman_use = 10
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

        # Variables for generating and storing information on the wave function
        self.n_points_for_fft = int(self.sampling_rate)*10
        self.points_between_fft = int(self.sampling_rate)*5
        self.last_fft = -1
        self.n_saved_wave_functions = 50
        self.wave_function_buffer = np.zeros(shape=[self.n_saved_wave_functions, self.n_points_for_fft//2], dtype=float)
        # Pointer points to the last saved wave function
        self.wave_function_buffer_pointer = -1
        # Determines whether or not wave function information is used in vertical position bias control
        self.fft_aided_bias = False

        self.rotations = Rotations()

        # Development mode variables
        self.dev_mode = dev_mode
        if dev_mode:
            # Extended internal memory to examine different internal variables post processing
            self.dev_bank_angle = np.zeros(shape=[self.input_len], dtype=float)
            self.dev_vertical_velocity = np.zeros(shape=[self.input_len], dtype=float)
            self.dev_gyro_state = np.zeros(shape=[self.input_len, 2], dtype=float)
            self.dev_acc_state = np.zeros(shape=[self.input_len, 2], dtype=float)

            self.n_bias_updates = np.zeros(shape=[5], dtype=int)  # Gyro, xy acc, vert acc, vel, pos

            # Biases for each timestep are also kept for examination
            self.acc_bias_array = np.zeros(shape=[self.input_len, 3], dtype=float)
            self.gyro_bias_array = np.zeros(shape=[self.input_len, 2], dtype=float)
            self.vert_vel_bias_array = np.zeros(shape=[self.input_len], dtype=float)
            self.vert_pos_bias_array = np.zeros(shape=[self.input_len], dtype=float)

            # Some control variables for testing with a purpose of controling vertical position output
            self.no_vert_pos_bias = False
            self.no_vert_vel_bias = False

            # Controlling output post-processing
            self.use_output_filtering = True

            warnings.filterwarnings('error')

    def process(self, number_of_rows: int):
        """
        Tell FloatService to process the next number_of_rows rows in input, starting from last_row + 1.
        :param number_of_rows: Number of input data rows to be processed.
        Format of output: N rows x [x-angle, y-angle, vertical position]
        """
        if self.last_row + number_of_rows + 1 <= g.rows:
            start = self.last_row + 1
        else:
            start = 0

            # Information on last actual buffer index is kept
            self.last_valid = self.last_row

            self.update_counters_on_buffer_reuse()
            self.copy_data_to_last_index_on_buffer_reuse()

        end = start + number_of_rows

        if self.use_minibursts:
            self.minibursts(start=start, end=end)
            self.postprocess_output(start=start, end=end)
        else:
            self.preprocess_data(start=start, end=end)

            # Check whether burst is declared discarded
            if not self.burst_is_discarded:
                self.run_processing_iterations(start=start, end=end)
                self.postprocess_output(start=start, end=end)

        self.last_row = end - 1

    def minibursts(self, start: int, end: int):
        """
        Minibursts are activated when a given burst is of greater size than some threshold. This is to make sure
        some preprocessing steps like real time calibration of sensors (averaging of sensor input) is performed
        regularily
        :param start: Start index of miniburst
        :param end: End index of miniburst
        """
        s_i = start
        e_i = min(end, s_i + self.miniburst_size)
        while s_i < end:
            self.preprocess_data(start=s_i, end=e_i)
            # Check whether burst is declared discarded
            if not self.burst_is_discarded:
                self.run_processing_iterations(start=s_i, end=e_i)

            s_i += self.miniburst_size
            e_i = min(end, s_i+self.miniburst_size)

    def preprocess_data(self, start: int, end: int):
        """
        NaN-handling
        Bias updates
        Set processed input
        Data transformations
        Outlier correction/smoothing
        """

        # NaN-handling
        self.nan_handling(start=start, end=end)
        # Check whether burst was declared as discarded due to NaN_handling
        if self.burst_is_discarded:
            return

        # Update gyroscope and accelerometer bias
        self.update_acc_bias(row_no=start)
        self.update_gyro_bias(row_no=start)

        # If nan_handling() detected any NaN-values in the burst without discarding the burst, separate methods for
        # inserting processed input are used
        if self.burst_contains_nan:
            self.set_processed_acc_input_nan(start=start, end=end)
            self.set_processed_gyro_input_nan(start=start, end=end)
        else:
            # Insert raw acceleration data into processed_input
            self.set_processed_acc_input(start=start, end=end)
            # Adjust current gyro burst according to gyro bias and insert the result in preprocessed_input
            self.set_processed_gyro_input(start=start, end=end)

        # Convert angular velocities from deg/s to rad/s
        self.degrees_to_radians(start=start, end=end)

        # Transform gyro data and a single axis of accelerometer data so that the input matches mathematical convention
        self.reverse_some_processed_input(start=start, end=end)

        # Filtering of both accelerometer and gyroscope data using a low-pass filter
        self.low_pass_filter_input(start=start, end=end)

    def run_processing_iterations(self, start: int, end: int):
        for i in range(start, end):
            self.wave_function(row_no=i)

            self.estimate_pose(row_no=i)

            # Adjust velocity and position dampening factors
            self.adjust_pos_and_vel_dampening_factors()

    def estimate_pose(self, row_no: int):
        # A Kalman filter iteration is performed to estimate x- and y-angles,
        # which later makes up the bank angle
        if row_no % self.rows_per_kalman_use == 0:
            self.kalman_iteration(row_no=row_no)
            self.output[row_no, 0:2] = self.kal_state_post
        else:
            self.estimate_angles_using_gyro(row_no)
            self.kal_state_post = self.output[row_no, 0:2]

        # Vertical acceleration, velocity and position is estimated and stored internally
        self.estimate_vertical_acceleration(row_no=row_no)
        self.estimate_vertical_velocity(row_no=row_no)
        self.estimate_vertical_position(row_no=row_no)

    def update_counters_on_buffer_reuse(self):
        self.last_acc_bias_update = self.last_acc_bias_update - self.last_valid - 1
        self.last_gyro_bias_update = self.last_gyro_bias_update - self.last_valid - 1
        # Information on time since last bias update is updated to match self.last_valid
        self.last_vert_vel_bias_update = self.last_vert_vel_bias_update - self.last_valid - 1
        self.last_vert_pos_bias_update = self.last_vert_pos_bias_update - self.last_valid - 1
        # Same goes with self.last_fft
        self.last_fft = self.last_fft - self.last_valid - 1

    def copy_data_to_last_index_on_buffer_reuse(self):
        # Since some methods use data indexed in [row_no-1], data from [self.last_valid] is copied to
        # [-1]. One could also check for start==0 in each of these methods but this is less expensive.
        self.dampened_vertical_velocity[-1] = self.dampened_vertical_velocity[self.last_valid]
        self.dampened_vertical_position[-1] = self.dampened_vertical_position[self.last_valid]
        self.output[-1] = self.output[self.last_valid]
        if self.dev_mode:
            self.dev_gyro_state[-1] = self.dev_gyro_state[self.last_valid]

    def postprocess_output(self, start: int, end: int):
        if self.dev_mode:
            if self.use_output_filtering:
                self.low_pass_filter_output(start=start, end=end)
        else:
            self.low_pass_filter_output(start=start, end=end)

    def wave_function(self, row_no: int):
        # Check if it is time to perform a Fourier transform
        if row_no - self.last_fft >= self.points_between_fft:
            # If so, check whether fft can be performed on a single stretch of the buffer
            if self.historic_data_is_contiguous(self.n_points_for_fft, end_row_of_data=self.last_row + 1):
                # First, we do a transform of the vertical acceleration signal
                fourier_transform = fft(self.actual_vertical_acceleration[self.last_row+1 - self.n_points_for_fft:
                                                                          self.last_row+1])
            else:
                last_indices, first_indices = self.patched_buffer_indices(self.n_points_for_fft,
                                                                          current_row=self.last_row+1)
                temp_buffer = np.concatenate([self.actual_vertical_acceleration[last_indices[0]:last_indices[1]],
                                              self.actual_vertical_acceleration[first_indices[0]:first_indices[1]]])
                fourier_transform = fft(temp_buffer)

            coeff_array = 2.0 / self.n_points_for_fft * np.abs(fourier_transform[0:self.n_points_for_fft // 2])
            self.save_wave_function(coefficient_array=coeff_array)

            self.last_fft = row_no
            if self.fft_aided_bias:
                self.update_position_bias_control()

    def save_wave_function(self, coefficient_array: np.ndarray):
        # Update buffer pointer
        self.wave_function_buffer_pointer = (self.wave_function_buffer_pointer+1) % self.n_saved_wave_functions
        # Store wave function coefficients in buffer
        #   - In case the buffer is too small to perform FFT using requested number of points we specify that
        #   - self.wave_function_buffer is filled by the number of data points in coefficient_array to avoid a
        #   - ValueError.
        self.wave_function_buffer[self.wave_function_buffer_pointer][0:len(coefficient_array)] = coefficient_array

    def update_position_bias_control(self):
        # TODO: Test how this method affects vert pos bias window during NaN-packages, especially on time usage due to
        #       increased window size. If this is a problem, package discarding must be strengthened with using the
        #       main frequency from the fft
        """
        Use information from wave function to control mean window size for vertical position
        """

        frequencies = np.linspace(0.0, self.sampling_rate/2, self.n_points_for_fft//2)
        top_frequency = frequencies[np.argmax(self.wave_function_buffer[self.wave_function_buffer_pointer])]

        if top_frequency == 0.0:
            return
        new_n_data_points = int(self.sampling_rate * 1/top_frequency)
        # Update bias control, as long as the period of the greatest amplitude is greater than some threshold
        if new_n_data_points > 128:
            self.n_points_for_pos_mean = new_n_data_points
            # print(f'When last row was {self.last_row}, pos bias window size set to {self.n_points_for_pos_mean}')
            self.set_position_average_weights()

    def estimate_vertical_acceleration(self, row_no: int):
        """
        Estimates vertical acceleration of the float and, if necessary, updates acceleration bias.
        """
        # # In order to go from z-sensor acceleration to vertical acceleration, the bank angle is required
        # bank_angle = self.bank_angle(row_no)
        # # In development mode, store information about the bank angle for each time step
        # if self.dev_mode:
        #     self.dev_bank_angle[row_no] = bank_angle
        #
        # # proper_vertical_acceleration = sensor_z_acc / np.cos(bank_angle)
        # # actual_vertical_acceleration = proper_vertical_acceleration + self.gravitational_constant
        # # self.vertical_acceleration = -actual_vertical_acceleration
        # # Alternative take: Assume that most if not all movement is vertical
        #
        # # self.update_proper_vert_acc_bias(row_no=row_no)
        # self.proper_vertical_acceleration[row_no] = np.sqrt(np.sum(np.power(self.processed_input[row_no, 0:3], 2)))
        # actual_vertical_acceleration = self.proper_vertical_acceleration[row_no] - self.proper_vert_acc_bias - \
        #     self.gravitational_constant
        # # actual_vertical_acceleration = self.proper_vertical_acceleration[row_no] - self.gravitational_constant
        #
        # self.vertical_acceleration = actual_vertical_acceleration

        a_vector = self.processed_input[row_no, 0:3]
        inverse_orientation = [-self.output[row_no, 0], -self.output[row_no, 1], 0.0]
        global_a_vector = self.rotations.rotate_system(sys_ax=a_vector, sensor_angles=inverse_orientation)

        self.vertical_acceleration = - (global_a_vector[2] + self.gravitational_constant)

        # Store information on vertical acceleration for wave function estimation
        self.actual_vertical_acceleration[row_no] = self.vertical_acceleration

    def estimate_vertical_velocity(self, row_no: int):
        # Vertical velocity is updated using current vertical acceleration
        self.dampened_vertical_velocity[row_no] = self.dampened_vertical_velocity[row_no - 1] + \
                                                  self.sampling_period * self.vertical_acceleration
        # Vertical velocity is adjusted by a dampening factor to compensate for integration drift
        self.dampened_vertical_velocity[row_no] -= self.dampened_vertical_velocity[row_no] * self.vel_dampening_factor

        # If a number of rows equal to or greater than the threshold for updating vertical vel bias has been traversed,
        # update vertical velocity bias.
        self.update_vert_vel_bias(row_no=row_no)

        # Vertical velocity is adjusted by bias and stored internally
        self.vertical_velocity = self.dampened_vertical_velocity[row_no] - self.vert_vel_bias
        # In development mode, store information on vertical velocity for each time step
        if self.dev_mode:
            self.dev_vertical_velocity[row_no] = self.vertical_velocity
            self.vert_vel_bias_array[row_no] = self.vert_vel_bias

    def estimate_vertical_position(self, row_no: int):
        # Vertical position is updated using current vertical velocity
        self.dampened_vertical_position[row_no] = self.dampened_vertical_position[row_no - 1] + \
                                                  self.sampling_period * self.vertical_velocity
        # Vertical position is adjusted by a dampening factor to compensate for integration drift
        self.dampened_vertical_position[row_no] -= self.dampened_vertical_position[row_no] * self.pos_dampening_factor

        # If a number of rows equal to or greater than the threshold for updating vertical pos bias has been traversed,
        # update vertical position bias.
        self.update_vert_pos_bias(row_no=row_no)

        # Vertical position is adjusted by the bias and stored as output
        self.output[row_no, 2] = self.dampened_vertical_position[row_no] - self.vert_pos_bias

        if self.dev_mode:
            self.vert_pos_bias_array[row_no] = self.vert_pos_bias

    def bank_angle(self, row_no: int):

        return np.arctan(np.sqrt(np.tan(self.output[row_no, 0]) ** 2 + np.tan(self.output[row_no, 1]) ** 2))

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

    def patched_buffer_indices(self, request_size: int, current_row: int):
        """
        Returns two index pairs of the parts of the buffer that make up the most recent historic data of size
        request_size, by slicing the buffer.
        Historic data does not include the current burst.
        Assumes that the buffer is circular in the way that recent data stops at [self.last_valid] and picks up at
        [0], and has last data point at [self.last_row].

        :param request_size: Number of historic data points being requested.
        :param current_row: Index that points to the start of the current burst.

        :returns: Two pairs of indices that can be used directly to access the most recent data,
        ie. buffer[first_pair[0]:first_pair[1]]
        """
        # buffer_beginning is analogous to some variable request_end
        buffer_beginning = [0, current_row]
        # buffer_end is analogous to some variable request_beginning
        buffer_end = [self.last_valid + 1 - (request_size - current_row), self.last_valid + 1]

        # indices are returned in the order that data was written to the buffer
        return buffer_end, buffer_beginning

    def set_processed_acc_input(self, start: int, end: int):
        # self.processed_input[start:end, 0:3] = (self.input[start:end, 0:3] - self.acc_bias_final) * \
        #                                        self.gravitational_constant
        self.processed_input[start:end, 0:3] = self.input[start:end, 0:3] * self.gravitational_constant
        self.processed_input[start:end, 0:2] -= self.acc_bias_final[0:2] * self.gravitational_constant
        # self.processed_input[start:end, 0:3] = self.input[start:end, 0:3] * \
        #                                        self.gravitational_constant

    def set_processed_gyro_input(self, start: int, end: int):
        """
        Adjusts gyro data using a sensor bias.
        :param start: Index that slices the buffer part that is to be adjusted.
        :param end: Index that slices the buffer part that is to be adjusted.
        """
        self.processed_input[start: end, 3:5] = self.input[start: end, 3:5] - self.gyro_bias_final

    def set_processed_acc_input_nan(self, start: int, end: int):
        # TODO: Consider a variable that updates for each data row in this method. The value increases by some metric
        #       for each non-NaN value, and decreases for each NaN-value. If the variable stays below a certain value
        #       for some number of rows, these rows are discarded. NB! Should use a [0, 1] array for this, and for
        #       setting all inputs where NaN values are present.
        # Method uses assumtion 1
        first_useable = float('nan')
        first_useable_i = 0
        last_useable = float('nan')
        last_useable_i = 0
        for i in range(start, end):
            # Check if the current value is NaN
            if np.any(np.isnan(self.input[i, 0:3])):
                # Check if last_useable has been assigned
                if np.any(np.isnan(last_useable)):
                    # If last_useable has not been assigned, the current value is the first NaN value of
                    # some number of NaN values. Therefore, assign last_useable as the previous value
                    last_useable = self.input[i - 1, 0:3]
                    last_useable_i = i - 1
                    # Taking into account that self.input[i-1] might also be a NaN value (in case this is the beginning
                    # of a burst and the last value of the previous burst was NaN), a generic set of values may need
                    # to be used
                    if np.any(np.isnan(last_useable)):
                        last_useable = np.array([0.0, 0.0, -1.0])   # Generic set of [accX, accY, accZ]

                    # If first_useable has also been assigned, all values in
                    # self.input[first_useable: i] can be copied directly to self.processed_input
                    if not np.any(np.isnan(first_useable)):
                        self.processed_input[first_useable_i:i, 0:3] = self.input[first_useable_i: i, 0:3]
                        first_useable = float('nan')
                    # If last_useable has already been assigned, the current value is somewhere within a row of
                    # several NaN. Thus, nothing is done.
            else:
                # If the current value is not NaN, check if last_useable has been assigned
                # If last_useable has NOT been assigned, the current value is somewhere within a row of
                # actual values. Nothing needs to be done
                if not np.any(np.isnan(last_useable)):
                    # If last_useable has in fact been assigned, the current value is the first actual value
                    # after some number of NaN values. Therefore, next_useable is assigned as current value.
                    next_useable = self.input[i, 0:3]
                    # The values between last_useable and next_useable are then interpolated
                    # Incremental values are identified
                    steps = i - last_useable_i
                    increment = (next_useable - last_useable) / steps
                    increment_array = np.linspace(increment, increment * (steps - 1), steps - 1)
                    # Currently encapsulated NaN-values are set equal to last_useable
                    self.processed_input[last_useable_i + 1:i, 0:3] = last_useable
                    # Increments are added
                    # TODO: Solve this try/except
                    try:
                        self.processed_input[last_useable_i + 1:i, 0:3] += increment_array
                    except ValueError:
                        print(f'Couldn\'t insert the interpolated gyro array into processed_input:\n'
                              f'Steps: {steps}, last_usable_i: {last_useable_i}, increment_array: {increment_array}')

                    # Finally, both last_useable and next_useable are set to NaN again
                    last_useable = float('nan')

                # If first_useable is not set to any value, this is the first non-NaN value in some row of
                # non-NaN values.
                if np.any(np.isnan(first_useable)):
                    first_useable = self.input[i, 0:3]
                    first_useable_i = i

        # When the entire row is checked for NaN, there may still be the case that the last values are
        # NaN values. In this case we cannot interpolate anything and instead we perform a simple
        # extrapolation, by copying last_useable into each of the values.
        if not np.any(np.isnan(last_useable)):
            self.processed_input[last_useable_i + 1:end, 0:3] = last_useable
        else:
            # If last_useable is not set, this bursts ends with non-NaN values. These can be inserted into
            # self.processed_input
            self.processed_input[first_useable_i:end, 0:3] = self.input[first_useable_i:end, 0:3]
        # Finally, after having copied/inter/extrapolated input to processed_input, the data is adjusted according
        # to input bias
        self.processed_input[start:end, 0:2] -= self.acc_bias_final[0:2]
        self.processed_input[start:end, 0:3] *= self.gravitational_constant

    def set_processed_gyro_input_nan(self, start: int, end: int):
        # TODO: Consider a variable that updates for each data row in this method. The value increases by some metric
        #       for each non-NaN value, and decreases for each NaN-value. If the variable stays below a certain value
        #       for some number of rows, these rows are discarded.
        # Method uses assumtion 1
        first_useable = float('nan')
        first_useable_i = 0
        last_useable = float('nan')
        last_useable_i = 0
        for i in range(start, end):
            # Check if the current value is NaN
            if np.any(np.isnan(self.input[i, 3:5])):
                # Check if last_useable has been assigned
                if np.any(np.isnan(last_useable)):
                    # If last_useable has not been assigned, the current value is the first NaN value of
                    # some number of NaN values. Therefore, assign last_useable as the previous value
                    last_useable = self.input[i - 1, 3:5]
                    last_useable_i = i - 1

                    # If first_useable has also been assigned, all values in
                    # self.input[first_useable: i] can be copied directly to self.processed_input
                    if not np.any(np.isnan(first_useable)):
                        self.processed_input[first_useable_i:i, 3:5] = self.input[first_useable_i: i, 3:5]
                        first_useable = float('nan')

                    # Taking into account that self.input[i-1] might also be a NaN value (in case this is the beginning
                    # of a burst and the last value of the previous burst was NaN), a generic set of values may need
                    # to be used
                    if np.any(np.isnan(last_useable)):
                        last_useable = np.array([0.0, 0.0])   # Generic set of [gyroX, gyroY]

                    # If last_useable has already been assigned, the current value is somewhere within a row of
                    # several NaN. Thus, nothing is done.
            else:
                # If the current value is not NaN, check if last_useable has been assigned
                # If last_useable has NOT been assigned, the current value is somewhere within a row of
                # actual values. Nothing needs to be done
                if not np.any(np.isnan(last_useable)):
                    # If last_useable has in fact been assigned, the current value is the first actual value
                    # after some number of NaN values. Therefore, next_useable is assigned as current value.
                    next_useable = self.input[i, 3:5]
                    # The values between last_useable and next_useable are then interpolated
                    # Incremental values are identified
                    steps = i - last_useable_i
                    increment = (next_useable - last_useable) / steps
                    increment_array = np.linspace(increment, increment * (steps - 1), steps - 1)
                    # Currently encapsulated NaN-values are set equal to last_useable
                    self.processed_input[last_useable_i + 1:i, 3:5] = last_useable
                    # Increments are added
                    # TODO: Solve this try/except
                    try:
                        self.processed_input[last_useable_i + 1:i, 3:5] += increment_array
                    except ValueError:
                        print(f'Couldn\'t insert the interpolated gyro array into processed_input:\n'
                              f'Steps: {steps}, last_usable_i: {last_useable_i}, increment_array: {increment_array}')
                    # Finally, both last_useable and next_useable are set to NaN again
                    last_useable = float('nan')

                # If first_useable is not set to any value, this is the first non-NaN value in some row of
                # non-NaN values.
                if np.any(np.isnan(first_useable)):
                    first_useable = self.input[i, 3:5]
                    first_useable_i = i

        # When the entire row is checked for NaN, there may still be the case that the last values are
        # NaN values. In this case we cannot interpolate anything and instead we perform a simple
        # extrapolation, by copying last_useable into each of the values.
        if not np.any(np.isnan(last_useable)):
            self.processed_input[last_useable_i + 1:end, 3:5] = last_useable
        else:
            # If last_useable is not set, this bursts ends with non-NaN values. These can be inserted into
            # self.processed_input
            self.processed_input[first_useable_i:end, 3:5] = self.input[first_useable_i:end, 3:5]

        # Finally, after having copied/inter/extrapolated input to processed_input, the data is adjusted according
        # to input bias
        self.processed_input[start:end, 3:5] -= self.gyro_bias_final

    def update_gyro_bias(self, row_no: int):
        """
        Updates gyroscope sensor bias from a mean of historic data.
        """
        # Update gyroscope bias if enough data has arrived since last update
        if row_no - self.last_gyro_bias_update >= self.points_between_gyro_bias_update:
            if self.dev_mode:
                self.n_bias_updates[0] += 1

            self.update_sliding_window_gyro_bias(row_no=row_no)
            self.update_adaptive_gyro_bias()

            if self.dev_mode:
                if row_no > self.n_points_for_gyro_mean:
                    self.gyro_bias_array[self.last_gyro_bias_update: row_no] = self.gyro_bias_final

            self.last_gyro_bias_update = row_no

    def update_sliding_window_gyro_bias(self, row_no: int):
        # Check whether the entire data request can be indexed within [0, row_no]
        if self.historic_data_is_contiguous(request_size=self.n_points_for_gyro_mean,
                                            end_row_of_data=row_no):
            self.gyro_bias_sliding_window = np.nanmean(self.input[row_no - self.n_points_for_gyro_mean:
                                                                  row_no, 3:5],
                                                       axis=0)
        else:
            # If data is split between the end and the start of the buffer,
            # indices are generated to slice these parts
            last_indices, first_indices = self.patched_buffer_indices(request_size=self.n_points_for_gyro_mean,
                                                                      current_row=row_no)
            # The number of NaN-rows (if any) in the part of the buffer being used is required when finding the mean
            # of two separate parts of the buffer.
            n_nan = np.count_nonzero(np.isnan(self.input[last_indices[0]:last_indices[1], 0])) \
                + np.count_nonzero(np.isnan(self.input[first_indices[0]:first_indices[1], 0]))

            self.gyro_bias_sliding_window = (np.nansum(self.input[last_indices[0]:last_indices[1], 3:5], axis=0) +
                                             np.nansum(self.input[first_indices[0]:first_indices[1], 3:5], axis=0)) \
                / (self.n_points_for_gyro_mean - n_nan)

    def update_adaptive_gyro_bias(self):
        self.adav_gyro_x.update(sample=self.gyro_bias_sliding_window[0])
        self.adav_gyro_y.update(sample=self.gyro_bias_sliding_window[1])
        # self.adav_gyro_z.update(sample=self.gyro_bias_sliding_window[2])
        self.gyro_bias_final[0] = self.adav_gyro_x.adaptive_average
        self.gyro_bias_final[1] = self.adav_gyro_y.adaptive_average
        # self.gyro_bias_final[2] = self.adav_gyro_z.adaptive_average

    def update_acc_bias(self, row_no: int):
        """
        Updates x- and y-acceleration sensor bias from a mean of historic data.

        :param row_no: First index of current burst.
        """

        if row_no - self.last_acc_bias_update >= self.points_between_acc_bias_update:
            if self.dev_mode:
                self.n_bias_updates[1] += 1

            self.update_sliding_window_acc_bias(row_no=row_no)
            self.update_adaptive_acc_bias()

            if self.dev_mode:
                if row_no > self.n_points_for_acc_mean:
                    self.acc_bias_array[self.last_acc_bias_update: row_no] = self.acc_bias_final

            self.last_acc_bias_update = row_no

    def update_sliding_window_acc_bias(self, row_no: int):
        # Check whether the entire data request can be indexed within [0, row_no]
        if self.historic_data_is_contiguous(request_size=self.n_points_for_acc_mean, end_row_of_data=row_no):
            self.acc_bias_sliding_window = \
                np.nanmean(self.input[row_no - self.n_points_for_acc_mean: row_no, 0:3], axis=0)
        else:
            # If data is split between the end and the start of the buffer,
            # indices are generated to slice these parts
            last_indices, first_indices = self.patched_buffer_indices(request_size=self.n_points_for_acc_mean,
                                                                      current_row=row_no)
            # The number of NaN-rows (if any) in the part of the buffer being used is required when finding the mean
            # of two separate parts of the buffer.
            n_nan = np.count_nonzero(np.isnan(self.input[last_indices[0]:last_indices[1], 0])) \
                + np.count_nonzero(np.isnan(self.input[first_indices[0]:first_indices[1], 0]))

            self.acc_bias_sliding_window = \
                (np.nansum(self.input[last_indices[0]:last_indices[1], 0:3], axis=0) +
                 np.nansum(self.input[first_indices[0]:first_indices[1], 0:3], axis=0)) \
                / (self.n_points_for_acc_mean - n_nan)

        # Since we expect the z acceleration to have an average value of -1.0 g, we subtract this from the
        # accumulated z acceleration bias
        # self.acc_bias[2] += 1.0

        # Instead of setting acc_bias so that it assumes a mean of -1.0 g in the z-acc sensor, we skip the bias
        # adjustment at this point an revisits it in the form of assuming the proper vertical acceleration of the
        # float averages at -1.0 in estimate_vertical_acceleration
        # self.acc_bias_sliding_window[2] = 0.0

    def update_adaptive_acc_bias(self):
        self.adav_acc_x.update(sample=self.acc_bias_sliding_window[0])
        self.adav_acc_y.update(sample=self.acc_bias_sliding_window[1])
        self.adav_acc_z.update(sample=self.acc_bias_sliding_window[2])
        self.acc_bias_final[0] = self.adav_acc_x.adaptive_average
        self.acc_bias_final[1] = self.adav_acc_y.adaptive_average
        self.acc_bias_final[2] = self.adav_acc_z.adaptive_average

    def update_proper_vert_acc_bias(self, row_no: int):
        if row_no - self.last_proper_vert_acc_bias_update >= self.points_between_proper_vert_acc_bias_update:
            if self.historic_data_is_contiguous(request_size=self.n_points_for_proper_vert_acc_mean,
                                                end_row_of_data=row_no):
                self.proper_vert_acc_bias = \
                    np.mean(self.proper_vertical_acceleration[row_no - self.n_points_for_proper_vert_acc_mean: row_no])
            else:
                last_indices, first_indices = \
                    self.patched_buffer_indices(request_size=self.n_points_for_proper_vert_acc_mean, current_row=row_no)
                self.proper_vert_acc_bias = \
                    np.sum(self.proper_vertical_acceleration[last_indices[0]: last_indices[1]]) + \
                    np.sum(self.proper_vertical_acceleration[first_indices[0]:first_indices[1]])
                self.proper_vert_acc_bias /= self.n_points_for_proper_vert_acc_mean

            self.proper_vert_acc_bias -= self.gravitational_constant

            self.last_proper_vert_acc_bias_update = row_no

    def update_vert_vel_bias(self, row_no: int):
        """
        Updates vertical velocity bias from a mean of historic data.

        :param row_no: Current row index
        """
        if row_no - self.last_vert_vel_bias_update >= self.points_between_vert_vel_bias_update:
            # Check whether the entire data request can be indexed within [0, self.last_row]
            if self.historic_data_is_contiguous(request_size=self.n_points_for_vel_mean, end_row_of_data=row_no):
                self.vert_vel_bias = np.mean(self.dampened_vertical_velocity[row_no - self.n_points_for_vel_mean:
                                                                             row_no])
            else:
                # If data is split between the end and the start of the buffer,
                # indices are generated to slice these parts
                last_indices, first_indices = self.patched_buffer_indices(request_size=self.n_points_for_vel_mean,
                                                                          current_row=row_no)
                self.vert_vel_bias = np.sum(self.dampened_vertical_velocity[last_indices[0]:last_indices[1]]) + \
                    np.sum(self.dampened_vertical_velocity[first_indices[0]:first_indices[1]])
                self.vert_vel_bias /= self.n_points_for_vel_mean

            if self.dev_mode:
                self.n_bias_updates[3] += 1
                if self.no_vert_vel_bias:
                    self.vert_vel_bias = 0.0

            self.last_vert_vel_bias_update = row_no

    def update_vert_pos_bias(self, row_no: int):
        """
        Updates vertical position bias from a mean of historic data.

        :param row_no: Current row index
        """
        if row_no - self.last_vert_pos_bias_update >= self.points_between_vert_pos_bias_update:
            # Check whether the entire data request can be indexed within [0, self.last_row]
            if self.historic_data_is_contiguous(request_size=self.n_points_for_pos_mean, end_row_of_data=row_no):
                self.vert_pos_bias = np.average(self.dampened_vertical_position[row_no - self.n_points_for_pos_mean:
                                                                                row_no],
                                                weights=self.vert_pos_average_weights)
            else:
                # If data is split between the end and the start of the buffer,
                # indices are generated to slice these parts
                last_indices, first_indices = self.patched_buffer_indices(request_size=self.n_points_for_pos_mean,
                                                                          current_row=row_no)
                self.vert_pos_bias = np.sum(self.dampened_vertical_position[last_indices[0]:last_indices[1]]) + \
                    np.sum(self.dampened_vertical_position[first_indices[0]:first_indices[1]])

                self.vert_pos_bias /= self.n_points_for_pos_mean

            if self.dev_mode:
                self.n_bias_updates[4] += 1
                if self.no_vert_pos_bias:
                    self.vert_pos_bias = 0.0

            self.last_vert_pos_bias_update = row_no

    def degrees_to_radians(self, start: int, end: int):
        """
        Converts data of angular velocity from deg/s to rad/s.
        :param start: Start index
        :param end: End index
        """
        self.processed_input[start: end, 3:5] = self.processed_input[start: end, 3:5] * np.pi / 180.0

    def reverse_some_processed_input(self, start: int, end: int):
        """
        From testing we know that the sensor output does not match mathematical convention. This method uses information
        provided in __init__() to reverse all gyro axes and exactly one of the accelerometer axes.
        :param start: First index of input to be reversed
        :param end: First index not to be reversed
        """
        if self.perform_axis_reversal:
            # Reverse gyro data
            self.processed_input[start: end, 3:5] *= self.gyro_reversal_coefficient
            # Reverse accelerometer data. Since positive direction of accelerometer and gyro is linked, the same gyro axis
            # must be reversed again.
            self.processed_input[start: end, self.axis_reversal_index] *= self.axis_reversal_coefficient
            if self.axis_reversal_index < 2:
                self.processed_input[start: end, 3 + self.axis_reversal_index] *= self.axis_reversal_coefficient

    def low_pass_filter_input(self, start: int, end: int):
        """
        Filters each of the 6 DOFs using a low-pass filter.
        :param start: Index of first row to be filtered
        :param end: Index of first row not to be filtered
        """
        if end - start < 19:
            if end < 19:
                for dof_i in range(5):
                    temp_input = np.concatenate([self.processed_input[-(19 - end):, dof_i],
                                                 self.processed_input[0:end, dof_i]])
                    self.processed_input[start: end, dof_i] = \
                        np.array(filtfilt(self.low_b,
                                          self.low_a,
                                          temp_input))[start - end:]

            else:
                for dof_i in range(5):
                    self.processed_input[start: end, dof_i] = \
                        np.array(filtfilt(self.low_b,
                                          self.low_a,
                                          self.processed_input[end - 19: end, dof_i]))[start - end:]

        else:
            for dof_i in range(5):
                self.processed_input[start: end, dof_i] = \
                    np.array(filtfilt(self.low_b,
                                      self.low_a,
                                      self.processed_input[start: end, dof_i]))

    def low_pass_filter_output(self, start: int, end: int):
        """
        Filters the outputted vertical position using a low-pass filter.
        :param start: Index of first row to be filtered
        :param end: Index of first row not to be filtered
        """
        if end - start < 19:
            if end < 19:
                temp_output = np.concatenate([self.output[-(19 - end):, 2],
                                              self.output[0:end, 2]])
                self.output[start: end, 2] = \
                    np.array(filtfilt(self.low_b,
                                      self.low_a,
                                      temp_output))[start - end:]

            else:
                self.output[start: end, 2] = \
                    np.array(filtfilt(self.low_b,
                                      self.low_a,
                                      self.output[end - 19: end, 2]))[start - end:]

        else:
            self.output[start: end, 2] = \
                np.array(filtfilt(self.low_b,
                                  self.low_a,
                                  self.output[start: end, 2]))

    def polynomial_smoothing(self, start: int, end: int):
        """
        Alternative smoothing method to low-pass filter.
        Fit a burst of data to an n-polynomial and extrapolate data points from the polynomial curve.
        """
        degree = (end - start + 8) // 10
        for dof in range(5):
            z = np.polyfit(x=np.arange(0, end - start, 1), y=self.processed_input[start:end, dof], deg=degree)
            polynomial = np.poly1d(z)
            self.processed_input[start:end, dof] = polynomial(np.arange(0, end - start, 1))

    def nan_handling(self, start: int, end: int):
        # Method uses assumtion 1
        if np.any(np.isnan(self.input[start:end, 0])):
            self.burst_contains_nan = True
            # If in fact all values of any DoF are NaN, the entire processing of this burst should be handled in a
            # separate method
            self.nan_in_burst = np.count_nonzero(np.isnan(self.input[start:end, 0]))
            if self.nan_in_burst > int((end-start)*self.discard_burst_nan_threshold):
                self.discard_burst(start=start, end=end)
                return
            else:
                self.burst_is_discarded = False
        else:
            self.burst_contains_nan = False
            self.burst_is_discarded = False

    def discard_burst(self, start: int, end: int):
        """
        This method is only called when an entire burst is unusable as a result of NaN values.
        It should:
        1. Set input and processed input to some very standard values, like rows of [0.0, 0.0, -1.0, 0.0, 0.0].
        2. Set other internal storage values to something reasonable.
        3. Set some output other than zeros. Copying previous output might be ok.
        4. Set self.discard_current_burst to True.
        """
        # Set processed input
        self.processed_input[start:end, 0:5] = np.array([0.0, 0.0, -self.gravitational_constant, 0.0, 0.0])
        self.actual_vertical_acceleration[start:end] = 0.0
        self.dampened_vertical_velocity[start:end] = self.vert_vel_bias
        self.dampened_vertical_position[start:end] = self.vert_pos_bias
        # The scalars self.vertical_acceleration and self.vertical_velocity are left unhandled since
        # they are calculated before use anyways

        # Set dev_mode storage
        if self.dev_mode:
            self.dev_bank_angle[start:end] = 0.0
            self.dev_vertical_velocity[start:end] = 0.0
            self.dev_gyro_state[start:end] = 0.0
            self.dev_acc_state[start:end] = 0.0
            self.gyro_bias_array[start:end] = self.gyro_bias_final
            self.acc_bias_array[start:end] = np.array([0.0, 0.0, 0.0])
            self.vert_vel_bias_array[start:end] = self.vert_vel_bias
            self.vert_pos_bias_array[start:end] = self.vert_pos_bias

        # Set output
        self.output[start:end] = np.array([0.0, 0.0, 0.0])

        # Declare burst as discarded
        self.burst_is_discarded = True

        # If the burst is discarded, one of the measures taken to quickly adjust to incoming data is to increase
        # the dampening factors of speed and velocity. These factors are then themselves dampened subsequently.
        self.boost_dampeners()

    def boost_dampeners(self, vel_dampening_factor: float = 0.0, pos_dampening_factor: float = 0.0):
        if vel_dampening_factor > 0.0:
            self.vel_dampening_factor = vel_dampening_factor
        else:
            self.vel_dampening_factor = self.vel_dampening_factor_big

        if pos_dampening_factor > 0.0:
            self.pos_dampening_factor = pos_dampening_factor
        else:
            self.pos_dampening_factor = self.pos_dampening_factor_big

    def adjust_pos_and_vel_dampening_factors(self):
        # Dampening factor is itself dampened
        self.pos_dampening_factor -= self.dampening_factor_dampener * (self.pos_dampening_factor -
                                                                       self.pos_dampening_factor_end)
        # Dampening factor is itself dampened
        self.vel_dampening_factor -= self.dampening_factor_dampener * (self.vel_dampening_factor -
                                                                       self.vel_dampening_factor_end)

    def set_position_average_weights(self):
        """
        Create some array of floats, usable as weights for a weighted average.
        This method exists because the length of the average window for position corrections varies with time.
        """
        # self.vert_pos_average_weights = np.linspace(0.0, 1.0, self.n_points_for_pos_mean)
        self.vert_pos_average_weights = np.ones(shape=[self.n_points_for_pos_mean])

    def estimate_angles_using_gyro(self, row_no):
        self.output[row_no, 0:2] = \
            self.output[row_no-1, 0:2] + \
            self.sampling_period * self.processed_input[row_no, 3:5]\
            * np.flip(np.cos(self.output[row_no-1, 0:2]))
        if self.dev_mode:
            self.dev_gyro_state[row_no] = self.dev_gyro_state[row_no - 1] + \
                                                 self.sampling_period * self.processed_input[row_no, 3:5]\
                                                 * np.flip(np.cos(self.dev_gyro_state[row_no-1]))

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
        :return: np.ndarray of shape=[2], containing calculates x- and y-angles.
        """
        # Data merge row number
        g_size = np.sqrt(np.sum(np.power(self.processed_input[row_no, 0:3], 2)))
        # Special case where nan-values might cause [0.0, 0.0, 0.0] as an acc measurement
        z = np.array([0.0, 0.0])

        if g_size == 0.0:
            if self.dev_mode:
                self.dev_acc_state[row_no] = z
            return self.output[row_no, 0:2]
        z[0] = -np.arcsin(self.processed_input[row_no, 1] / g_size)
        z[1] = 0.5 * np.pi - np.arccos(self.processed_input[row_no, 0] / g_size)

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


class AdaptiveMovingAverage:

    adaptive_average = 0.0
    alpha = 0.0

    def __init__(self, alpha_min=0.01, alpha_max=1.0, alpha_gain=0.05, dev_mode: bool = False):
        self.alpha_tracker = AlphaTracker(alpha_min, alpha_max, alpha_gain)
        self.dev_mode = dev_mode
        if self.dev_mode:
            self.adaptive_average_array = []
            self.alpha_array = []

    def update(self, sample):
        self.alpha_tracker.update(sample)
        self.alpha = self.alpha_tracker.get_alpha()

        self.adaptive_average = self.alpha * sample + (1 - self.alpha) * self.adaptive_average

        if self.dev_mode:
            self.adaptive_average_array.append(self.adaptive_average)
            self.alpha_array.append(self.alpha)

    def get_state(self):
        return self.adaptive_average

    def get_alpha(self):
        return self.alpha


class AlphaTracker:
    low_power = 0.5
    max_shrink = 0.9
    last_sample = None

    def __init__(self, alpha_min=0.01, alpha_max=1.0, alpha_gain=0.05):
        self.alpha = alpha_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_gain = alpha_gain

        if alpha_min > alpha_max:
            raise Exception(f"max value of alpha must be larger than minimum value of alpha. Values given:"
                            f" alpha_min = {alpha_min}, alpha_max = {alpha_max}")

        self.gain = ExponentialMovingAverage(alpha_gain)
        self.loss = ExponentialMovingAverage(alpha_gain)

    @staticmethod
    def sign(x):
        return 1 if x > 0 else -1

    def get_instability(self, avg_gain, avg_loss):
        abs_elevation = abs(avg_gain + avg_loss)
        distance_travelled = avg_gain - avg_loss

        if distance_travelled < 0 or abs_elevation > distance_travelled:
            raise Exception("Invalid distance or elevation")

        if distance_travelled == 0:
            return 1.0

        r = abs_elevation / distance_travelled
        d = 2 * (r - 0.5)
        instability = 0.5 + (self.sign(d) * pow(abs(d), self.low_power)) / 2.0

        if instability < 0 or instability > 1:
            raise Exception(f"Invalid instability: {instability}")

        return instability

    def update(self, sample):
        slope = 0 if not self.last_sample else (sample - self.last_sample)
        self.gain.update(max(slope, 0))
        self.loss.update(min(slope, 0))

        self.last_sample = sample

        instability = self.get_instability(self.gain.get_state(), self.loss.get_state())

        alpha_ideal = self.alpha_min + (self.alpha_max - self.alpha_min) * instability
        self.alpha = max(alpha_ideal, self.max_shrink * self.alpha)

    def get_alpha(self):
        return self.alpha

    def get_avg_gain(self):
        return self.gain.get_state()

    def get_avg_loss(self):
        return self.loss.get_state()


class ExponentialMovingAverage:

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.age = 0
        self.moving_average = 0.0
        self.age_min = int(math.ceil(1.0 / alpha))

    def set_alpha(self, alpha):
        if alpha < 0 or alpha > 1:
            old_alpha = alpha
            alpha = min(alpha, 1)
            alpha = max(alpha, 0)

            warnings.warn(f"Alpha should be between 0 and 1. Alpha was set to {old_alpha} and constrained to {alpha}")

        self.alpha = alpha

    def update(self, state):
        self.age += 1

        if self.age <= 1:
            self.moving_average = state
            return

        if self.age > self.age_min:
            alpha_t = self.alpha
        else:
            alpha_t = 1.0 / self.age

        self.moving_average = (1 - alpha_t) * self.moving_average + alpha_t * state

    def get_age(self):
        return self.age

    def get_state(self):
        return self.moving_average


class Rotations:
    @staticmethod
    def absolute_rotation(rotation):
        """
        Given three rotation angles, calculate absolute rotation along some axis.
        :param rotation: Angles of rotation in [roll, pitch, yaw].
        :return: Absolute rotation as a single angle.
        """
        return np.sqrt(rotation[0]**2 + rotation[1]**2 + rotation[2]**2)

    @staticmethod
    def rotation_axis(rotation, abs_rot):
        """
        Calculate the rotation axis around which the float rotates for some interval where the axis is assumed
        stationary.
        :param rotation: The recorded rotations in roll, pitch and yaw direction. Can be either actual angles or angular
        velocity.
        :param abs_rot: If already calculated, the absolute rotation given the three angles.
        :returns: An [x, y, z] directed, normalized rotation axis.
        """
        return np.asarray([rotation[0], rotation[1], rotation[2]])/abs_rot

    def rotation_matrix(self, sensor_angles):
        """
        Produce a rotation matrix given a sensor reading of rotations.
        :param sensor_angles: A set of [roll, pitch, yaw] angles collected from sensors.
        :return: The rotation matrix needed to rotate a system of coordinates according to the given measured rotation.
        """
        abs_rot = self.absolute_rotation(sensor_angles)
        # If the entire rotation is found to be zero, return the identity matrix
        if abs_rot == 0.0:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        [vx, vy, vz] = self.rotation_axis(sensor_angles, abs_rot=abs_rot)

        c = np.cos(abs_rot)
        s = np.sin(abs_rot)
        r1 = [c + vx**2*(1-c), vx*vy*(1-c)-vz*s, vx*vz*(1-c)+vy*s]
        r2 = [vx*vy*(1-c)+vz*s, c+vy**2*(1-c), vy*vz*(1-c)-vx*s]
        r3 = [vx*vz*(1-c)-vy*s, vz*vy*(1-c)+vx*s, c+vz**2*(1-c)]

        return np.array([r1,
                         r2,
                         r3])

    def rotate_system(self, sys_ax, sensor_angles):
        """
        Rotates a sensor in the global reference frame according to local gyro data.

        Sensor is represented as three unit vectors emerging from origo, denoting the three axes of the system:
          [x, y, z]
         =
        [[x_X, y_X, z_X],
         [x_Y, y_Y, z_Y],
         [x_Z, y_Z, z_Z]]

         Where x, y, z are 3x1 the axes of the inner system, represented by coordinates from the outer system, X, Y, Z,
         Such that x_Y is the outer Y-coordinate of the inner x-axis.

        :param sys_ax: Orientation of sensor axes in a 3x3 matrix. Columns are 3x1 position vectors.
        :param sensor_angles: Some angles rotated around an assumed-to-be-stationary axis.
        :return: A 3x3 matrix of the axes of the new local frame. Columns are 3x1 position vectors.
        """
        rot_mat = self.rotation_matrix(sensor_angles=sensor_angles)
        res = np.matmul(sys_ax, rot_mat)
        return res
