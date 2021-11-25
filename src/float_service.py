"""
Assumptions:
1. If any single value in a row of 5 values is NaN, then the entire row is NaN

Explainations:
Pose - Both position and orientation of an object
DOF/dof - Degree of freedom, here used to describe data from one of the six sensor readings (acc x-y-z, gyro x-y-z)
"""
import string

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import config as cfg

import warnings

# The operating buffer size of FloatService should be a global variable so that it may be set once by some other process
from src.utils.bias_estimator import BiasEstimator
from src.utils.damping import Damping
from src.utils.utils import Utils
from src.utils.adaptive_moving_average import AdaptiveMovingAverage
from src.utils.kalman_filter import KalmanFilter
from src.utils.rotations import Rotations
from src.utils.mem_map_utils import MemMapUtils
from src.utils.low_pass_filter import LowPassFilter

global n_rows


class FloatService:
    """
    FloatService serves the purpose of estimating the height and the two angles of the x- and y-axis to the horizontal
    plane, of an IMU sensor. The FloatService process follows the steps of preprocessing, pose estimation and
    post processing.
    """

    def __init__(self, name: str, input, output, dev_mode: bool = False):
        """
        :param name: str, unique ID of sensor
        :param input: np.memmap/np.ndarray, input buffer
        :param output: np.memmap/np.ndarray, output buffer
        :param dev_mode: bool, dev_mode enables features only usable in development
        """
        self.name = name
        self.dev_mode = dev_mode
        self.imu_mmap = input  # Input = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        self.orientation_mmap = output  # Output = [x_angle, y_angle, vertical_pos]
        self.input_len = np.shape(input)[0]
        self.last_row = -1

        # Index of highest valid index from last time the buffer counter was reset
        self.last_valid = self.input_len - 1

        # The following are variables to control how the burst is handled due to NaN values
        self.burst_contains_nan = False
        self.burst_is_discarded = False
        self.nan_in_burst = 0

        # Internal storage
        self.processed_input = np.zeros(shape=(self.input_len, 5), dtype=float)
        self.actual_vertical_acceleration = np.zeros(shape=(self.input_len,), dtype=float)
        self.proper_vertical_acceleration = np.zeros(shape=(self.input_len,), dtype=float)
        self.dampened_vertical_velocity = np.zeros(shape=(self.input_len,), dtype=float)
        self.dampened_vertical_position = np.zeros(shape=(self.input_len,), dtype=float)
        self.vertical_acceleration = 0.0
        self.vertical_velocity = 0.0

        self.bias_estimators = np.empty(shape=(9,), dtype=BiasEstimator)
        self.initialize_bias_estimators()
        self.biases = np.empty(shape=(9,), dtype=float)

        # Weights for weighted averages
        self.n_points_for_pos_mean = cfg.n_points_for_pos_mean_initial

        # damping factors to counteract integration drift (adjusted based on current best estimate)

        self.vel_damping = Damping(cfg.vel_damping_factor_initial, cfg.vel_damping_factor_end,
                                   cfg.vel_damping_factor_big, cfg.damping_factor_dampener)
        self.pos_damping = Damping(cfg.vel_damping_factor_initial, cfg.vel_damping_factor_end,
                                   cfg.vel_damping_factor_big, cfg.damping_factor_dampener)

        # Low-pass filtering coefficients
        nyquist_freq = cfg.sampling_rate / 2
        self.low_b, self.low_a = butter(int(1 / (cfg.butterworth_cutoff_rate * cfg.sampling_rate) * nyquist_freq + 0.5),
                                        cfg.butterworth_cutoff_rate, btype='lowpass', output='ba')

        # Variables for generating and storing information on the wave function
        self.wave_function_buffer = np.zeros(shape=(cfg.n_saved_wave_functions, cfg.n_points_for_fft // 2),
                                             dtype=float)
        self.last_fft = -1
        # Pointer points to the last saved wave function
        self.wave_function_buffer_pointer = -1

        # Development mode variables

        if dev_mode:
            # Extended internal memory to examine different internal variables post processing
            self.dev_bank_angle = np.zeros(shape=(self.input_len,), dtype=float)
            self.dev_vertical_velocity = np.zeros(shape=(self.input_len,), dtype=float)
            self.dev_gyro_state = np.zeros(shape=(self.input_len, 2), dtype=float)
            self.dev_acc_state = np.zeros(shape=(self.input_len, 2), dtype=float)

            self.n_bias_updates = np.zeros(shape=(5,), dtype=int)  # Gyro, xy acc, vert acc, vel, pos

            # Biases for each timestep are also kept for examination
            self.acc_bias_array = np.zeros(shape=(self.input_len, 3), dtype=float)
            self.gyro_bias_array = np.zeros(shape=(self.input_len, 2), dtype=float)
            self.vertical_vel_bias_array = np.zeros(shape=(self.input_len,), dtype=float)
            self.vertical_pos_bias_array = np.zeros(shape=(self.input_len,), dtype=float)

            # Some control variables for testing with a purpose of controlling vertical position output
            self.no_vert_pos_bias = False
            self.no_vert_vel_bias = False

            warnings.filterwarnings('error')
        else:
            self.dev_acc_state = None
            self.dev_gyro_state = None

        self.kalman_filter = KalmanFilter(
            processed_input=self.processed_input,
            output=self.orientation_mmap,
            sampling_period=cfg.sampling_period,
            rows_per_kalman_use=cfg.rows_per_kalman_use,
            dev_acc_state=self.dev_acc_state,
            dev_gyro_state=self.dev_gyro_state,
            dev_mode=self.dev_mode)

    def initialize_bias_estimators(self):
        # Acceleration biases
        self.bias_estimators[cfg.acc_x_identifier] = BiasEstimator(cfg.points_between_acc_bias_update, use_moving_average=True,
                                                                   track_bias=self.dev_mode,
                                                                   bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.acc_y_identifier] = BiasEstimator(cfg.points_between_acc_bias_update, use_moving_average=True,
                                                                   track_bias=self.dev_mode,
                                                                   bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.acc_z_identifier] = BiasEstimator(cfg.points_between_acc_bias_update, use_moving_average=True,
                                                                   track_bias=self.dev_mode,
                                                                   bias_tracking_length=self.input_len)

        # Gyro biases
        self.bias_estimators[cfg.gyro_x_identifier] = BiasEstimator(cfg.points_between_gyro_bias_update, use_moving_average=True,
                                                                    track_bias=self.dev_mode,
                                                                    bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.gyro_y_identifier] = BiasEstimator(cfg.points_between_gyro_bias_update, use_moving_average=True,
                                                                    track_bias=self.dev_mode,
                                                                    bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.gyro_z_identifier] = BiasEstimator(cfg.points_between_gyro_bias_update, use_moving_average=True,
                                                                    track_bias=self.dev_mode,
                                                                    bias_tracking_length=self.input_len)

        # Calculated vertical biases
        self.bias_estimators[cfg.vertical_acc_identifier] = BiasEstimator(cfg.points_between_vertical_acc_bias_update,
                                                                          expected_value=cfg.gravitational_constant,
                                                                          use_moving_average=False,
                                                                          track_bias=self.dev_mode,
                                                                          bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.vertical_velocity_identifier] = BiasEstimator(cfg.points_between_vertical_vel_bias_update,
                                                                               use_moving_average=False,
                                                                               track_bias=self.dev_mode,
                                                                               bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.vertical_position_identifier] = BiasEstimator(cfg.points_between_vertical_pos_bias_update,
                                                                               use_moving_average=False,
                                                                               track_bias=self.dev_mode,
                                                                               bias_tracking_length=self.input_len)

    def process(self, number_of_rows: int):
        """
        Tell FloatService to process the next number_of_rows rows in input, starting from last_row + 1.
        :param number_of_rows: Number of input data rows to be processed.
        Format of output: N rows x [x-angle, y-angle, vertical position]
        """
        # TODO: end goal - only create an array of sufficient size from the cyclic array once,
        #  and use it for the remainder of the iteration
        if self.last_row + number_of_rows + 1 <= n_rows:
            start = self.last_row + 1
        else:
            start = 0
            # Information on last actual buffer index is kept
            self.last_valid = self.last_row

            # Previously update_counters_on_buffer_reuse()
            self.last_fft = self.last_fft - self.last_valid - 1
            for bias_estimator in self.bias_estimators:
                bias_estimator.update_counter(self.last_valid)

            self.copy_data_to_last_index_on_buffer_reuse()

        end = start + number_of_rows

        if cfg.use_minibursts:
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
        e_i = min(end, s_i + cfg.miniburst_size)
        while s_i < end:
            self.preprocess_data(start=s_i, end=e_i)
            # Check whether burst is declared discarded
            if not self.burst_is_discarded:
                self.run_processing_iterations(start=s_i, end=e_i)

            s_i += cfg.miniburst_size
            e_i = min(end, s_i + cfg.miniburst_size)

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
        # TODO: possible one-off mistakes with last_valid
        imu_data = MemMapUtils.get_interval_with_min_size(self.imu_mmap, start - cfg.n_points_for_acc_mean, start,
                                                          cfg.n_points_for_acc_mean, self.last_valid)

        for idx, bias_estimator in enumerate(self.bias_estimators[cfg.acc_identifier[0]:cfg.acc_identifier[-1] + 1]):
            bias_estimator.update(imu_data[:, cfg.acc_identifier[idx]], start)

        for idx, bias_estimator in enumerate(self.bias_estimators[cfg.gyro_identifier[0]:cfg.gyro_identifier[-1] + 1]):
            bias_estimator.update(imu_data[:, cfg.gyro_identifier[idx]], start)

        self.biases = np.array(list(bias.value() for bias in self.bias_estimators))

        if self.dev_mode:
            self.acc_bias_array[start - cfg.n_points_for_acc_mean:start] = self.biases[0:3]
            self.gyro_bias_array[start - cfg.n_points_for_acc_mean:start] = self.biases[3:5]

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
        self.processed_input[start: end, 3:5] = Utils.degrees_to_radians(self.processed_input[start:end, 3:5])

        # Transform gyro data and a single axis of accelerometer data so that the input matches mathematical convention
        self.convert_to_right_handed_coords(start=start, end=end)

        # Filtering of both accelerometer and gyroscope data using a low-pass filter
        array_to_filter = MemMapUtils.get_interval_with_min_size(self.processed_input, start, end,
                                                                 cfg.min_filter_size, self.last_valid)

        filtered_array = LowPassFilter.process(array_to_filter, self.low_a, self.low_b, end - start)
        self.processed_input[start:end] = filtered_array

    def run_processing_iterations(self, start: int, end: int):
        for i in range(start, end):
            self.wave_function(row_no=i)

            self.estimate_pose(row_no=i)

            # Adjust velocity and position damping factors
            self.pos_damping.update()
            self.vel_damping.update()

    def estimate_pose(self, row_no: int):
        # A Kalman filter iteration is performed to estimate x- and y-angles,
        # which later makes up the bank angle
        if row_no % cfg.rows_per_kalman_use == 0:
            self.kalman_filter.kalman_iteration(row_no=row_no)
            self.orientation_mmap[row_no, 0:2] = self.kalman_filter.kal_state_post
        else:
            self.estimate_angles_using_gyro(row_no)
            self.kalman_filter.kal_state_post = self.orientation_mmap[row_no, 0:2]

        # Vertical acceleration, velocity and position is estimated and stored internally
        self.estimate_vertical_acceleration(row_no=row_no)
        self.estimate_vertical_velocity(row_no=row_no)
        self.estimate_vertical_position(row_no=row_no)

    def estimate_angles_using_gyro(self, row_no: int):
        self.orientation_mmap[row_no, 0:2] = \
            self.orientation_mmap[row_no - 1, 0:2] + \
            cfg.sampling_period * self.processed_input[row_no, 3:5] \
            * np.flip(np.cos(self.orientation_mmap[row_no - 1, 0:2]))
        if self.dev_mode:
            self.dev_gyro_state[row_no] = self.dev_gyro_state[row_no - 1] + \
                                          cfg.sampling_period * self.processed_input[row_no, 3:5] \
                                          * np.flip(np.cos(self.dev_gyro_state[row_no - 1]))

    def estimate_vertical_acceleration(self, row_no: int):
        """
        Estimates vertical acceleration of the sensor.
        Rotates the set of acceleration vectors produced by the angles found in angle estimation, then uses the
        z-component of the resulting set as the vertical acceleration of the sensor.
        """

        a_vector = self.processed_input[row_no, 0:3]
        inverse_orientation = [-self.orientation_mmap[row_no, 0], -self.orientation_mmap[row_no, 1], 0.0]
        global_a_vector = Rotations.rotate_system(sys_ax=a_vector, sensor_angles=inverse_orientation)

        self.vertical_acceleration = - (global_a_vector[2] + cfg.gravitational_constant)

        # Store information on vertical acceleration for wave function estimation
        self.actual_vertical_acceleration[row_no] = self.vertical_acceleration

    def estimate_vertical_velocity(self, row_no: int):
        # Vertical velocity is updated using current vertical acceleration
        self.dampened_vertical_velocity[row_no] = self.dampened_vertical_velocity[row_no - 1] + \
                                                  cfg.sampling_period * self.vertical_acceleration
        # Vertical velocity is adjusted by a damping factor to compensate for integration drift
        self.dampened_vertical_velocity[row_no] -= self.dampened_vertical_velocity[row_no] * self.vel_damping.value()

        # If a number of rows equal to or greater than the threshold for updating vertical vel bias has been traversed,
        # update vertical velocity bias.
        vertical_vel = MemMapUtils.get_interval_with_min_size(self.dampened_vertical_velocity,
                                                              row_no - cfg.n_points_for_vel_mean, row_no,
                                                              cfg.n_points_for_vel_mean, self.last_valid)

        self.bias_estimators[cfg.vertical_velocity_identifier].update(vertical_vel, row_no)
        self.biases[cfg.vertical_velocity_identifier] = self.bias_estimators[cfg.vertical_velocity_identifier].value()

        vert_velocity_bias = self.bias_estimators[cfg.vertical_velocity_identifier].value()
        # Vertical velocity is adjusted by bias and stored internally
        self.vertical_velocity = self.dampened_vertical_velocity[row_no] - vert_velocity_bias
        # In development mode, store information on vertical velocity for each time step
        if self.dev_mode:
            self.dev_vertical_velocity[row_no] = self.vertical_velocity
            self.vertical_vel_bias_array[row_no] = vert_velocity_bias

    def estimate_vertical_position(self, row_no: int):
        # Vertical position is updated using current vertical velocity
        self.dampened_vertical_position[row_no] = self.dampened_vertical_position[row_no - 1] + \
                                                  cfg.sampling_period * self.vertical_velocity
        # Vertical position is adjusted by a damping factor to compensate for integration drift
        self.dampened_vertical_position[row_no] -= self.dampened_vertical_position[row_no] * self.pos_damping.value()

        # If a number of rows equal to or greater than the threshold for updating vertical pos bias has been traversed,
        # update vertical position bias.
        vertical_positions = MemMapUtils.get_interval_with_min_size(self.dampened_vertical_position,
                                                                    row_no - cfg.n_points_for_vel_mean, row_no,
                                                                    cfg.n_points_for_vel_mean, self.last_valid)

        self.bias_estimators[cfg.vertical_position_identifier].update(vertical_positions, row_no)
        self.biases[cfg.vertical_position_identifier] = self.bias_estimators[cfg.vertical_position_identifier].value()

        vertical_position_bias = self.bias_estimators[cfg.vertical_position_identifier].value()
        # Vertical position is adjusted by the bias and stored as output
        self.orientation_mmap[row_no, 2] = self.dampened_vertical_position[row_no] - vertical_position_bias

        if self.dev_mode:
            self.vertical_pos_bias_array[row_no] = vertical_position_bias

    def copy_data_to_last_index_on_buffer_reuse(self):
        # Since some methods use data indexed in [row_no-1], data from [self.last_valid] is copied to
        # [-1]. One could also check for start==0 in each of these methods but this is less expensive.
        self.dampened_vertical_velocity[-1] = self.dampened_vertical_velocity[self.last_valid]
        self.dampened_vertical_position[-1] = self.dampened_vertical_position[self.last_valid]
        self.orientation_mmap[-1] = self.orientation_mmap[self.last_valid]
        if self.dev_mode:
            self.dev_gyro_state[-1] = self.dev_gyro_state[self.last_valid]

    def postprocess_output(self, start: int, end: int):
        if not self.dev_mode or cfg.use_output_filtering:
            array_to_filter = MemMapUtils.get_interval_with_min_size(self.orientation_mmap[:, 2], start, end,
                                                                     cfg.min_filter_size, self.last_valid)

            filtered_array = LowPassFilter.process(array_to_filter, self.low_a, self.low_b, end - start)
            self.orientation_mmap[start:end, 2] = filtered_array

    def wave_function(self, row_no: int):
        # Check if it is time to perform a Fourier transform
        if row_no - self.last_fft >= cfg.n_points_between_fft:
            # If so, check whether fft can be performed on a single stretch of the buffer
            if MemMapUtils.historic_data_is_contiguous(cfg.n_points_for_fft, end_row_of_data=self.last_row + 1):
                # First, we do a transform of the vertical acceleration signal
                fourier_transform = fft(self.actual_vertical_acceleration[self.last_row + 1 - cfg.n_points_for_fft:
                                                                          self.last_row + 1])
            else:
                last_indices, first_indices = MemMapUtils.patched_buffer_indices(cfg.n_points_for_fft,
                                                                                 current_row=self.last_row + 1,
                                                                                 prev_last_index=self.last_valid)
                temp_buffer = np.concatenate([self.actual_vertical_acceleration[last_indices[0]:last_indices[1]],
                                              self.actual_vertical_acceleration[first_indices[0]:first_indices[1]]])
                fourier_transform = fft(temp_buffer)

            coeff_array = 2.0 / cfg.n_points_for_fft * np.abs(fourier_transform[0:cfg.n_points_for_fft // 2])
            self.save_wave_function(coefficient_array=coeff_array)

            self.last_fft = row_no
            if cfg.fft_aided_bias:
                self.update_position_bias_window_size()

    def save_wave_function(self, coefficient_array: np.ndarray):
        # Update buffer pointer
        self.wave_function_buffer_pointer = (self.wave_function_buffer_pointer + 1) % cfg.n_saved_wave_functions
        # Store wave function coefficients in buffer
        #   - In case the buffer is too small to perform FFT using requested number of points we specify that
        #   - self.wave_function_buffer is filled by the number of data points in coefficient_array to avoid a
        #   - ValueError.
        self.wave_function_buffer[self.wave_function_buffer_pointer][0:len(coefficient_array)] = coefficient_array

    def update_position_bias_window_size(self):
        # TODO: Test how this method affects vert pos bias window during NaN-packages, especially on time usage due to
        #       increased window size. If this is a problem, package discarding must be strengthened with using the
        #       main frequency from the fft
        """
        Use information from wave function to control mean window size for vertical position
        """

        frequencies = np.linspace(0.0, cfg.sampling_rate / 2, cfg.n_points_for_fft // 2)
        top_frequency = frequencies[np.argmax(self.wave_function_buffer[self.wave_function_buffer_pointer])]

        if top_frequency == 0.0:
            return
        new_n_data_points = int(cfg.sampling_rate * 1 / top_frequency)
        # Update bias control, as long as the period of the greatest amplitude is greater than some threshold
        if new_n_data_points > cfg.min_points_for_pos_mean:
            self.n_points_for_pos_mean = new_n_data_points
            # print(f'When last row was {self.last_row}, pos bias window size set to {self.n_points_for_pos_mean}')

    def set_processed_acc_input(self, start: int, end: int):
        self.processed_input[start:end, 0:2] = (self.imu_mmap[start:end, 0:2] - self.biases[0:2]) * cfg.gravitational_constant

        self.processed_input[start:end, 2] = self.imu_mmap[start:end, 2] * cfg.gravitational_constant

    def set_processed_gyro_input(self, start: int, end: int):
        """
        Adjusts gyro data using a sensor bias.
        :param start: Index that slices the buffer part that is to be adjusted.
        :param end: Index that slices the buffer part that is to be adjusted.
        """
        self.processed_input[start: end, 3:5] = self.imu_mmap[start: end, 3:5] - self.biases[3:5]

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
            if np.any(np.isnan(self.imu_mmap[i, 0:3])):
                # Check if last_useable has been assigned
                if np.any(np.isnan(last_useable)):
                    # If last_useable has not been assigned, the current value is the first NaN value of
                    # some number of NaN values. Therefore, assign last_useable as the previous value
                    last_useable = self.imu_mmap[i - 1, 0:3]
                    last_useable_i = i - 1
                    # Taking into account that self.input[i-1] might also be a NaN value (in case this is the beginning
                    # of a burst and the last value of the previous burst was NaN), a generic set of values may need
                    # to be used
                    if np.any(np.isnan(last_useable)):
                        last_useable = np.array([0.0, 0.0, -1.0])  # Generic set of [accX, accY, accZ]

                    # If first_useable has also been assigned, all values in
                    # self.input[first_useable: i] can be copied directly to self.processed_input
                    if not np.any(np.isnan(first_useable)):
                        self.processed_input[first_useable_i:i, 0:3] = self.imu_mmap[first_useable_i: i, 0:3]
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
                    next_useable = self.imu_mmap[i, 0:3]
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
                    first_useable = self.imu_mmap[i, 0:3]
                    first_useable_i = i

        # When the entire row is checked for NaN, there may still be the case that the last values are
        # NaN values. In this case we cannot interpolate anything and instead we perform a simple
        # extrapolation, by copying last_useable into each of the values.
        if not np.any(np.isnan(last_useable)):
            self.processed_input[last_useable_i + 1:end, 0:3] = last_useable
        else:
            # If last_useable is not set, this bursts ends with non-NaN values. These can be inserted into
            # self.processed_input
            self.processed_input[first_useable_i:end, 0:3] = self.imu_mmap[first_useable_i:end, 0:3]
        # Finally, after having copied/inter/extrapolated input to processed_input, the data is adjusted according
        # to input bias
        self.processed_input[start:end, 0:2] -= self.bias_estimators[0:2]
        self.processed_input[start:end, 0:3] *= cfg.gravitational_constant

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
            if np.any(np.isnan(self.imu_mmap[i, 3:5])):
                # Check if last_useable has been assigned
                if np.any(np.isnan(last_useable)):
                    # If last_useable has not been assigned, the current value is the first NaN value of
                    # some number of NaN values. Therefore, assign last_useable as the previous value
                    last_useable = self.imu_mmap[i - 1, 3:5]
                    last_useable_i = i - 1

                    # If first_useable has also been assigned, all values in
                    # self.input[first_useable: i] can be copied directly to self.processed_input
                    if not np.any(np.isnan(first_useable)):
                        self.processed_input[first_useable_i:i, 3:5] = self.imu_mmap[first_useable_i: i, 3:5]
                        first_useable = float('nan')

                    # Taking into account that self.input[i-1] might also be a NaN value (in case this is the beginning
                    # of a burst and the last value of the previous burst was NaN), a generic set of values may need
                    # to be used
                    if np.any(np.isnan(last_useable)):
                        last_useable = np.array([0.0, 0.0])  # Generic set of [gyroX, gyroY]

                    # If last_useable has already been assigned, the current value is somewhere within a row of
                    # several NaN. Thus, nothing is done.
            else:
                # If the current value is not NaN, check if last_useable has been assigned
                # If last_useable has NOT been assigned, the current value is somewhere within a row of
                # actual values. Nothing needs to be done
                if not np.any(np.isnan(last_useable)):
                    # If last_useable has in fact been assigned, the current value is the first actual value
                    # after some number of NaN values. Therefore, next_useable is assigned as current value.
                    next_useable = self.imu_mmap[i, 3:5]
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
                    first_useable = self.imu_mmap[i, 3:5]
                    first_useable_i = i

        # When the entire row is checked for NaN, there may still be the case that the last values are
        # NaN values. In this case we cannot interpolate anything and instead we perform a simple
        # extrapolation, by copying last_useable into each of the values.
        if not np.any(np.isnan(last_useable)):
            self.processed_input[last_useable_i + 1:end, 3:5] = last_useable
        else:
            # If last_useable is not set, this bursts ends with non-NaN values. These can be inserted into
            # self.processed_input
            self.processed_input[first_useable_i:end, 3:5] = self.imu_mmap[first_useable_i:end, 3:5]

        # Finally, after having copied/inter/extrapolated input to processed_input, the data is adjusted according
        # to input bias
        self.processed_input[start:end, 3:5] -= self.bias_estimators[3:5]

    def convert_to_right_handed_coords(self, start: int, end: int):
        """
        From testing we know that the sensor output does not match mathematical convention. This method uses information
        provided in config.py to reverse all gyro axes and exactly one of the accelerometer axes.
        :param start: First index of input to be reversed
        :param end: First index not to be reversed
        """
        if cfg.perform_axis_reversal:
            # Reverse gyro data
            self.processed_input[start: end, 3:5] *= cfg.gyro_reversal_coefficient
            # Reverse accelerometer data. Since positive direction of accelerometer and gyro is linked, the same gyro axis
            # must be reversed again.
            self.processed_input[start: end, cfg.axis_reversal_index] *= cfg.axis_reversal_coefficient
            if cfg.axis_reversal_index < 2:
                self.processed_input[start: end, 3 + cfg.axis_reversal_index] *= cfg.axis_reversal_coefficient

    def nan_handling(self, start: int, end: int):
        # Method uses assumtion 1
        if np.any(np.isnan(self.imu_mmap[start:end, 0])):
            self.burst_contains_nan = True
            # If in fact all values of any DoF are NaN, the entire processing of this burst should be handled in a
            # separate method
            self.nan_in_burst = np.count_nonzero(np.isnan(self.imu_mmap[start:end, 0]))
            if self.nan_in_burst > int((end - start) * cfg.discard_burst_nan_threshold):
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
        self.processed_input[start:end, 0:5] = np.array([0.0, 0.0, -cfg.gravitational_constant, 0.0, 0.0])
        self.actual_vertical_acceleration[start:end] = 0.0
        self.dampened_vertical_velocity[start:end] = self.bias_estimators[cfg.vertical_velocity_identifier]
        self.dampened_vertical_position[start:end] = self.bias_estimators[cfg.vertical_position_identifier]
        # self.vertical_acceleration and self.vertical_velocity are left unhandled since
        # they are calculated before use anyways

        # Set dev_mode storage
        if self.dev_mode:
            self.dev_bank_angle[start:end] = 0.0
            self.dev_vertical_velocity[start:end] = 0.0
            self.dev_gyro_state[start:end] = 0.0
            self.dev_acc_state[start:end] = 0.0
            self.gyro_bias_array[start:end] = np.array(bias.value() for bias in self.bias_estimators[0:2])
            self.acc_bias_array[start:end] = np.array([0.0, 0.0, 0.0])
            self.vertical_vel_bias_array[start:end] = self.bias_estimators[cfg.vertical_velocity_identifier].value()
            self.vertical_pos_bias_array[start:end] = self.bias_estimators[cfg.vertical_position_identifier].value()

        # Set output
        self.orientation_mmap[start:end] = np.array([0.0, 0.0, 0.0])

        # Declare burst as discarded
        self.burst_is_discarded = True

        # If the burst is discarded, one of the measures taken to quickly adjust to incoming data is to increase
        # the damping factors of speed and velocity. These factors are then themselves dampened subsequently.
        self.pos_damping.boost()
        self.vel_damping.boost()

    def get_position_averaging_weights(self):
        """
        returns some array of float numbers, usable as weights for a weighted average.
        This method exists because the length of the average window for position corrections varies with time.
        """
        # return np.linspace(0.0, 1.0, self.n_points_for_pos_mean)
        return np.ones(shape=(self.n_points_for_pos_mean,))
