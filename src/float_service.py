"""
Assumptions:
1. Either all measurements in a row are NaN values, or all measurements are valid

Explainations:
Pose - Both position and orientation of an object
DOF/dof - Degree of freedom, here used to describe data from one of the six sensor readings (acc x-y-z, gyro x-y-z)
"""

import warnings

import numpy as np
from scipy.fft import fft
from scipy.signal import butter

import config as cfg
from utils.bias_estimator import BiasEstimator
from utils.damping import Damping
from utils.kalman_filter import KalmanFilter
from utils.low_pass_filter import LowPassFilter
from utils.mem_map_utils import MemMapUtils
from utils.nan_handling import NanHandling
from utils.rotations import Rotations
from utils.utils import Utils

# The operating buffer size of FloatService should be a global variable so that it may be set once by some other process
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

        # How the burst is handled due to NaN values
        self.burst_is_discarded = False

        # Internal storage
        self.processed_input = np.zeros(shape=(self.input_len, 5), dtype=float)
        self.actual_vertical_acceleration = np.zeros(shape=(self.input_len,), dtype=float)
        # TODO: proper vertical acceleration not used
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

        self.dev_acc_state = None
        self.dev_gyro_state = None

        # Development mode variables
        if dev_mode:
            # Extended internal memory to examine different internal variables post processing
            self.dev_vertical_velocity = np.zeros(shape=(self.input_len,), dtype=float)
            self.dev_gyro_state = np.zeros(shape=(self.input_len, 2), dtype=float)
            self.dev_acc_state = np.zeros(shape=(self.input_len, 2), dtype=float)

            # TODO: Not currently used
            self.n_bias_updates = np.zeros(shape=(5,), dtype=int)  # Gyro, xy acc, vert acc, vel, pos

            # Biases for each timestep are also kept for examination
            self.acc_bias_array = np.zeros(shape=(self.input_len, 3), dtype=float)
            self.gyro_bias_array = np.zeros(shape=(self.input_len, 2), dtype=float)
            self.vertical_vel_bias_array = np.zeros(shape=(self.input_len,), dtype=float)
            self.vertical_pos_bias_array = np.zeros(shape=(self.input_len,), dtype=float)

            warnings.filterwarnings('error')

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
        for acc_identifier in cfg.acc_identifiers:
            self.bias_estimators[acc_identifier] = BiasEstimator(cfg.points_between_acc_bias_update,
                                                                 use_moving_average=True,
                                                                 track_bias=self.dev_mode,
                                                                 bias_tracking_length=self.input_len)

        # Gyro biases
        for gyro_identifier in cfg.gyro_identifiers:
            self.bias_estimators[gyro_identifier] = BiasEstimator(cfg.points_between_gyro_bias_update,
                                                                  use_moving_average=True,
                                                                  track_bias=self.dev_mode,
                                                                  bias_tracking_length=self.input_len)

        # Calculated vertical biases
        self.bias_estimators[cfg.vertical_acc_identifier] = BiasEstimator(cfg.points_between_vertical_acc_bias_update,
                                                                          expected_value=cfg.gravitational_constant,
                                                                          use_moving_average=False,
                                                                          track_bias=self.dev_mode,
                                                                          bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.vertical_velocity_identifier] = BiasEstimator(
            cfg.points_between_vertical_vel_bias_update,
            use_moving_average=False,
            track_bias=self.dev_mode,
            bias_tracking_length=self.input_len)
        self.bias_estimators[cfg.vertical_position_identifier] = BiasEstimator(
            cfg.points_between_vertical_pos_bias_update,
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
            self.handle_buffer_wrapping()

        end = start + number_of_rows

        if cfg.use_minibursts:
            step = cfg.miniburst_size
        else:
            step = number_of_rows

        for start_i in range(start, end, step):
            end_i = min(end, start_i + step)
            self.preprocess_data(start_i, end_i)

            # Check whether burst is declared discarded
            if not self.burst_is_discarded:
                self.run_processing_iterations(start_i, end_i)

        if not self.burst_is_discarded:
            self.postprocess_output(start, end)

        self.last_row = end - 1

    def handle_buffer_wrapping(self):
        # Information on last actual buffer index is kept
        self.last_valid = self.last_row
        # Previously update_counters_on_buffer_reuse()
        self.last_fft = self.last_fft - self.last_valid - 1
        for bias_estimator in self.bias_estimators:
            bias_estimator.update_counter(self.last_valid)
        self.copy_data_to_last_index_on_buffer_reuse()

    def preprocess_data(self, start: int, end: int):
        """
        NaN-handling
        Bias updates
        Set processed input
        Data transformations
        Outlier correction/smoothing
        """

        # Check whether burst should be discarded because of NaN values
        if NanHandling.should_discard_burst(self.imu_mmap, start, end):
            self.discard_burst(start, end)
            self.burst_is_discarded = True
            return
        else:
            self.burst_is_discarded = False

        # Update gyroscope and accelerometer bias
        # TODO: possible one-off mistakes with last_valid
        imu_data = MemMapUtils.get_interval_with_min_size(self.imu_mmap, start - cfg.n_points_for_acc_mean, start,
                                                          cfg.n_points_for_acc_mean, self.last_valid)

        for idx, bias_estimator in enumerate(self.bias_estimators[cfg.acc_identifiers[0]:cfg.acc_identifiers[-1] + 1]):
            bias_estimator.update(imu_data[:, cfg.acc_identifiers[idx]], start)

        for idx, bias_estimator in enumerate(
                self.bias_estimators[cfg.gyro_identifiers[0]:cfg.gyro_identifiers[-1] + 1]):
            bias_estimator.update(imu_data[:, cfg.gyro_identifiers[idx]], start)

        self.biases = np.array(list(bias.value() for bias in self.bias_estimators))

        if self.dev_mode:
            self.acc_bias_array[start - cfg.n_points_for_acc_mean:start] = self.biases[0:3]
            self.gyro_bias_array[start - cfg.n_points_for_acc_mean:start] = self.biases[3:5]

        # If nan_handling() detected any NaN-values in the burst without discarding the burst, separate methods for
        # inserting processed input are used
        if NanHandling.burst_contains_nan(self.imu_mmap, start, end):
            interpolated_input = NanHandling.interpolate_missing_values(self.imu_mmap, start, end)
        else:
            interpolated_input = self.imu_mmap[start:end]

        self.set_processed_gyro_input(interpolated_input, start, end)
        self.set_processed_acc_input(interpolated_input, start, end)

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
            self.kalman_filter.iterate(row_no=row_no)
            self.orientation_mmap[row_no, 0:2] = self.kalman_filter.get_state_estimate()
        else:
            self.estimate_angles_using_gyro(row_no)
            self.kalman_filter.state_posteriori = self.orientation_mmap[row_no, 0:2]

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
        if self.dev_mode and cfg.ignore_vertical_velocity_bias:
            self.vertical_velocity = self.dampened_vertical_velocity[row_no]
        else:
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
        if self.dev_mode and cfg.ignore_vertical_position_bias:
            self.orientation_mmap[row_no, 2] = self.dampened_vertical_position[row_no]
        else:
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

            self.orientation_mmap[start:end, 2] = LowPassFilter.process(array_to_filter, self.low_a, self.low_b,
                                                                        end - start)

    def wave_function(self, row_no: int):
        # Check if it is time to perform a Fourier transform
        if row_no - self.last_fft < cfg.n_points_between_fft:
            return

        # TODO: why is self.last_row + 1 used here instead of row_no ?
        # First, we do a transform of the vertical acceleration signal
        vertical_acceleration = MemMapUtils.get_interval_with_min_size(self.actual_vertical_acceleration,
                                                                       self.last_row + 1 - cfg.n_points_for_fft,
                                                                       self.last_row + 1, cfg.n_points_for_fft,
                                                                       self.last_valid)

        fourier_transform = fft(vertical_acceleration)

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

    def set_processed_acc_input(self, imu_array: np.array, start: int, end: int):
        self.processed_input[start:end, 0:2] = (imu_array[:, 0:2] - self.biases[
                                                                    0:2]) * cfg.gravitational_constant
        self.processed_input[start:end, 2] = imu_array[:, 2] * cfg.gravitational_constant

    def set_processed_gyro_input(self, imu_array: np.array, start: int, end: int):
        """
        Adjusts gyro data using a sensor bias.
        :param imu_array: Array containing the processed imu measurements to use
        :param start: Index that slices the buffer part that is to be adjusted.
        :param end: Index that slices the buffer part that is to be adjusted.
        """
        self.processed_input[start:end, 3:5] = imu_array[:, 3:5] - self.biases[3:5]

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
            self.dev_vertical_velocity[start:end] = 0.0
            self.dev_gyro_state[start:end] = 0.0
            self.dev_acc_state[start:end] = 0.0
            self.gyro_bias_array[start:end] = np.array(bias.value() for bias in self.bias_estimators[0:2])
            self.acc_bias_array[start:end] = np.array([0.0, 0.0, 0.0])
            self.vertical_vel_bias_array[start:end] = self.bias_estimators[cfg.vertical_velocity_identifier].value()
            self.vertical_pos_bias_array[start:end] = self.bias_estimators[cfg.vertical_position_identifier].value()

        # Set output
        self.orientation_mmap[start:end] = np.array([0.0, 0.0, 0.0])

        # If the burst is discarded, one of the measures taken to quickly adjust to incoming data is to increase
        # the damping factors of speed and velocity. These factors are then themselves dampened subsequently.
        self.pos_damping.boost()
        self.vel_damping.boost()
