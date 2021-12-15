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
from utils.cyclic_array import CyclicArray
from utils.damping import Damping
from utils.kalman_filter import KalmanFilter
from utils.low_pass_filter import LowPassFilter
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
        np.seterr(all='raise')
        self.name = name
        self.dev_mode = dev_mode
        self.imu_mmap = input  # Input = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        self.orientation_mmap = output  # Output = [x_angle, y_angle, vertical_pos]
        self.input_length = np.shape(input)[0]
        self.last_row = -1

        self.points_since_wave_update = 0

        # Index of highest valid index from last time the buffer counter was reset
        self.last_valid = self.input_length - 1

        # How the burst is handled due to NaN values
        self.burst_is_discarded = False

        self.bias_estimators = np.empty(shape=(9,), dtype=BiasEstimator)
        self.initialize_bias_estimators()
        self.biases = np.empty(shape=(9,), dtype=float)

        # Weights for weighted averages
        self.points_for_mean = cfg.points_for_mean

        # Tracking of previous values
        self.n_inputs = max(self.points_for_mean[cfg.acc_identifier], self.points_for_mean[cfg.gyro_identifier])
        self.input_tracker = CyclicArray(self.input_length, dimensions=6)
        self.processed_input_tracker = CyclicArray(self.input_length, dimensions=5)
        self.vertical_acceleration_tracker = CyclicArray(self.input_length)
        self.vertical_velocity_tracker = CyclicArray(self.input_length)
        self.vertical_velocity_bias_adjusted_tracker = CyclicArray(self.input_length)
        self.vertical_position_tracker = CyclicArray(self.input_length)
        self.orientations_tracker = CyclicArray(self.input_length, dimensions=3)

        # damping factors to counteract integration drift (adjusted based on current best estimate)
        self.vel_damping = Damping(cfg.vel_damping_factor_initial, cfg.vel_minimum_damping,
                                   cfg.vel_damping_factor_big, cfg.damping_factor_dampener)
        self.pos_damping = Damping(cfg.vel_damping_factor_initial, cfg.vel_minimum_damping,
                                   cfg.vel_damping_factor_big, cfg.damping_factor_dampener)

        # Low-pass filtering coefficients
        nyquist_freq = cfg.sampling_rate / 2
        self.low_b, self.low_a = butter(int(1 / (cfg.butterworth_cutoff_rate * cfg.sampling_rate) * nyquist_freq + 0.5),
                                        cfg.butterworth_cutoff_rate, btype='lowpass', output='ba')

        # Variables for generating and storing information on the wave function
        self.wave_function_buffer = np.zeros(shape=(cfg.n_saved_wave_functions, cfg.n_points_for_fft // 2),
                                             dtype=float)
        # Pointer points to the last saved wave function
        self.wave_function_buffer_pointer = -1

        self.dev_acc_state = None
        self.dev_gyro_state = None

        # Development mode variables
        if dev_mode:
            self.processed_input = np.zeros(shape=(self.input_length, 5), dtype=float)
            self.vertical_acceleration = np.zeros(shape=(self.input_length,), dtype=float)
            self.vertical_velocity = np.zeros(shape=(self.input_length,), dtype=float)
            self.vertical_position = np.zeros(shape=(self.input_length,), dtype=float)

            self.vertical_velocity_dampened = np.zeros(shape=(self.input_length,), dtype=float)
            self.vertical_position_dampened = np.zeros(shape=(self.input_length,), dtype=float)

            # Extended internal memory to examine different internal variables post processing
            self.dev_vertical_velocity = np.zeros(shape=(self.input_length,), dtype=float)
            self.dev_gyro_state = np.zeros(shape=(self.input_length, 2), dtype=float)
            self.dev_acc_state = np.zeros(shape=(self.input_length, 2), dtype=float)

            # Biases for each time step are also kept for examination
            self.acc_bias_array = np.zeros(shape=(self.input_length, 3), dtype=float)
            self.gyro_bias_array = np.zeros(shape=(self.input_length, 2), dtype=float)
            self.vertical_vel_bias_array = np.zeros(shape=(self.input_length,), dtype=float)
            self.vertical_pos_bias_array = np.zeros(shape=(self.input_length,), dtype=float)

            warnings.filterwarnings('error')

        self.kalman_filter = KalmanFilter(
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
                                                                 allow_nan=cfg.allow_nan_for_bias,
                                                                 track_bias=self.dev_mode)

        # Gyro biases
        for gyro_identifier in cfg.gyro_identifiers:
            self.bias_estimators[gyro_identifier] = BiasEstimator(cfg.points_between_gyro_bias_update,
                                                                  use_moving_average=True,
                                                                  allow_nan=cfg.allow_nan_for_bias,
                                                                  track_bias=self.dev_mode)

        # Calculated vertical biases
        self.bias_estimators[cfg.vertical_acc_identifier] = BiasEstimator(cfg.points_between_vertical_acc_bias_update,
                                                                          expected_value=cfg.gravitational_constant,
                                                                          use_moving_average=False,
                                                                          track_bias=self.dev_mode)
        self.bias_estimators[cfg.vertical_velocity_identifier] = BiasEstimator(
            cfg.points_between_vertical_vel_bias_update,
            use_moving_average=False,
            track_bias=self.dev_mode)
        self.bias_estimators[cfg.vertical_position_identifier] = BiasEstimator(
            cfg.points_between_vertical_pos_bias_update,
            use_moving_average=False,
            track_bias=self.dev_mode)

    def process(self, number_of_rows: int):
        """
        Tell FloatService to process the next number_of_rows rows in input, starting from last_row + 1.
        :param number_of_rows: Number of input data rows to be processed.
        Format of output: N rows x [x-angle, y-angle, vertical position]
        """
        orientations = np.empty(shape=(number_of_rows, 3))

        start, end, step = self.get_process_step(number_of_rows)
        for start_i in range(start, end, step):

            end_i = min(end, start_i + step)
            current_input = self.imu_mmap[start_i:end_i]
            self.input_tracker.enqueue_n(current_input)

            processed_input = self.preprocess_data(current_input)
            if not self.burst_is_discarded:
                self.processed_input_tracker.enqueue_n(processed_input)
            else:
                self.discard_burst(start_i, end_i)
                continue

            if self.dev_mode:
                self.processed_input[start_i:end_i] = processed_input
                self.acc_bias_array[start - self.points_for_mean[cfg.acc_identifier]:start] = self.biases[0:3]
                self.gyro_bias_array[start - self.points_for_mean[cfg.gyro_identifier]:start] = self.biases[3:5]

            orientations[start_i - start:end_i - start] = self.run_processing_iterations(processed_input, start_i)

        if not self.burst_is_discarded:
            orientations = self.post_process_output(orientations)
            self.orientations_tracker.enqueue_n(orientations)

            if self.dev_mode:
                self.orientation_mmap[start:end] = orientations

        self.last_row = end - 1

    def get_process_step(self, number_of_rows):
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
        return start, end, step

    def preprocess_data(self, current_input: np.array):
        """
        NaN-handling
        Bias updates
        Set processed input
        Data transformations
        Outlier correction/smoothing
        """

        # Check whether burst should be discarded because of NaN values
        if NanHandling.should_discard_burst(current_input):
            self.burst_is_discarded = True
            return None
        else:
            self.burst_is_discarded = False

        # Update gyroscope and accelerometer bias
        input_length = max(self.points_for_mean[cfg.acc_identifier], self.points_for_mean[cfg.gyro_identifier])
        imu_data = np.empty(shape=(input_length, 6))

        imu_history_fetched = False
        for idx, bias_estimator in enumerate(self.bias_estimators[cfg.acc_identifiers[0]:cfg.acc_identifiers[-1] + 1]):
            bias_estimator.update_counter(current_input.shape[0])

            if bias_estimator.should_update():
                if not imu_history_fetched:
                    imu_history_fetched = True
                    imu_data = self.input_tracker.get_latest_n(input_length)
                bias_estimator.update(imu_data[-self.points_for_mean[cfg.acc_identifier]:, cfg.acc_identifiers[idx]])

        for idx, bias_estimator in enumerate(
                self.bias_estimators[cfg.gyro_identifiers[0]:cfg.gyro_identifiers[-1] + 1]):
            bias_estimator.update_counter(current_input.shape[0])

            if bias_estimator.should_update():
                if not imu_history_fetched:
                    imu_history_fetched = True
                    imu_data = self.input_tracker.get_latest_n(input_length)
                bias_estimator.update(imu_data[-self.points_for_mean[cfg.gyro_identifier]:, cfg.gyro_identifiers[idx]])

        self.biases = np.array(list(bias.value() for bias in self.bias_estimators))

        # If nan_handling() detected any NaN-values in the burst without discarding the burst, separate methods for
        # inserting processed input are used
        if NanHandling.burst_contains_nan(current_input):
            current_input = NanHandling.interpolate_missing_values(current_input)

        processed_acc = self.get_processed_acc_input(current_input)
        processed_gyro = self.get_processed_gyro_input(current_input)

        # Convert angular velocities from deg/s to rad/s
        processed_gyro = Utils.degrees_to_radians(processed_gyro)

        # Transform gyro data and a single axis of accelerometer data so that the input matches mathematical convention
        processed_input = self.convert_to_right_handed_coords(np.concatenate((processed_acc, processed_gyro), axis=1))

        input_size = current_input.shape[0]

        if input_size < cfg.min_filter_size:
            remaining_size = cfg.min_filter_size - input_size
            processed_input = np.concatenate(
                (self.processed_input_tracker.get_latest_n(remaining_size), processed_input))

        return LowPassFilter.process(processed_input, self.low_a, self.low_b, input_size)

    def run_processing_iterations(self, processed_inputs: np.array, start_row: int):

        orientations = np.empty((processed_inputs.shape[0], 3))
        for row in range(processed_inputs.shape[0]):
            # TODO: can be optimized
            prev_accelerations = self.vertical_acceleration_tracker.get_latest_n(
                self.points_for_mean[cfg.acc_identifier])
            self.wave_function(prev_accelerations)

            orientations[row] = self.estimate_pose(processed_inputs[row], start_row + row)
            self.orientations_tracker.enqueue(orientations[row])

            # Adjust velocity and position damping factors
            self.pos_damping.update()
            self.vel_damping.update()

        return orientations

    def estimate_pose(self, processed_input: np.array, row_number: int):
        # A Kalman filter iteration is performed to estimate x- and y-angles,
        # which later makes up the bank angle
        orientation = np.empty(3)
        if row_number % cfg.rows_per_kalman_use == 0:
            self.kalman_filter.iterate(processed_input)
            # self.orientation_mmap[row_no, 0:2]
            orientation[0:2] = self.kalman_filter.get_state_estimate()
        else:
            prev_orientation = self.orientations_tracker.get_tail()
            orientation[0:2] = self.estimate_angles_using_gyro(processed_input, prev_orientation)
            self.kalman_filter.state_posteriori = orientation[0:2]

            if self.dev_mode:
                self.dev_gyro_state[row_number] = self.dev_gyro_state[row_number - 1] + \
                                                  cfg.sampling_period * self.processed_input[row_number, 3:5] \
                                                  * np.flip(np.cos(self.dev_gyro_state[row_number - 1]))

        # Vertical acceleration, velocity and position is estimated and stored internally
        vertical_acceleration = self.estimate_vertical_acceleration(processed_input, orientation[0:2])

        vertical_velocity = self.estimate_vertical_velocity(vertical_acceleration)
        vertical_position = self.estimate_vertical_position(vertical_velocity)

        orientation[2] = vertical_position
        self.orientations_tracker.enqueue(orientation)

        # In development mode, store information
        if self.dev_mode:
            self.vertical_acceleration[row_number] = vertical_acceleration
            self.vertical_velocity[row_number] = vertical_velocity
            self.vertical_velocity_dampened[row_number] = vertical_velocity + self.biases[
                cfg.vertical_velocity_identifier]

            self.vertical_position[row_number] = vertical_position
            self.vertical_position_dampened[row_number] = vertical_position + self.biases[
                cfg.vertical_position_identifier]

            self.dev_vertical_velocity[row_number] = vertical_velocity

            self.vertical_vel_bias_array[row_number] = self.biases[cfg.vertical_velocity_identifier]

        if self.dev_mode:
            self.vertical_pos_bias_array[row_number] = self.biases[cfg.vertical_position_identifier]

        ##########################################################################

        return orientation

    @staticmethod
    def estimate_angles_using_gyro(processed_input: np.array, prev_orientation: np.array):
        return prev_orientation[0:2] + cfg.sampling_period * processed_input[3:5] * np.flip(
            np.cos(prev_orientation[0:2]))

    @staticmethod
    def estimate_vertical_acceleration(processed_input, angles):
        """
        Estimates vertical acceleration of the sensor.
        Rotates the set of acceleration vectors produced by the angles found in angle estimation, then uses the
        z-component of the resulting set as the vertical acceleration of the sensor.
        """
        a_vector = processed_input[0:3]
        inverse_orientation = [-angles[0], -angles[1], 0.0]
        global_a_vector = Rotations.rotate_system(sys_ax=a_vector, sensor_angles=inverse_orientation)

        return -(global_a_vector[2] + cfg.gravitational_constant)

    def estimate_vertical_velocity(self, vertical_acceleration: int):

        vertical_velocity = self.vertical_velocity_tracker.get_tail() + cfg.sampling_period * vertical_acceleration
        # Vertical velocity is adjusted by a damping factor to compensate for integration drift
        vertical_velocity = vertical_velocity * (1 - self.vel_damping.value())

        self.vertical_velocity_tracker.enqueue(vertical_velocity)

        bias = self.update_bias(cfg.vertical_velocity_identifier, self.vertical_velocity_tracker, 1)

        if cfg.ignore_vertical_velocity_bias:
            return vertical_velocity
        else:
            return vertical_velocity - bias

    def update_bias(self, identifier, cyclic_array, number_of_new_values):
        # If a number of rows equal to or greater than the threshold for updating bias has been reached, update bias
        self.bias_estimators[identifier].update_counter(number_of_new_values)

        if self.bias_estimators[identifier].should_update():
            values = cyclic_array.get_latest_n(self.points_for_mean[identifier])
            self.bias_estimators[identifier].update(values)

        self.biases[identifier] = self.bias_estimators[identifier].value()
        return self.biases[identifier]

    def estimate_vertical_position(self, vertical_velocity: int):
        # Vertical position is updated using current vertical velocity
        vertical_position = self.vertical_position_tracker.get_tail() + cfg.sampling_period * vertical_velocity
        # Vertical position is adjusted by a damping factor to compensate for integration drift
        vertical_position = vertical_position * (1 - self.pos_damping.value())

        self.vertical_position_tracker.enqueue(vertical_position)

        bias = self.update_bias(cfg.vertical_position_identifier, self.vertical_position_tracker, 1)

        # Vertical position is adjusted by the bias and stored as output
        if cfg.ignore_vertical_position_bias:
            return vertical_position
        else:
            return vertical_position - bias

    def copy_data_to_last_index_on_buffer_reuse(self):
        # Since some methods use data indexed in [row_no-1], data from [self.last_valid] is copied to
        # [-1]. One could also check for start==0 in each of these methods but this is less expensive.
        self.vertical_velocity[-1] = self.vertical_velocity[self.last_valid]
        self.vertical_position[-1] = self.vertical_position[self.last_valid]
        self.orientation_mmap[-1] = self.orientation_mmap[self.last_valid]
        if self.dev_mode:
            self.dev_gyro_state[-1] = self.dev_gyro_state[self.last_valid]

    def post_process_output(self, current_orientations: np.array):
        if self.dev_mode and not cfg.use_output_filtering:
            return current_orientations

        numb_current_orientations = current_orientations.shape[0]
        if numb_current_orientations < cfg.min_filter_size:
            remaining_values = cfg.min_filter_size - numb_current_orientations
            prev_orientations = self.orientations_tracker.get_latest_n(remaining_values)
            orientations = np.concatenate((prev_orientations, current_orientations))
        else:
            orientations = current_orientations

        return LowPassFilter.process(orientations, self.low_a, self.low_b,
                                     numb_current_orientations)

    def wave_function(self, prev_accelerations: np.array):
        # Check if it is time to perform a Fourier transform
        if self.points_since_wave_update < cfg.n_points_between_fft:
            self.points_since_wave_update += 1
            return

        self.points_since_wave_update = 0
        # First, we do a transform of the vertical acceleration signal
        fourier_transform = fft(prev_accelerations[-cfg.n_points_for_fft:])

        coefficient_array = 2.0 / cfg.n_points_for_fft * np.abs(fourier_transform[0:cfg.n_points_for_fft // 2])
        self.save_wave_function(coefficient_array)

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
            self.points_for_mean[cfg.vertical_position_identifier] = new_n_data_points

            # print(f'When last row was {self.last_row}, pos bias window size set to {self.n_points_for_pos_mean}')

    @staticmethod
    def convert_to_right_handed_coords(processed_input: np.array):
        """
        From testing we know that the sensor output does not match mathematical convention. This method uses information
        provided in config.py to reverse all gyro axes and exactly one of the accelerometer axes.
        :param processed_input: array with the input to correct
        """
        if cfg.perform_axis_reversal:
            # Reverse gyro data
            processed_input[:, 3:5] *= cfg.gyro_reversal_coefficient
            # Reverse accelerometer data. Since positive direction of accelerometer and gyro is linked, the same gyro
            # axis must be reversed again.
            processed_input[:, cfg.axis_reversal_index] *= cfg.axis_reversal_coefficient
            if cfg.axis_reversal_index < 2:
                processed_input[:, 3 + cfg.axis_reversal_index] *= cfg.axis_reversal_coefficient

        return processed_input

    def get_processed_acc_input(self, imu_array: np.array):
        processed_input = np.empty((imu_array.shape[0], 3))
        processed_input[:, 0:2] = (imu_array[:, 0:2] - self.biases[0:2]) * cfg.gravitational_constant
        processed_input[:, 2] = imu_array[:, 2] * cfg.gravitational_constant
        return processed_input

    def get_processed_gyro_input(self, imu_array: np.array):
        """
        Adjusts gyro data using a sensor bias.
        :param imu_array: Array containing the processed imu measurements to use
        :param start: Index that slices the buffer part that is to be adjusted.
        :param end: Index that slices the buffer part that is to be adjusted.
        """
        return imu_array[:, 3:5] - self.biases[3:5]

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
        empty_inputs = np.empty((end - start, 5)) * np.array([0.0, 0.0, -cfg.gravitational_constant, 0.0, 0.0])
        self.processed_input_tracker.enqueue_n(empty_inputs)
        self.vertical_acceleration[start:end] = 0.0
        self.vertical_velocity[start:end] = self.biases[cfg.vertical_velocity_identifier]
        self.vertical_position[start:end] = self.biases[cfg.vertical_position_identifier]

        # Set dev_mode storage
        if self.dev_mode:
            self.processed_input[start:end, 0:5] = np.array([0.0, 0.0, -cfg.gravitational_constant, 0.0, 0.0])
            self.dev_vertical_velocity[start:end] = 0.0
            self.dev_gyro_state[start:end] = 0.0
            self.dev_acc_state[start:end] = 0.0
            self.gyro_bias_array[start:end] = self.biases[0:2]
            self.acc_bias_array[start:end] = np.array([0.0, 0.0, 0.0])
            self.vertical_vel_bias_array[start:end] = self.biases[cfg.vertical_velocity_identifier]
            self.vertical_pos_bias_array[start:end] = self.biases[cfg.vertical_position_identifier]

        # Set output
        self.orientation_mmap[start:end] = np.array([0.0, 0.0, 0.0])

        # If the burst is discarded, one of the measures taken to quickly adjust to incoming data is to increase
        # the damping factors of speed and velocity. These factors are then themselves dampened subsequently.
        self.pos_damping.boost()
        self.vel_damping.boost()

    def handle_buffer_wrapping(self):
        # Information on last actual buffer index is kept
        self.last_valid = self.last_row
        # Previously update_counters_on_buffer_reuse()
        self.copy_data_to_last_index_on_buffer_reuse()
