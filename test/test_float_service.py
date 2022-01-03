import numpy as np
from numpy.random import Generator
from scipy.signal import butter, filtfilt
#
# from src import float_service as fs, float_service_dev_utils as fsdu, globals as g

import src.float_service as fs
import utils.kalman_filter
import utils.rotations

# def test_low_pass_filter_input():
#     # TODO: FIX test. processed_input and input cannot be compared the way it is being done
#     """
#     Should choose a selection of data, benchmark some measures, clean it, and compare benchmarks to new measures.
#     Examples of measures:
#     - Std deviation from some type of mean value
#     - Std dev from a polynomial regression of parts of the data
#     - Std dev from a smoothed line using something else than the Butterfield filter
#         - Decided to use scipy.interpolate.UnivariateSpline
#
#     TODO: Consider using other metrics than standard deviation from a regression line. For instance, another metric
#         could be used to ensure that the processing doesn't smooth the data too much.
#     """
#     print('----clean_data_test()----')
#     path = 'data/bestumkilen_04-23_timestamped_input.hdf5'
#     f_id = '39202020ff06230d'
#     n_rows = 10000
#
#     # In order to use the Butterfield filter, some values should be set up before-hand
#     sampling_rate = 104.0
#     nyquist_freq = sampling_rate / 2
#     cutoff_rate = 0.1
#     low_b, low_a = butter(int(1 / (cutoff_rate * sampling_rate) * nyquist_freq + 0.5),
#                           cutoff_rate,
#                           btype='lowpass',
#                           output='ba')
#
#     #                                 Alternative regression/smoothing/mean methods
#     def _butterfield_filter(y: np.ndarray):
#         interval = 26
#         out = y.copy()
#         ii = 0
#         while (ii+1)*interval <= len(y):
#             out[ii * interval: (ii + 1) * interval] = np.array(filtfilt(low_b,
#                                                                         low_a,
#                                                                         y[ii * interval: (ii + 1) * interval]))
#             ii += 1
#
#         return out
#
#     def _polynomial_regression(x: np.ndarray, y: np.ndarray, deg: int, dof: int = 0):
#         """
#         This method uses numpy.polyfit() to produce polynomials of some degree for individual bursts of the sensor
#         input.
#         It returns an entire column of data after having produced all the burst polynomials.
#         """
#         interval = 26
#         out = y.copy()
#         ii = 0
#         while (ii+1)*interval <= len(x):
#             # np.polyfit() returns the coefficients of an n-degree polynomial
#             try:
#                 z = np.polyfit(x=x[ii*interval: (ii+1)*interval], y=y[ii*interval: (ii+1)*interval], deg=deg)
#                 # np.poly1d() returns a polynomial mapping from these coefficients
#                 polynomial = np.poly1d(z)
#                 # polynomial is used to convert each burst to the resulting values
#                 out[ii * interval: (ii + 1) * interval] = polynomial(x[ii * interval: (ii + 1) * interval])
#             except np.linalg.LinAlgError as lae:
#                 print(f'\npolyfit() failed for dof {dof}, at interval [{ii*interval}, {(ii+1)*interval}].\n'
#                       f'The given interval was not used.')
#                 print(lae.__str__())
#             ii += 1
#         return out
#
#     def _neighbor_mean(y: np.ndarray, padding: int = 1):
#         """
#         Use some number of neighboring data points to predict every data point in the series (except for the same
#         number of beginning and end points).
#         :param y: Time series of data
#         :param padding: Number of data points used on both side of the mid point to calculate its value
#         """
#         out = np.zeros_like(y)
#         for ii in range(padding):
#             out[ii] = y[ii]
#             # Could be optimised to use a lesser numnber of data points to predict edge points, but this isn't
#             # noticeable in the long run
#             out[-ii-1] = y[-ii-1]
#         for ii in range(padding, len(y)-padding):
#             out[ii] = (np.sum(y[ii-padding:ii]) + np.sum(y[ii+1:ii+padding+1]))/(2*padding)
#
#         return out
#     ###########################################################################################
#
#     def _std_of_arrays(y1: np.ndarray, y2: np.ndarray):
#         """
#         Measure the standard deviation between not an array and it's mean, but between two arrays.
#         """
#         differences = y2-y1
#         sqr_diff = np.power(differences, 2)
#         sum_sqr_diff = np.sum(sqr_diff)
#         mean_sum_sqr_diff = sum_sqr_diff/len(y1)
#         return np.sqrt(mean_sum_sqr_diff)
#
#     # Test begins
#
#     # Create process, fill buffers of requested size
#     sis = fsdu.ProcessSimulator(data_path=path, data_rows_total=n_rows, buffer_size=n_rows, dev_mode=True)
#
#     # Focusing on data from a single sensor
#     sensor_id = f_id
#
#     # Process data (including preprocessing)
#     sis.all_bursts_single_sensor(sensor_id=sensor_id)
#
#     # Save raw and processed input
#     input_raw = sis.float_services[sensor_id].input
#     input_cleaned = sis.float_services[sensor_id].processed_input
#
#     # Until timestamps are provided, we use indices as x-values:
#     x_indices = np.arange(0, len(input_raw), step=1, dtype=int)
#
#     # For each of the 6 degrees of freedom, measure std of both raw and cleaned data, compared to their respective
#     # results from the other regression/smoothing/mean methods
#     raw_data_deviation = np.zeros([5, 4], dtype=float)
#     clean_data_deviation = np.zeros_like(raw_data_deviation)
#     for i in range(5):
#         # Apply each method, save information on standard deviations
#         poly_reg_line = _polynomial_regression(x=x_indices, y=input_raw[:, i], deg=3, dof=i)
#         raw_data_deviation[i, 0] = _std_of_arrays(y1=poly_reg_line, y2=input_raw[:, i])
#         # uni_line = _univariate_spline(x=x, y=input_raw[:, i], smoothing_factor=0.003)
#         # raw_data_deviation[i, 1] = _std_of_arrays(y1=uni_line, y2=input_raw[:, i])
#         mean_line_1 = _neighbor_mean(y=input_raw[:, i], padding=1)
#         raw_data_deviation[i, 1] = _std_of_arrays(y1=mean_line_1, y2=input_raw[:, i])
#         mean_line_2 = _neighbor_mean(y=input_raw[:, i], padding=2)
#         raw_data_deviation[i, 2] = _std_of_arrays(y1=mean_line_2, y2=input_raw[:, i])
#         butterfield_line = _butterfield_filter(y=input_raw[:, i])
#         raw_data_deviation[i, 3] = _std_of_arrays(y1=butterfield_line, y2=input_raw[:, i])
#
#         # Do the same thing using cleaned data
#         cleaned_poly_reg_line = _polynomial_regression(x=x_indices, y=input_cleaned[:, i], deg=3, dof=i)
#         clean_data_deviation[i, 0] = _std_of_arrays(y1=cleaned_poly_reg_line, y2=input_cleaned[:, i])
#         # cleaned_uni_line = _univariate_spline(x=x, y=input_cleaned[:, i], smoothing_factor=0.003)
#         # clean_data_deviation[i, 1] = _std_of_arrays(y1=cleaned_uni_line, y2=input_cleaned[:, i])
#         cleaned_mean_line_1 = _neighbor_mean(y=input_cleaned[:, i], padding=1)
#         clean_data_deviation[i, 1] = _std_of_arrays(y1=cleaned_mean_line_1, y2=input_cleaned[:, i])
#         cleaned_mean_line_2 = _neighbor_mean(y=input_cleaned[:, i], padding=2)
#         clean_data_deviation[i, 2] = _std_of_arrays(y1=cleaned_mean_line_2, y2=input_cleaned[:, i])
#         cleaned_butterfield_line = _butterfield_filter(y=input_cleaned[:, i])
#         clean_data_deviation[i, 3] = _std_of_arrays(y1=cleaned_butterfield_line, y2=input_cleaned[:, i])
#
#     data_cleaning_tests = 0
#     dc_tests_passed = 0
#     # Inspect the standard deviation of every metric
#
#     num_2_dof = {0: 'accX', 1: 'accY', 2: 'accZ', 3: 'gyroX', 4: 'gyroY', 5: 'gyroZ'}
#     num_2_func = {0: 'polynomial', 1: 'univariate', 2: 'neighbourAverage1', 3: 'neighborAverage2'}
#
#     err_msg = ''
#     for i in range(np.shape(raw_data_deviation)[0]):
#         for j in range(np.shape(raw_data_deviation)[1]):
#             data_cleaning_tests += 1
#             if raw_data_deviation[i, j] <= clean_data_deviation[i, j]:
#                 err_msg += f'{num_2_dof[i]}: {num_2_func[j]}, '
#             else:
#                 dc_tests_passed += 1
#
#     print(err_msg)
#     print('----clean_data_test() ended----\n')
#     assert dc_tests_passed == data_cleaning_tests
#     #     self.fail_messages.append(f'{self.tc[2]}FAILED{self.tc[0]}\n' +
#     #                               err_msg + f'\n{dc_tests_passed}/{data_cleaning_tests} passed.\n'
#     #                                                       f'If some of the univariate tests fail, there might still'
#     #                                                       f' be no problem at all.\n'
#     #                                                       f'NB test is broken as of 12. jan 2021.\n')
#     # else:
#     #     self.n_passed += 1
#     #     self.passed_messages.append(f'{self.tc[1]}PASSED{self.tc[0]}\n'
#     #                                 'clean_data_test()\n'
#     #                                 f'{dc_tests_passed}/{data_cleaning_tests} tests passed.\n')



def test_nan_handling():
    print('----nan_smoothing_test()----')
    err_msg = ''
    n_tests = 0
    n_passed = 0

    data_size = 1000
    name = 'Copernicus'

    # 1. Testing for whether or not the burst is discarded
    # 1.1 All data is NaN
    mock_input = np.zeros([data_size, 6], dtype=float)
    output = np.zeros([data_size, 3], dtype=float)
    float_service = fs.FloatService(name=name, input=mock_input, output=output)
    mock_input[:] = float('nan')
    float_service.nan_handling(start=0, end=data_size)

    n_tests += 1
    if not float_service.burst_is_discarded:
        err_msg += f'- 1.1 Burst not properly discarded\n'
    else:
        n_passed += 1

    # 1.2 Slightly more data than threshold is NaN
    mock_input = np.zeros([data_size, 6], dtype=float)
    output = np.zeros([data_size, 3], dtype=float)
    float_service = fs.FloatService(name=name, input=mock_input, output=output)
    n_nan = int(float_service.discard_burst_nan_threshold * data_size + 1)
    mock_input[0:n_nan, :] = float('nan')
    float_service.nan_handling(start=0, end=data_size)

    n_tests += 1
    if not float_service.burst_is_discarded:
        err_msg += f'- 1.2 Burst not properly discarded\n'
    else:
        n_passed += 1

    # 2. The correct number of NaN-values in a burst
    n_tests += 1
    if not float_service.nan_in_burst == n_nan:
        err_msg += f'- 2.  Wrong number of NaN values found\n' \
                   f'      Found = {float_service.nan_in_burst}, correct = {n_nan}\n'
    else:
        n_passed += 1

    # 3. Clean dataset
    mock_input = np.zeros([data_size, 6], dtype=float)
    output = np.zeros([data_size, 3], dtype=float)
    float_service = fs.FloatService(name=name, input=mock_input, output=output)
    float_service.nan_handling(start=0, end=data_size)

    n_tests += 1
    if not float_service.nan_in_burst == 0:
        err_msg += f'- 2.  Wrong number of NaN values found\n' \
                   f'      Found = {float_service.nan_in_burst}, correct = {n_nan}\n'
    else:
        n_passed += 1

    print(err_msg)
    print('----nan_handling_test() ended----\n')
    assert n_passed == n_tests


def test_set_processed_data_nan():
    """
    This test sets up a mock input data set with some values being NaN values.
    It checks whether the NaN values have been converted to non-NaN values in output and processed input.
    """
    print('----nan_smoothing_test()----')
    err_msg = ''
    n_tests = 0
    n_passed = 0

    data_size = 1000
    name = 'Copernicus'
    mock_input = np.zeros([data_size, 6], dtype=float)
    output = np.zeros([data_size, 3], dtype=float)

    # Every n-th entry in the input is set to NaN
    nan_step = 3
    for i in range(0, data_size, nan_step):
        mock_input[i, :] = float('nan')

    mock_input[-1, :] = float('nan')

    # Every 3rd entry from index 1 is set to 1.0
    for i in range(1, data_size, nan_step):
        mock_input[i, :] = 1.0

    float_service = fs.FloatService(name=name, input=mock_input, output=output)

    # Apply NaN smoothing
    for i in range(6):
        float_service.set_processed_acc_input_nan(start=0, end=data_size)
        float_service.set_processed_gyro_input_nan(start=0, end=data_size)

    # Check that input is not overwritten
    n_tests += 1
    passed = True
    for i in range(0, data_size, nan_step):
        if not np.all(np.isnan(mock_input[i])):
            err_msg += f'- 1.1 Input overwritten\n'
            passed = False
            break
    if passed:
        n_passed += 1

    # Check that output is not NaN
    n_tests += 1
    if np.any(np.isnan(output)):
        err_msg += f'- 1.2 NaN in output\n'
    else:
        n_passed += 1

    # Check that processed input is not NaN
    n_tests += 1
    if np.any(np.isnan(float_service.processed_input)):
        err_msg += f'- 1.3 NaN in processed input'
    else:
        n_passed += 1

    print(err_msg)
    print('----nan_smoothing_test() ended----\n')
    assert n_passed == n_tests


def test_discard_burst():
    print('----discard_burst_test()----')
    data_size = 1000
    burst_size = 100
    name = 'Copernicus'
    err_msg = ''
    n_tests = 0
    n_passed = 0

    # 1. All data is NaN
    mock_input = np.zeros(shape=(data_size, 6), dtype=float)
    mock_input[:] = float('nan')
    output = np.zeros(shape=(data_size, 3), dtype=float)
    float_service = fs.FloatService(name=name, input=mock_input, output=output)
    fs.n_rows = data_size
    for i in range(data_size//burst_size):
        float_service.process(number_of_rows=burst_size)

    # Check that input is not overwritten
    n_tests += 1
    if not np.all(np.isnan(mock_input)):
        err_msg += f'- 1.1 Input overwritten\n'
    else:
        n_passed += 1

    # Check that output is not NaN
    n_tests += 1
    if np.any(np.isnan(output)):
        err_msg += f'- 1.2 NaN in output\n'
    else:
        n_passed += 1

    # Check that processed input is not NaN
    n_tests += 1
    if np.any(np.isnan(float_service.processed_input)):
        err_msg += f'- 1.3 NaN in processed input'
    else:
        n_passed += 1

    print(err_msg)
    print('----discard_burst_test() ended----\n')
    assert n_passed == n_tests


def test_adjust_pos_and_vel_damping_factors():
    print('----adjust_pos_and_vel_damping_factors_test()----')
    data_size = 100
    mock_input = np.zeros(shape=(data_size, 6), dtype=float)
    output = np.zeros(shape=(data_size, 3), dtype=float)
    name = 'Copernicus'
    float_service = fs.FloatService(name=name, input=mock_input, output=output, dev_mode=True)

    # Set damping factors to max
    float_service.vel_damping_factor = float_service.vel_damping_factor_big
    float_service.pos_damping_factor = float_service.pos_damping_factor_big

    vdf = float_service.vel_damping_factor
    pdf = float_service.pos_damping_factor
    for i in range(data_size-1):
        float_service.process(number_of_rows=1)
        assert vdf >= float_service.vel_damping_factor
        assert pdf >= float_service.pos_damping_factor
        vdf = float_service.vel_damping_factor
        pdf = float_service.pos_damping_factor

    print('----adjust_pos_and_vel_damping_factors_test() ended----\n')


def test_boost_dampeners():
    print('----boost_dampeners_test()----')
    data_size = 100
    mock_input = np.zeros(shape=(data_size, 6), dtype=float)
    output = np.zeros(shape=(data_size, 3), dtype=float)
    name = 'Copernicus'
    err_msg = ''
    passed = True

    # 1. Setting damping factors to custom value
    float_service = fs.FloatService(name=name, input=mock_input, output=output)
    vdf = float_service.vel_damping_factor
    pdf = float_service.pos_damping_factor
    float_service.boost_dampeners(vel_damping_factor=2*vdf, pos_damping_factor=2*pdf)
    if vdf > float_service.vel_damping_factor\
            or pdf > float_service.pos_damping_factor:
        passed = False
        err_msg += '\n- 1. Custom value damping factor'

    # 2. Setting damping factors to default value
    float_service = fs.FloatService(name=name, input=mock_input, output=output)
    float_service.boost_dampeners()
    if not(float_service.vel_damping_factor_big == float_service.vel_damping_factor and
           float_service.pos_damping_factor_big == float_service.pos_damping_factor):
        passed = False
        err_msg += '\n- 2. Default value damping factor'

    print(err_msg)
    print('----boost_dampeners_test() ended----\n')
    assert passed


def test_set_position_average_weights():
    print('----set_position_average_weights_test()----')
    data_size = 100
    sensor_input = np.zeros(shape=(data_size, 6))
    sensor_output = np.zeros(shape=(data_size, 3))
    name = 'copernicus'
    float_service = fs.FloatService(name=name, input=sensor_input, output=sensor_output, dev_mode=True)
    pos_mean_window_len = float_service.n_points_for_pos_mean

    assert pos_mean_window_len == len(float_service.get_position_averaging_weights())

    new_pos_mean_window_len = 350
    float_service.n_points_for_pos_mean = new_pos_mean_window_len
    weights = float_service.get_position_averaging_weights()
    assert new_pos_mean_window_len == len(weights)

    print('----set_position_average_weights_test() ended----\n')


def test_kalman_project_state():
    """
    This test is written with prototype 2 in mind, where the y-axis is flipped, as well as all gyro values.
    """
    print('----kalman_project_state_test()----')
    kalman_project_state_tests = 0
    kalman_project_state_tests_passed = 0
    err_msg = ''

    data_size = 100
    fs.n_rows = data_size

    # 1 - Still, horizontal sensor
    kalman_project_state_tests += 1
    sensor_name = 'copernicus_01'
    input_buffers = np.zeros(shape=(fs.n_rows, 6), dtype=float)
    output_buffers = np.zeros(shape=(fs.n_rows, 3), dtype=float)

    float_service = fs.FloatService(name=sensor_name, input=input_buffers, output=output_buffers, dev_mode=True)
    # Turn off gyroscope mean adjustment
    float_service.points_between_gyro_bias_update = np.inf

    # Mock input manipulation
    float_service.imu_mmap[:, 2] = 1.0
    float_service.process(number_of_rows=data_size)
    passed = True
    for i in range(len(float_service.orientation_mmap)):
        # This test expects the acceleration induced angles to be 0.0
        if float_service.dev_gyro_state[i, 0] != 0.0 \
                or float_service.dev_gyro_state[i, 1] != 0.0:
            err_msg += '- 1. Still, horizontal sensor\n'
            passed = False
            break
    if passed:
        kalman_project_state_tests_passed += 1

    # 2 - Constant angular velocity, single axis
    kalman_project_state_tests += 1
    sensor_name = 'copernicus_02'
    input_buffers = np.zeros(shape=(fs.n_rows, 6), dtype=float)
    output_buffers = np.zeros(shape=(fs.n_rows, 3), dtype=float)
    float_service = fs.FloatService(name=sensor_name, input=input_buffers, output=output_buffers, dev_mode=True)
    # Turn off gyroscope mean adjustment
    float_service.points_between_gyro_bias_update = np.inf

    # Mock input manipulation
    input_burst = np.zeros(shape=(data_size, 6), dtype=float)
    # Constant angular velocity of 1 deg/s around the x-axis
    # Gyro data flipped
    input_burst[:, 3] = - 1.0
    float_service.points_between_gyro_bias_update = np.inf
    passed = True
    for i in range(len(float_service.orientation_mmap) - 1):
        # Check that the next gyro induced state has a greater x-angle than the last
        if float_service.dev_gyro_state[i+1, 0] < \
                float_service.dev_gyro_state[i, 0]:
            err_msg += '- 2. Constant angular velocity, single axis\n'
            passed = False
            break
    if passed:
        kalman_project_state_tests_passed += 1

    print(err_msg)
    print('----kalman_project_state_test() ended----\n')
    assert kalman_project_state_tests_passed == kalman_project_state_tests


def test_kalman_z():
    """
    This test is written with prototype 2 in mind, where the y-axis is flipped, as well as all gyro values.
    """
    print('----kalman_z_test()----')
    kalman_z_tests = 0
    kalman_z_passed = 0
    err_msg = ''

    data_size = 100

    # These timestamps are used for nothing except in producing mock input,
    # and does not influence sampling rate etc
    timestamps = np.linspace(0.0, 10, data_size)

    # 1 - Still, horizontal sensor
    kalman_z_tests += 1
    sensor_name = 'copernicus_01'
    fs.n_rows = data_size
    input_buffers = np.zeros(shape=(fs.n_rows, 6), dtype=float)
    output_buffers = np.zeros(shape=(fs.n_rows, 3), dtype=float)
    float_service = fs.FloatService(name=sensor_name, input=input_buffers, output=output_buffers, dev_mode=True)
    # Turn off acceleration mean adjustment
    float_service.points_between_acc_bias_update = np.inf

    # Mock input manipulation
    float_service.imu_mmap[:, 2] = 1.0
    float_service.process(number_of_rows=data_size)
    passed = True
    for i in range(len(float_service.orientation_mmap)):
        # This test expects the acceleration induced angles to be 0.0
        if float_service.dev_acc_state[i, 0] != 0.0 \
                or float_service.dev_acc_state[i, 1] != 0.0:
            err_msg += '- 1. Still, horizontal sensor\n'
            passed = False
            break
    if passed:
        kalman_z_passed += 1

    # 2 - Moving sensor
    kalman_z_tests += 1
    sensor_name = 'copernicus_02'
    fs.n_rows = data_size
    input_buffers = np.zeros(shape=(fs.n_rows, 6), dtype=float)
    output_buffers = np.zeros(shape=(fs.n_rows, 3), dtype=float)
    fs.n_rows = data_size
    float_service = fs.FloatService(name=sensor_name, input=input_buffers, output=output_buffers, dev_mode=True)
    # Turn off acceleration mean adjustment
    float_service.points_between_acc_bias_update = np.inf
    # Make sure the Kalman filter is activated at every data row
    float_service.rows_per_kalman_use = 1

    # Mock input manipulation
    float_service.imu_mmap[:, 0] = np.sin(timestamps)
    # Y-axis flipped
    float_service.imu_mmap[:, 1] = - np.cos(timestamps)
    float_service.process(number_of_rows=data_size)
    passed = True
    for i in range(len(float_service.orientation_mmap)):
        # This test expects the absolute value of the acceleration tests to
        if abs(float_service.imu_mmap[i, 0]) > abs(float_service.imu_mmap[i, 1]) and \
                abs(float_service.dev_acc_state[i, 0]) > \
                abs(float_service.dev_acc_state[i, 1]):
            err_msg += '- 2.1 Moving sensor, absolute value\n'
            passed = False
            print(i)
            break
    if passed:
        kalman_z_passed += 1

    print(err_msg)
    print('----kalman_z_test() ended----\n')
    assert kalman_z_passed == kalman_z_tests


def test_get_corrected_angle():
    pi = np.pi
    input_angles = [0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi, pi, 1.25 * pi, 1.5 * pi, 1.75 * pi, 2.0 * pi, 2.25 * pi,
                    3.25 * pi,
                    -0.25 * pi, -0.5 * pi, -0.75 * pi, -pi, -1.25 * pi, -1.5 * pi, -1.75 * pi, -2.0 * pi,
                    -2.25 * pi, -3.25 * pi]
    output_angles = [0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi, pi, -0.75 * pi, -0.5 * pi, -0.25 * pi, 0.0, 0.25 * pi,
                     -0.75 * pi,
                     -0.25 * pi, -0.5 * pi, -0.75 * pi, -pi, 0.75 * pi, 0.5 * pi, 0.25 * pi, 0.0, -0.25 * pi,
                     0.75 * pi]

    for i in range(len(input_angles)):
        assert abs(
            utils.kalman_filter.KalmanFilter.get_corrected_angle(input_angles[i]) - output_angles[i]) < 0.00001


def test_rotate_system():

    """
    System is represented as three unit vectors emerging from origo, denoting the three axes of the system:
      [x, y, z]
     =
    [[x_X, y_X, z_X],
     [x_Y, y_Y, z_Y],
     [x_Z, y_Z, z_Z]]

    Where x, y, z are 3x1 the axes of the inner system, represented by coordinates from the outer system, X, Y, Z,
    Such that x_Y is the outer Y-coordinate of the inner x-axis.

    Since the rotate_system() method is the only method that will ever access the rotation_matrix() method, this is
    probably actually more of a test to see whether the rotation matrix correctly rotates the axes of the system.
    Tests that may be used:
    - No rotation: Simply testing that the system is conserved if there is no rotation.
    - Elemental rotations: Rotations along single axes to see if the vectors of the affected axes end up correctly
    placed.
    - Linked rotations: Rotate a system around one axis, and then another one, to ensure the matrix rotates the
    system intrinsically (along the axes of the inner system, or sensor) and not extrinsically (along the axes of
    the outer system).
    - Small multi-axis rotations: Rotate a system using angles from more than one axis. Do small, realistic steps
    and verify that the system rotates correctly. Look at angles between original and new axes, for instance. This
    could be expanded to ensure that the same minor rotation done twice provides the same result.
    # TODO: Write expansion of small multi-axis rotation tests
    - Angles between axes: Since the entire system is to be rotated as one, rigid body, the orthogonality of the
    three axes must be kept true. Tiny anomalies are acceptable due to float point accuracy.
    - Linked rotation error: The error arising from many linked rotations should be measured in order to know how
    often to adjust for this by for instance checking for axis orthogonality
    """

    #                                    Rotation specific methods

    def _check_for_ndarray_element_similarity(m1: np.ndarray, m2: np.ndarray, threshold: float = 0.00001):
        """
        Helper method for rotate_system_test()
        """
        if np.shape(m1) != np.shape(m2):
            return False
        for i in range(np.shape(m1)[0]):
            for j in range(np.shape(m2)[1]):
                if abs(m1[i, j] - m2[i, j]) > threshold:
                    return False

        return True

    def _check_system_coordinate_movement_directions(m1: np.ndarray, m2: np.ndarray, diffs: list,
                                                     threshold: float = 0.00001):
        """
        Helper method for rotate_system_test().

        :param m1: The system before a minor rotation is applied.
        :param m2: The resulting system after the rotation.
        :param diffs: A list of assumptions regarding the relationship between pairs of coordinates. If a diffs-element
        equals -1, check that the resulting coordinate is of a smaller  value than the original one. If the
        diffs-element equals 1, check the opposite. If 0 (which I doubt will be tested for in reality with rotation
        around more than one axis) check that the two coordinates are within the threshold of each other. NB: This list
        is written row-wise from the two ndarrays.
        :param threshold: Only used for cases where two coordinates are assumed equal to each other.
        """
        for i in range(np.shape(m1)[0]):
            for j in range(np.shape(m1)[1]):
                if diffs[i * np.shape(m1)[0] + j] == -1:
                    if m2[i, j] > m1[i, j]:
                        print(f'Wrong assumption on negative movement in {i, j}')
                        return False
                if diffs[i * np.shape(m1)[0] + j] == 1:
                    if m2[i, j] < m1[i, j]:
                        print(f'Wrong assumption on positive movement in {i, j}')
                        return False
                if diffs[i * np.shape(m1)[0] + j] == 0:
                    if abs(m2[i, j] - m1[i, j]) > threshold:
                        return False

        return True

    def _angle(v1, v2):
        return np.arccos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
                          / (np.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) *
                             np.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2))))

    def _check_orthogonal_axes(system: np.ndarray, threshold: float = 0.0001):
        """
        Helper method for rotate_system_test().
        Control whether the angles between the axes are within some threshold of 90 degrees.
        """

        v1 = system[:, 0]
        v2 = system[:, 1]
        v3 = system[:, 2]

        if abs(_angle(v1, v2) - 0.5 * np.pi) > threshold:
            return False
        if abs(_angle(v1, v3) - 0.5 * np.pi) > threshold:
            return False
        if abs(_angle(v2, v3) - 0.5 * np.pi) > threshold:
            return False
        return True

    def _stats_from_skewed_system(sys_1: np.ndarray, sys_2: np.ndarray):
        """
        Used for observing the error that stems from multiple roundtrip rotations.
        """

        # Measure individual coordinate drift of each axis
        max_coordinate_drift = np.max(np.abs(sys_2.flatten() - sys_1.flatten()))
        mean_coordinate_drift = np.mean(np.abs(sys_2.flatten() - sys_1.flatten()))

        # Measure angles between original axes and resulting axes
        angles = np.zeros([3], dtype=float)
        for i in 0, 1, 2:
            angles[i] = _angle(sys_1[:, i], sys_2[:, i])
        max_angular_drift = np.max(angles)
        mean_angular_drift = np.mean(angles)

        # Measure the orthogonality in the resulting system

        angles[0] = _angle(sys_2[:, 0], sys_2[:, 1])
        angles[1] = _angle(sys_2[:, 1], sys_2[:, 2])
        angles[2] = _angle(sys_2[:, 0], sys_2[:, 2])

        mean_orthogonality = np.mean(angles) - 0.5*np.pi

        return max_coordinate_drift, mean_coordinate_drift, max_angular_drift, mean_angular_drift, mean_orthogonality

    #######################################################################

    print('_______ rotate_system_test() started _______\n')
    error_message = 'FAILED\n' \
                    'rotate_system_test()\n' \
                    'Tests failed:'
    rotation_tests = 0
    passed_rotation_tests = 0

    float_service = utils.rotations.Rotations()

    #                                          Testing no rotation

    def _no_rotation_tests():
        er_msg = ''
        r_tests = 0
        r_passed = 0

        system = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        # anyways
        angles = np.array([0.0, 0.0, 0.0])

        test_sys = float_service.rotate_system(sys_ax=system,
                                               sensor_angles=angles)

        # 0.1
        r_tests += 1
        if not _check_for_ndarray_element_similarity(system, test_sys):
            er_msg += ' 0.1,'
        else:
            r_passed += 1

        # 0.2
        r_tests += 1
        if not _check_for_ndarray_element_similarity(system, test_sys):
            er_msg += ' 0.2,'
        else:
            r_passed += 1

        return r_tests, r_passed, er_msg

    t, p, m = _no_rotation_tests()
    rotation_tests += t
    passed_rotation_tests += p
    error_message += m

    ###########################################################

    #                                    Elemental rotation tests

    def _elemental_rotation_tests():
        er_msg = ''
        r_tests = 0
        r_passed = 0

        system = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        # 90 degrees around x-axis
        el_rot_1 = np.array([0.5 * np.pi, 0, 0])
        el_rot_2 = np.array([-0.5 * np.pi, 0, 0])
        # 90 degrees around y-axis
        el_rot_3 = np.array([0, 0.5 * np.pi, 0])
        el_rot_4 = np.array([0, -0.5 * np.pi, 0])
        # 90 degrees around z-axis
        el_rot_5 = np.array([0, 0, 0.5 * np.pi])
        el_rot_6 = np.array([0, 0, -0.5 * np.pi])

        el_rot_sys_1 = float_service.rotate_system(system, el_rot_1)
        el_rot_sys_2 = float_service.rotate_system(system, el_rot_2)
        el_rot_sys_3 = float_service.rotate_system(system, el_rot_3)
        el_rot_sys_4 = float_service.rotate_system(system, el_rot_4)
        el_rot_sys_5 = float_service.rotate_system(system, el_rot_5)
        el_rot_sys_6 = float_service.rotate_system(system, el_rot_6)

        el_rot_sys_1_true = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0],
                                      [0.0, 1.0, 0.0]])

        el_rot_sys_2_true = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0],
                                      [0.0, -1.0, 0.0]])

        el_rot_sys_3_true = np.array([[0.0, 0.0, 1.0],
                                      [0.0, 1.0, 0.0],
                                      [-1.0, 0.0, 0.0]])

        el_rot_sys_4_true = np.array([[0.0, 0.0, -1.0],
                                      [0.0, 1.0, 0.0],
                                      [1.0, 0.0, 0.0]])

        el_rot_sys_5_true = np.array([[0.0, -1.0, 0.0],
                                      [1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])

        el_rot_sys_6_true = np.array([[0.0, 1.0, 0.0],
                                      [-1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])

        # 1.1
        r_tests += 1
        if not _check_for_ndarray_element_similarity(el_rot_sys_1, el_rot_sys_1_true):
            er_msg += ' 1.1,'
        else:
            r_passed += 1
        # 1.2
        r_tests += 1
        if not _check_for_ndarray_element_similarity(el_rot_sys_2, el_rot_sys_2_true):
            er_msg += ' 1.2,'
        else:
            r_passed += 1
        # 1.3
        r_tests += 1
        if not _check_for_ndarray_element_similarity(el_rot_sys_3, el_rot_sys_3_true):
            er_msg += ' 1.3,'
        else:
            r_passed += 1
        # 1.4
        r_tests += 1
        if not _check_for_ndarray_element_similarity(el_rot_sys_4, el_rot_sys_4_true):
            er_msg += ' 1.4,'
        else:
            r_passed += 1
        # 1.5
        r_tests += 1
        if not _check_for_ndarray_element_similarity(el_rot_sys_5, el_rot_sys_5_true):
            er_msg += ' 1.5,'
        else:
            r_passed += 1
        # 1.6
        r_tests += 1
        if not _check_for_ndarray_element_similarity(el_rot_sys_6, el_rot_sys_6_true):
            er_msg += ' 1.6,'
        else:
            r_passed += 1

        return r_tests, r_passed, er_msg

    t, p, m = _elemental_rotation_tests()
    rotation_tests += t
    passed_rotation_tests += p
    error_message += m

    ###########################################################

    #                                              Linked rotation tests

    def _linked_rotation_tests():
        er_msg = ''
        r_tests = 0
        r_passed = 0

        system = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        # 90 degrees around x-axis
        el_rot_1 = np.array([0.5 * np.pi, 0, 0])
        el_rot_2 = np.array([-0.5 * np.pi, 0, 0])
        # 90 degrees around y-axis
        el_rot_3 = np.array([0, 0.5 * np.pi, 0])
        el_rot_4 = np.array([0, -0.5 * np.pi, 0])
        # 90 degrees around z-axis
        el_rot_5 = np.array([0, 0, 0.5 * np.pi])
        el_rot_6 = np.array([0, 0, -0.5 * np.pi])

        # Sys 1: 90 deg x-axis, 90 deg y-axis
        linked_rot_sys_1 = float_service.rotate_system(sys_ax=system, sensor_angles=el_rot_1)
        linked_rot_sys_1 = float_service.rotate_system(sys_ax=linked_rot_sys_1, sensor_angles=el_rot_3)

        # Sys 2: 90 deg z-axis, -90 deg y-axis
        linked_rot_sys_2 = float_service.rotate_system(sys_ax=system, sensor_angles=el_rot_5)
        linked_rot_sys_2 = float_service.rotate_system(sys_ax=linked_rot_sys_2, sensor_angles=el_rot_4)

        # Sys 3: 180 deg y-axis, -90 deg z-axis, 90 deg x-axis
        linked_rot_sys_3 = float_service.rotate_system(sys_ax=system, sensor_angles=el_rot_3)
        linked_rot_sys_3 = float_service.rotate_system(sys_ax=linked_rot_sys_3, sensor_angles=el_rot_3)
        linked_rot_sys_3 = float_service.rotate_system(sys_ax=linked_rot_sys_3, sensor_angles=el_rot_6)
        linked_rot_sys_3 = float_service.rotate_system(sys_ax=linked_rot_sys_3, sensor_angles=el_rot_1)

        linked_rot_sys_1_true = np.array([[0.0, 0.0, 1.0],
                                          [1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0]])

        linked_rot_sys_2_true = np.array([[0.0, -1.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [1.0, 0.0, 0.0]])

        linked_rot_sys_3_true = np.array([[0.0, 0.0, 1.0],
                                          [-1.0, 0.0, 0.0],
                                          [0.0, -1.0, 0.0]])

        # 2.1
        r_tests += 1
        if not _check_for_ndarray_element_similarity(linked_rot_sys_1, linked_rot_sys_1_true):
            er_msg += ' 2.1,'
        else:
            r_passed += 1
        # 2.2
        r_tests += 1
        if not _check_for_ndarray_element_similarity(linked_rot_sys_2, linked_rot_sys_2_true):
            er_msg += ' 2.2'
        else:
            r_passed += 1
        # 2.3
        r_tests += 1
        if not _check_for_ndarray_element_similarity(linked_rot_sys_3, linked_rot_sys_3_true):
            er_msg += ' 2.3,'
        else:
            r_passed += 1

        return r_tests, r_passed, er_msg

    t, p, m = _linked_rotation_tests()
    rotation_tests += t
    passed_rotation_tests += p
    error_message += m

    ###########################################################

    #                                         Multi-axis rotation tests

    def _multi_axis_rotation_tests():
        """
        This test cannot be validated on the same terms as the other ones, by using check_for_ndarray_similarity().
        This is because the other ones test for precise answers, while the only method we have to calculate the
        result of this test(as of now) is the method being tested.
        We can however look for signs as to whether it performs the way we want it to.
        """
        er_msg = ''
        r_tests = 0
        r_passed = 0

        system = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        systemx = float_service.rotate_system(sys_ax=system, sensor_angles=np.array([0.5*np.pi, 0.0, 0.0]))
        systemy = float_service.rotate_system(sys_ax=system, sensor_angles=np.array([0.0, 0.5*np.pi, 0.0]))
        systemz = float_service.rotate_system(sys_ax=system, sensor_angles=np.array([0.0, 0.0, 0.5*np.pi]))

        # Small rotations around x- and y-axis
        rotation_xy = np.array([0.01, 0.01, 0.0])
        system_xy = float_service.rotate_system(sys_ax=system, sensor_angles=rotation_xy)
        systemx_xy = float_service.rotate_system(sys_ax=systemx, sensor_angles=rotation_xy)
        systemy_xy = float_service.rotate_system(sys_ax=systemy, sensor_angles=rotation_xy)
        systemz_xy = float_service.rotate_system(sys_ax=systemz, sensor_angles=rotation_xy)
        # Small rotations around x- and z-axis
        rotation_xz = np.array([0.01, 0.0, 0.01])
        system_xz = float_service.rotate_system(sys_ax=system, sensor_angles=rotation_xz)
        systemx_xz = float_service.rotate_system(sys_ax=systemx, sensor_angles=rotation_xz)
        systemy_xz = float_service.rotate_system(sys_ax=systemy, sensor_angles=rotation_xz)
        systemz_xz = float_service.rotate_system(sys_ax=systemz, sensor_angles=rotation_xz)
        # Small rotations around y- and z-axis
        rotation_yz = np.array([0.0, 0.01, 0.01])
        system_yz = float_service.rotate_system(sys_ax=system, sensor_angles=rotation_yz)
        systemx_yz = float_service.rotate_system(sys_ax=systemx, sensor_angles=rotation_yz)
        systemy_yz = float_service.rotate_system(sys_ax=systemy, sensor_angles=rotation_yz)
        systemz_yz = float_service.rotate_system(sys_ax=systemz, sensor_angles=rotation_yz)

        system_xy_true_diff = [-1, 1, 1,
                               1, -1, -1,
                               -1, 1, -1]
        systemx_xy_true_diff = [-1, 1, 1,
                                1, -1, 1,
                                1, -1, -1]
        systemy_xy_true_diff = [-1, 1, -1,
                                1, -1, -1,
                                1, -1, -1]
        systemz_xy_true_diff = [-1, 1, 1,
                                -1, 1, 1,
                                -1, 1, -1]
        system_xz_true_diff = [-1, -1, 1,
                               1, -1, -1,
                               1, 1, -1]
        systemx_xz_true_diff = [-1, -1, 1,
                                -1, -1, 1,
                                1, -1, -1]
        systemy_xz_true_diff = [1, 1, -1,
                                1, -1, -1,
                                1, 1, -1]
        systemz_xz_true_diff = [-1, 1, 1,
                                -1, -1, 1,
                                1, 1, -1]
        system_yz_true_diff = [-1, -1, 1,
                               1, -1, 1,
                               -1, 1, -1]
        systemx_yz_true_diff = [-1, -1, 1,
                                1, -1, 1,
                                1, -1, 1]
        systemy_yz_true_diff = [-1, 1, -1,
                                1, -1, 1,
                                1, 1, -1]
        systemz_yz_true_diff = [-1, 1, -1,
                                -1, -1, 1,
                                -1, 1, -1]

        # 3.1
        r_tests += 1
        if not _check_system_coordinate_movement_directions(system, system_xy, system_xy_true_diff):
            er_msg += ' 3.1,'
        else:
            r_passed += 1

        # 3.2
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemx, systemx_xy, systemx_xy_true_diff):
            er_msg += ' 3.2,'
        else:
            r_passed += 1

        # 3.3
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemy, systemy_xy, systemy_xy_true_diff):
            er_msg += ' 3.3,'
        else:
            r_passed += 1

        # 3.4
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemz, systemz_xy, systemz_xy_true_diff):
            er_msg += ' 3.4,'
        else:
            r_passed += 1

        # 3.5
        r_tests += 1
        if not _check_system_coordinate_movement_directions(system, system_xz, system_xz_true_diff):
            er_msg += ' 3.5,'
        else:
            r_passed += 1

        # 3.6
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemx, systemx_xz, systemx_xz_true_diff):
            er_msg += ' 3.6,'
        else:
            r_passed += 1

        # 3.7
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemy, systemy_xz, systemy_xz_true_diff):
            er_msg += ' 3.7,'
        else:
            r_passed += 1

        # 3.8
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemz, systemz_xz, systemz_xz_true_diff):
            er_msg += ' 3.8,'
        else:
            r_passed += 1

        # 3.9
        r_tests += 1
        if not _check_system_coordinate_movement_directions(system, system_yz, system_yz_true_diff):
            er_msg += ' 3.9,'
        else:
            r_passed += 1

        # 3.10
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemx, systemx_yz, systemx_yz_true_diff):
            er_msg += ' 3.10,'
        else:
            r_passed += 1

        # 3.11
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemy, systemy_yz, systemy_yz_true_diff):
            er_msg += ' 3.11,'
        else:
            r_passed += 1

        # 3.12
        r_tests += 1
        if not _check_system_coordinate_movement_directions(systemz, systemz_yz, systemz_yz_true_diff):
            er_msg += ' 3.12,'
        else:
            r_passed += 1

        return r_tests, r_passed, er_msg

    t, p, m = _multi_axis_rotation_tests()
    rotation_tests += t
    passed_rotation_tests += p
    error_message += m

    ###########################################################

    #                                              Orthogonality test

    def _orthogonality_tests():
        er_msg = ''
        r_tests = 0
        r_passed = 0

        system = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        # Predefined rotations
        rotation_1 = np.array([0.5, 0.5, 0.5])
        rotation_2 = np.array([0.0, 0.5*np.pi, 0.0])
        rotation_3 = np.array([-0.321, -0.654, 1.11])

        # Random rotations
        rand_rot_1 = np.random.rand(3)*0.5*np.pi
        rand_rot_2 = np.random.rand(3)*np.pi
        rand_rot_3 = np.random.rand(3)*2*np.pi

        # Resulting orientations
        system_1 = float_service.rotate_system(sys_ax=system, sensor_angles=rotation_1)
        system_2 = float_service.rotate_system(sys_ax=system, sensor_angles=rotation_2)
        system_3 = float_service.rotate_system(sys_ax=system, sensor_angles=rotation_3)
        system_4 = float_service.rotate_system(sys_ax=system, sensor_angles=rand_rot_1)
        system_5 = float_service.rotate_system(sys_ax=system, sensor_angles=rand_rot_2)
        system_6 = float_service.rotate_system(sys_ax=system, sensor_angles=rand_rot_3)

        for i, s in enumerate([system_1, system_2, system_3, system_4, system_5, system_6]):
            r_tests += 1
            if not _check_orthogonal_axes(s):
                er_msg += f' 4.{i+1},'
            else:
                r_passed += 1

        return r_tests, r_passed, er_msg

    t, p, m = _orthogonality_tests()
    rotation_tests += t
    passed_rotation_tests += p
    error_message += m

    ###########################################################

    #                                         Multiple linked rotations test

    def _multiple_linked_rotations_error_tests():
        """
        If the system is rotated around each of its x-, y- and z-axes twice, it ends up where it started.
        This property is used to link multiple rotations together, and to measure the error of the resulting system.
        Thus this test isn't a pass/fail test, but an examination.
        """
        message = 'Results from measuring error in multiple linked rotations():\n\n'

        system = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])

        x_rot = np.array([0.5*np.pi, 0.0, 0.0])
        y_rot = np.array([0.0, 0.5*np.pi, 0.0])
        z_rot = np.array([0.0, 0.0, 0.5*np.pi])

        # rotation_roundtrips = [1, 10, 100, 1000, 10_000, 100_000, 1000_000]
        rotation_roundtrips = [10_000]
        systems = np.zeros(shape=(len(rotation_roundtrips), 3, 3))

        for i, r in enumerate(rotation_roundtrips):
            print(f'Calculating stats for {r} rotation roundtrips.')
            systems[i] = system.copy()
            # Inner loop rotates the system along each axis twice, times the number of iterations.
            for j in range(2*r):
                systems[i] = float_service.rotate_system(sys_ax=systems[i], sensor_angles=x_rot)
                systems[i] = float_service.rotate_system(sys_ax=systems[i], sensor_angles=y_rot)
                systems[i] = float_service.rotate_system(sys_ax=systems[i], sensor_angles=z_rot)

            max_cd, mean_cd, max_ad, mean_ad, mean_or = _stats_from_skewed_system(system, sys_2=systems[i])
            message += f'{r:<9} roundtrip(s):\n' \
                       f'Max coordinate drift = {max_cd:.7f}. Mean coordinate drift = {mean_cd:.7f}.\n' \
                       f'Max angular drift = {max_ad:.7f}. Mean angular drift = {mean_ad:.7f}.\n' \
                       f'Mean orthogonality: {mean_or:.7f}.\n'

        return message

    message_mlret = _multiple_linked_rotations_error_tests()
    print(message_mlret)

    ###########################################################

    if not passed_rotation_tests == rotation_tests:
        print(error_message)
    print('_______ rotate_system_test() ended _______\n')
    assert passed_rotation_tests == rotation_tests
