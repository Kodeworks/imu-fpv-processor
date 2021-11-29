# Contains all constants used throughout the project
# Note: some values are calculated based on other constants

# Physical constants
gravitational_constant = 9.81

# Sensor sampling frequency and period
sampling_rate = 104.0
sampling_period = 1.0 / sampling_rate

# Coordinate system transformations
perform_axis_reversal = True
gyro_reversal_coefficient = -1.0
axis_reversal_coefficient = -1.0
# With the current physical implementation, the y-axis (index 1) is reversed
axis_reversal_index = 1


# Damping of measurements
# Initial values
vel_damping_factor_initial = 0.001
pos_damping_factor_initial = 0.001

# damping factors right after a discarded package
vel_damping_factor_big = 0.05
pos_damping_factor_big = 0.05
# damping factor end points
vel_damping_factor_end = 0.001
pos_damping_factor_end = 0.001
# Finally, a damping factor to dampen the damping factors
damping_factor_dampener = 0.05

# Mini bursts
use_minibursts = True
miniburst_size = 128

# NaN value algorithm

# TODO: The claim below was accurate when assumption 1 wasn't true. The current threshold is lower and
#       should be examined
# Tests show that the module handles 70% randomly placed NaN-values. Threshold is set to 50% to accommodate
# for groupings of several NaN-values, which are more likely than a uniform distribution
discard_burst_nan_threshold = 0.5

# Adaptive Moving Average tuning
adaptive_alpha_max = 0.5
adaptive_alpha_gain = 0.01


# Tuning
# TODO: read file with tuning for each float


# Bias 
points_between_acc_bias_update = 256
points_between_gyro_bias_update = 256
points_between_vertical_acc_bias_update = 256
points_between_vertical_vel_bias_update = 32
points_between_vertical_pos_bias_update = 8

ignore_vertical_position_bias = False
ignore_vertical_velocity_bias = False

# Number of data points for calculating different means (prone to change)
n_points_for_acc_mean = 4096
n_points_for_gyro_mean = 4096
n_points_for_proper_vert_acc_mean = 4096
n_points_for_vel_mean = 512
n_points_for_pos_mean_initial = 512

min_points_for_pos_mean = 128

# Butterworth filter configuration
butterworth_cutoff_rate = 0.1
min_filter_size = 19

# Kalman filter
rows_per_kalman_use = 10

# Wave estimation parameters
n_points_for_fft = int(sampling_rate) * 10
n_points_between_fft = int(sampling_rate) * 5
n_saved_wave_functions = 50
# Determines whether or not wave function information is used in vertical position bias control
fft_aided_bias = False

# Output post processing
use_output_filtering = True

# identifiers of measured and calculated values
acc_x_identifier = 0
acc_y_identifier = 1
acc_z_identifier = 2
acc_identifiers = [0, 1, 2]

gyro_x_identifier = 3
gyro_y_identifier = 4
gyro_z_identifier = 5
gyro_identifiers = [3, 4, 5]

vertical_acc_identifier = 6
vertical_velocity_identifier = 7
vertical_position_identifier = 8
