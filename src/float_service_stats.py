from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

import config as cfg

from float_service import FloatService
from float_service_utils import ProcessSimulator


class FloatServiceStats:
    @staticmethod
    def plot_multiple_arrays(arrays, labels):
        plot_handles = []
        plt.figure()
        for i in range(len(arrays)):
            temp, = plt.plot(arrays[i], label=labels[i])
            plot_handles.append(temp)
        plt.legend(handles=plot_handles)

    @staticmethod
    def plot_biases_in_float_service(float_service: FloatService):
        plt.figure()
        plt.title('Bias in sensors and derived values')
        gyro_bias_x, = plt.plot(float_service.gyro_bias_array[:, 0],
                                c='xkcd:light blue',
                                label='Angular vel x bias')
        gyro_bias_y, = plt.plot(float_service.gyro_bias_array[:, 1],
                                c='xkcd:cerulean',
                                label='Angular vel y bias')
        x_acc_bias, = plt.plot(float_service.acc_bias_array[:, 0],
                               c='xkcd:green',
                               label='X-acceleration bias')
        y_acc_bias, = plt.plot(float_service.acc_bias_array[:, 1],
                               c='xkcd:lime green',
                               label='Y-acceleration bias')
        z_acc_bias, = plt.plot(float_service.acc_bias_array[:, 2],
                               c='xkcd:scarlet',
                               label='Z-acceleration bias')
        vert_vel_bias, = plt.plot(float_service.vertical_vel_bias_array,
                                  c='xkcd:bright orange',
                                  label='Vert. vel. bias')
        vert_pos_bias, = plt.plot(float_service.vertical_pos_bias_array,
                                  c='xkcd:yellow',
                                  label='Vert. pos. bias')
        plt.plot([0, len(float_service.orientation_mmap)], [0.0, 0.0], 'k:')
        plt.legend(handles=[gyro_bias_x, gyro_bias_y, x_acc_bias,
                            y_acc_bias, z_acc_bias, vert_vel_bias, vert_pos_bias])

    @staticmethod
    def plot_vertical_position(float_service: FloatService):
        plt.figure()
        plt.title('Vertical position')
        z_pos_dampened, = plt.plot(float_service.dampened_vertical_position, c='b', label='Dampened vert. pos')
        z_pos, = plt.plot(float_service.orientation_mmap[:, 2], c='g', label='Final vert. pos')
        z_pos_bias, = plt.plot(float_service.vertical_pos_bias_array, 'r:', label='Current vert. pos bias')
        plt.plot([0, len(float_service.orientation_mmap)], [0.0, 0.0], 'k:')
        plt.legend(handles=[z_pos_dampened, z_pos, z_pos_bias])

    @staticmethod
    def plot_vertical_velocity(float_service: FloatService):
        plt.figure()
        plt.title('Vertical velocity')
        z_acc, = plt.plot(float_service.actual_vertical_acceleration, c='tab:orange', label='Final vert. acc')
        z_vel_dampened, = plt.plot(float_service.dampened_vertical_velocity, c='b', label='Dampened vert. vel')
        z_vel, = plt.plot(float_service.dev_vertical_velocity, c='g', label='Final vert. vel')
        z_vel_bias, = plt.plot(float_service.vertical_vel_bias_array, 'r:', label='Bias, dampened vert. vel')
        plt.plot([0, len(float_service.orientation_mmap)], [0.0, 0.0], 'k:')
        plt.legend(handles=[z_vel_dampened, z_vel, z_acc, z_vel_bias])

    @staticmethod
    def plot_angle_processing(float_service: FloatService):
        fig_, axes = plt.subplots(1, 2)
        x_rot_acc, = axes[0].plot(float_service.dev_acc_state[:, 0], c='tab:purple', label='Acc only x-rotation')
        x_rot_gyro, = axes[0].plot(float_service.dev_gyro_state[:, 0] - np.mean(float_service.dev_gyro_state[:, 0]),
                                   c='tab:orange', label='Gyro only x-rotation')
        x_rot_kalman, = axes[0].plot(float_service.orientation_mmap[:, 0], c='tab:blue',
                                     label='Kalman estimate x-rotation')
        axes[0].plot([0, len(float_service.orientation_mmap)], [0.0, 0.0], 'k:')
        axes[0].legend(handles=[x_rot_acc, x_rot_gyro, x_rot_kalman])

        y_rot_acc, = axes[1].plot(float_service.dev_acc_state[:, 1], c='tab:purple', label='Acc only y-rotation')
        y_rot_gyro, = axes[1].plot(float_service.dev_gyro_state[:, 1] - np.mean(float_service.dev_gyro_state[:, 1]),
                                   c='tab:orange', label='Gyro only y-rotation')
        y_rot_kalman, = axes[1].plot(float_service.orientation_mmap[:, 1], c='tab:blue',
                                     label='Kalman estimate y-rotation')
        axes[1].plot([0, len(float_service.orientation_mmap)], [0.0, 0.0], 'k:')
        axes[1].legend(handles=[y_rot_acc, y_rot_gyro, y_rot_kalman])
        ylim = max(np.abs(float_service.dev_acc_state.flatten()))
        if ylim > 0.0:
            axes[0].set_ylim(-ylim, ylim)
            axes[1].set_ylim(-ylim, ylim)

    @staticmethod
    def plot_vertical_acceleration_processing(float_service: FloatService):
        plt.figure()
        plt.title('Vertical acceleration')
        # z-axis acc
        # actual z-acc
        # bank angle
        local_z_acc, = plt.plot(float_service.imu_mmap[:, 2] * 9.81,
                                c='r',
                                label='Local z-acc')
        processed_z_acc, = plt.plot(float_service.processed_input[:, 2],
                                    c='tab:green',
                                    label='Local z-acc, processed')
        actual_z_acc, = plt.plot(float_service.actual_vertical_acceleration,
                                 c='tab:purple',
                                 label='Actual vert. acc')
        plt.plot([0, len(float_service.orientation_mmap)], [0.0, 0.0], 'k:')
        plt.legend(handles=[local_z_acc, processed_z_acc, actual_z_acc])

    @staticmethod
    def fft_spectrum(arr, freq):
        y_f = fft(arr)
        x_f_final = np.linspace(0.0, freq / 2, len(arr) // 2)
        y_f_final = 2.0 / len(arr) * np.abs(y_f[0:len(arr) // 2])
        return x_f_final, y_f_final

    @staticmethod
    def plot_fft_spectrums(fls, offset: int = 0, cutoff: int = -1):
        """
        :param fls: FloatService object
        :param offset: Data rows to be cut from the beginning, so as to leave out the first messy minute or so
        :param cutoff: Last index not to be used
        """
        plt.figure()
        plt.title('Fourier transformed data')
        freq = fls.sampling_rate

        # Raw z-accelaration
        input_arr = fls.imu_mmap[offset:cutoff, 2] * 9.81
        na_vert_acc_fft_x, na_vert_acc_fft_y = FloatServiceStats.fft_spectrum(arr=input_arr, freq=freq)
        nvap, = plt.plot(na_vert_acc_fft_x, na_vert_acc_fft_y, c='xkcd:blue', label='Raw Z-axis acceleration')

        # Vertical acceleration
        input_arr = fls.actual_vertical_acceleration[offset:cutoff]
        vert_acc_fft_x, vert_acc_fft_y = FloatServiceStats.fft_spectrum(arr=input_arr, freq=freq)
        vap, = plt.plot(vert_acc_fft_x, vert_acc_fft_y, c='xkcd:light blue', label='Vertical acceleration')

        # Dampened vertical velocity
        input_arr = fls.dampened_vertical_velocity[offset:cutoff]
        vert_vel_dampened_fft_x, vert_vel_dampened_fft_y = FloatServiceStats.fft_spectrum(arr=input_arr, freq=freq)
        vvdp, = plt.plot(vert_vel_dampened_fft_x, vert_vel_dampened_fft_y, c='xkcd:red',
                         label='Dampened vertical velocity')

        # Final vertical velocity
        input_arr = fls.dev_vertical_velocity[offset:cutoff]
        vert_vel_final_fft_x, vert_vel_final_fft_y = FloatServiceStats.fft_spectrum(arr=input_arr, freq=freq)
        vvfp, = plt.plot(vert_vel_final_fft_x, vert_vel_final_fft_y, c='xkcd:orange',
                         label='Final vertical velocity')

        # Dampened vertical position
        input_arr = fls.dampened_vertical_position[offset:cutoff]
        vert_pos_dampened_fft_x, vert_pos_dampened_fft_y = FloatServiceStats.fft_spectrum(arr=input_arr, freq=freq)
        vpdp, = plt.plot(vert_pos_dampened_fft_x, vert_pos_dampened_fft_y, c='xkcd:dark green',
                         label='Dampened vertical position')

        # Final vertical position
        input_arr = fls.orientation_mmap[:, 2][offset:cutoff]
        vert_pos_final_fft_x, vert_pos_final_fft_y = FloatServiceStats.fft_spectrum(arr=input_arr, freq=freq)
        vpfp, = plt.plot(vert_pos_final_fft_x, vert_pos_final_fft_y, c='xkcd:lime green',
                         label='Final vertical position')

        plt.legend(handles=[nvap, vap, vvdp, vvfp, vpdp, vpfp])

    @staticmethod
    def plot_float_service_input(float_service):
        plt.figure()
        plt.title('Accelerometer input')
        acc_x_plot, = plt.plot(float_service.imu_mmap[:, 0], c='xkcd:green', label='X acc data (g)')
        proc_acc_x_plot, = plt.plot(float_service.processed_input[:, 0], c='xkcd:light grass green',
                                    label='Processed X acc data (m/s^2)')
        acc_y_plot, = plt.plot(float_service.imu_mmap[:, 1], c='xkcd:blue', label='Y acc data (g)')
        proc_acc_y_plot, = plt.plot(float_service.processed_input[:, 1], c='xkcd:light blue',
                                    label='Processed Y acc data (m/s^2)')
        acc_z_plot, = plt.plot(float_service.imu_mmap[:, 2], c='xkcd:red', label='Z acc data (g)')
        proc_acc_z_plot, = plt.plot(float_service.processed_input[:, 2], c='xkcd:coral pink',
                                    label='Processed Z acc data (m/s^2)')
        plt.legend(handles=[acc_x_plot, proc_acc_x_plot, acc_y_plot, proc_acc_y_plot, acc_z_plot, proc_acc_z_plot])

        plt.figure()
        plt.title('Gyroscope input')
        gyro_x_plot, = plt.plot(float_service.imu_mmap[:, 3], c='xkcd:green', label='X gyro data (deg/s)')
        proc_gyro_x_plot, = plt.plot(float_service.processed_input[:, 3], c='xkcd:light grass green',
                                     label='Processed X gyro data (rad/s)')
        gyro_y_plot, = plt.plot(float_service.imu_mmap[:, 4], c='xkcd:blue', label='Y gyro data (deg/s)')
        proc_gyro_y_plot, = plt.plot(float_service.processed_input[:, 4], c='xkcd:light blue',
                                     label='Processed Y gyro data (rad/s)')
        gyro_z_plot, = plt.plot(float_service.imu_mmap[:, 5], c='xkcd:red', label='Z gyro data (deg/s)')
        # proc_gyro_z_plot, = plt.plot(float_service.processed_input[:, 5], c='xkcd:coral pink',
        #                              label='Processed Z gyro data')
        plt.legend(handles=[gyro_x_plot, proc_gyro_x_plot, gyro_y_plot, proc_gyro_y_plot, gyro_z_plot])

    @staticmethod
    def plot_adaptive_average_and_alpha(float_service: FloatService):
        plt.figure()
        adav_acc_x, = plt.plot(float_service.bias_estimators[cfg.acc_x_identifier].adaptive_bias.adaptive_average_array,
                               c='xkcd:dark red', label='ADAV acc X')
        adav_acc_x_alpha, = plt.plot(float_service.bias_estimators[cfg.acc_x_identifier].adaptive_bias.alpha_array,
                                     c='xkcd:light red', label='acc X alpha')

        adav_acc_y, = plt.plot(float_service.bias_estimators[cfg.acc_y_identifier].adaptive_bias.adaptive_average_array,
                               c='xkcd:green', label='ADAV acc Y')
        adav_acc_y_alpha, = plt.plot(float_service.bias_estimators[cfg.acc_y_identifier].adaptive_bias.alpha_array,
                                     c='xkcd:light green', label='acc Y alpha')

        adav_acc_z, = plt.plot(float_service.bias_estimators[cfg.acc_z_identifier].adaptive_bias.adaptive_average_array,
                               c='xkcd:blue', label='ADAV acc Z')
        adav_acc_z_alpha, = plt.plot(float_service.bias_estimators[cfg.acc_z_identifier].adaptive_bias.alpha_array,
                                     c='xkcd:light blue', label='acc Z alpha')

        plt.legend(handles=[adav_acc_x, adav_acc_x_alpha, adav_acc_y, adav_acc_y_alpha, adav_acc_z, adav_acc_z_alpha])


if __name__ == '__main__':
    # Grab the path to the data folder relative to this file, regardless of where we invoke it from
    pwd = Path(__file__)
    data = pwd.parent.parent / "data"

    # Default options
    default_data_path = str(data / "wave_like_office_generated_data_210507_1529.hdf5")
    default_sensor_id = "5"
    default_burst_size = 1000

    # Create a CLI interface and set some default values
    parser = ArgumentParser("float-service-stats", description="Float Service Statistics")
    parser.add_argument("--data_path", help="The path to the HDF5 file containing the data", default=default_data_path)
    parser.add_argument("--sensor_id", default="5", help="The sensor ID number")
    parser.add_argument("--burst_size", default=1000, type=int, help="The burst size")

    # Parse and set the options
    args = parser.parse_args()
    data_path = args.data_path
    sensor_id = args.sensor_id
    burst_size = args.burst_size

    process_sim = ProcessSimulator(
        hdf5_path=data_path,
        dev_mode=True
    )

    process_sim.all_bursts_single_float_service(sensor_id=sensor_id)

    FloatServiceStats.plot_biases_in_float_service(process_sim.float_services[sensor_id])
    FloatServiceStats.plot_vertical_position(process_sim.float_services[sensor_id])
    FloatServiceStats.plot_vertical_velocity(process_sim.float_services[sensor_id])
    FloatServiceStats.plot_vertical_acceleration_processing(process_sim.float_services[sensor_id])
    FloatServiceStats.plot_angle_processing(process_sim.float_services[sensor_id])

    FloatServiceStats.plot_adaptive_average_and_alpha(process_sim.float_services[sensor_id])
    FloatServiceStats.plot_float_service_input(process_sim.float_services[sensor_id])
    plt.show()
