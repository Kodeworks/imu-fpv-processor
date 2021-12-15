import numpy as np
import matplotlib.pyplot as plt

from src.float_service import FloatService
from src.float_service_utils import ProcessSimulator


def verify_predict_angles(hdf5_path: str, sensor_id, t: float, x_limits: tuple = None, y_limits: tuple = None):
    process_sim = ProcessSimulator(hdf5_path=hdf5_path)
    process_sim.all_bursts_single_float_service(sensor_id=sensor_id)

    float_service = process_sim.float_services[sensor_id]

    n_rows = len(float_service.output)

    # Extract some information from float_service
    x_angle_output = float_service.output[:, 0]
    y_angle_output = float_service.output[:, 1]

    # Compute angle predictions
    x_angle_predictions = np.zeros_like(x_angle_output)
    y_angle_predictions = np.zeros_like(y_angle_output)
    for i in range(n_rows):
        x_angle_predictions[i], y_angle_predictions[i] = float_service.predict_angles(row_no=i, t=t)

    # Time steps are needed instead of indices, to accurately compare predictions to actual output
    time_steps_output = np.linspace(start=0.0, stop=float_service.sampling_period * n_rows, num=n_rows)
    time_steps_predictions = time_steps_output + t

    # Plot the results
    fig, axes = plt.subplots(2, 1)

    axes[0].set_title(f'X-angle predictions at t={t} s')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Angle (rad)')
    x_output_handle, = \
        axes[0].plot(time_steps_output, x_angle_output, label='X-angle', c='xkcd:dark red')
    x_prediction_handle, = \
        axes[0].plot(time_steps_predictions, x_angle_predictions, label=f'X-angle prediction', c='xkcd:light red')

    axes[1].set_title(f'Y-angle predictions at t={t} s')
    axes[1].set_xlabel('Time(s)')
    axes[1].set_ylabel('Angle (rad)')
    y_output_handle, = \
        axes[1].plot(time_steps_output, y_angle_output, label='Y-angle', c='xkcd:dark blue')
    y_prediction_handle, = \
        axes[1].plot(time_steps_predictions, y_angle_predictions, label=f'Y-angle prediction', c='xkcd:light blue')

    axes[0].legend(handles=[x_output_handle, x_prediction_handle])
    axes[1].legend(handles=[y_output_handle, y_prediction_handle])

    if x_limits is not None:
        axes[0].set_xlim(x_limits[0], x_limits[1])
        axes[1].set_xlim(x_limits[0], x_limits[1])
    if y_limits is not None:
        axes[0].set_ylim(y_limits[0], y_limits[1])
        axes[1].set_ylim(y_limits[0], y_limits[1])

    plt.show()


if __name__ == '__main__':
    predict_angles = True
    if predict_angles:
        data_path = '../data/wave_like_office_generated_data_210507_1529.hdf5'
        sensor_id = '5'
        prediction_time = 0.05
        verify_predict_angles(hdf5_path=data_path, sensor_id=sensor_id, t=prediction_time,
                              x_limits=(350, 380), y_limits=(-0.4, 0.4))
