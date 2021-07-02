from src import float_service as fs

import time
import sys

import numpy as np
from scipy import interpolate
import h5py


class ProcessSimulator:
    def __init__(self,
                 hdf5_path: str,
                 buffer_size: int = 1_000_000_000,
                 data_rows_total: int = 1_000_000_000,
                 dev_mode: bool = False):
        """
        :param dev_mode: Whether or not to use the FloatService-class in development mode.
        """
        self.data_path = hdf5_path
        self.data_rows_total = data_rows_total
        self.buffer_size = buffer_size
        self.dev_mode = dev_mode

        fs.n_rows = buffer_size

        self.float_services = {}
        self.last_rows = {}
        self.all_input_data = {}
        self.outputs = {}
        self.inputs = {}
        self.all_input_data_pointer = {}

        with h5py.File(self.data_path, 'r') as f:
            self.sensor_ids = list(f.keys())
            max_rows = len(f[self.sensor_ids[0]]['data'])
            if data_rows_total > len(f[self.sensor_ids[0]]['data']):
                print(f'Requested number of rows, {data_rows_total}, is larger than max, {max_rows}. '
                      f'Requested set to max.')
                self.data_rows_total = max_rows
            if self.buffer_size > self.data_rows_total:
                print(f'Requested buffer size, {self.buffer_size}, is larger than available data,'
                      f' {self.data_rows_total}. Buffer size set to {self.data_rows_total}.')
                self.buffer_size = self.data_rows_total
                fs.n_rows = self.buffer_size
            for k in self.sensor_ids:
                self.add_float_service(k)

    def add_float_service(self, sensor_id):
        with h5py.File(self.data_path, 'r') as f:
            self.all_input_data[sensor_id] = np.zeros(shape=[self.data_rows_total, 6], dtype=float)
            self.inputs[sensor_id] = np.zeros(shape=[self.buffer_size, 6])
            self.outputs[sensor_id] = np.zeros(shape=[self.buffer_size, 3], dtype=float)
            temp_rows = min(self.data_rows_total, len(f[sensor_id]['data'][:]))
            self.all_input_data[sensor_id][:temp_rows] = f[sensor_id]['data'][:temp_rows]
            self.float_services[sensor_id] = fs.FloatService(name=sensor_id,
                                                             input=self.inputs[sensor_id],
                                                             output=self.outputs[sensor_id],
                                                             dev_mode=self.dev_mode)

            self.last_rows[sensor_id] = -1
            self.all_input_data_pointer[sensor_id] = 0

    def reset_float_service(self, sensor_id: str, offset: int = 0):
        """
        Reset FloatService, possibly with an offset in number of data points.

        :param sensor_id: ID of the FloatService to be reset.
        :param offset: Number of bursts skipped.
        """
        if offset*26 >= self.buffer_size:
            print('reset_float_service():\n'
                  f'Offset of {offset*26} lines is too large for buffer size of {self.buffer_size}.\n'
                  f'Aborting.\n')
            return
        self.float_services.pop(sensor_id)
        self.inputs.pop(sensor_id)
        self.outputs.pop(sensor_id)
        self.all_input_data.pop(sensor_id)

        self.add_float_service(sensor_id=sensor_id)
        self.last_rows[sensor_id] = offset * 26 - 1
        self.all_input_data_pointer[sensor_id] = offset * 26
        self.float_services[sensor_id].last_row = offset * 26 - 1

    def all_bursts(self, burst_size: int = 26):
        print('Processing all requested data...')
        start = time.time()

        # Progress tracking for visual confirmation
        n_bursts = self.data_rows_total // burst_size
        progress_index = 0
        progress_bars_per_sensor = 5
        total_progress = len(self.float_services)*progress_bars_per_sensor
        total_bursts = len(self.float_services)*n_bursts
        progress_milestone = total_bursts//total_progress
        p = 0
        for i in range(n_bursts):
            for k in self.sensor_ids:
                if progress_milestone != 0 and p % progress_milestone == 0:
                    sys.stdout.write(self.get_progress_string(progress_index, total_progress))
                    progress_index += 1
                self.next_burst(sensor_id=k, burst_size=burst_size)
                p += 1

        sys.stdout.write(self.get_progress_string(progress_index, total_progress))
        print(f'Data processed in {(time.time()-start):.3f}s')

    def process_buffer(self, sensor_id: str):
        """
        Returns:
            1 - Buffer processed, total data not depleted
            2 - Buffer processed, total data depleted
        """
        data_status = self.next_burst(sensor_id=sensor_id, wait_on_buffer_reuse=False)
        while data_status == 1:
            data_status = self.next_burst(sensor_id=sensor_id, wait_on_buffer_reuse=True)
        if data_status == 2:
            # Buffer is processed but data is not depleted
            return 1
        if data_status == 3:
            # Buffer is processed and data is depleted
            return 2

    def all_bursts_single_float_service(self, sensor_id: str, offset: int = 0, burst_size: int = 1000,
                                        print_progress: bool = False):
        if print_progress:
            print(f'Processing {self.data_rows_total} data rows with buffer size '
                  f'{self.buffer_size} for sensor {sensor_id}')
        start = time.time()

        # Progress tracking for visual confirmation
        actual_burst_number = self.data_rows_total // burst_size
        progress_index = 0
        total_progress = 10
        total_bursts = actual_burst_number - offset
        progress_milestone = total_bursts//total_progress
        p = 0

        for i in range(actual_burst_number - offset):
            if progress_milestone != 0 and p % progress_milestone == 0:
                if print_progress:
                    sys.stdout.write(self.get_progress_string(progress_index, total_progress))
                progress_index += 1
            self.next_burst(sensor_id=sensor_id, burst_size=burst_size)
            p += 1

        if print_progress:
            sys.stdout.write(self.get_progress_string(progress_index, total_progress))
        if print_progress:
            print(f'Data from {sensor_id} processed in {(time.time() - start):.3f}s')

    def next_burst(self, sensor_id: str, burst_size: int = 26, wait_on_buffer_reuse: bool = False):
        """
        Returns:
            1 - burst processed
            2 - burst aborted, marking the beginning of a buffer reuse
            3 - burst processed, total input data depleted
        """
        # TODO: Solve this as well
        try:
            # Check if entire burst can be added to buffer
            if self.last_rows[sensor_id] + 1 + burst_size <= self.buffer_size:
                # Add burst
                self.inputs[sensor_id][self.last_rows[sensor_id] + 1: self.last_rows[sensor_id] + 1 + burst_size] = \
                    self.all_input_data[sensor_id][self.all_input_data_pointer[sensor_id]:
                                                  self.all_input_data_pointer[sensor_id] + burst_size]
                self.all_input_data_pointer[sensor_id] += burst_size
                # Process
                self.float_services[sensor_id].process(number_of_rows=burst_size)
                self.last_rows[sensor_id] += burst_size
            else:
                if wait_on_buffer_reuse:
                    return 2
                self.inputs[sensor_id][0: burst_size] = \
                    self.all_input_data[sensor_id][self.all_input_data_pointer[sensor_id]:
                                                  self.all_input_data_pointer[sensor_id] + burst_size]
                self.all_input_data_pointer[sensor_id] += burst_size
                # Process
                self.float_services[sensor_id].process(number_of_rows=burst_size)
                self.last_rows[sensor_id] = burst_size - 1
                if self.all_input_data_pointer[sensor_id] + burst_size >= self.data_rows_total:
                    self.last_rows[sensor_id] = self.data_rows_total - self.all_input_data_pointer[sensor_id] - 1
                    return 3
        except ValueError as e:
            print(e)
            print(self.last_rows[sensor_id], burst_size, self.buffer_size, self.all_input_data_pointer[sensor_id])
        return 1

    @staticmethod
    def get_progress_string(progress: int, end: int = 9):
        """
        Return progress bar using a number 0-10
        """
        if progress > end:
            return ''

        out = '\rProgress: |'
        for i in range(progress):
            out += '|'

        for i in range(end-progress):
            out += '.'

        out += '|'

        if progress == end:
            out += '\n'

        return out


class FloatServiceHandler:
    def __init__(self, buffer_sizes: int):
        fs.n_rows = buffer_sizes
        self.float_services = {}
        self.sensor_ids = []
        self.input_buffers = {}
        self.output_buffers = {}
        self.last_line_counters = {}

    def process(self, sensor_id: str, burst):
        if sensor_id in self.sensor_ids:
            self._process(sensor_id=sensor_id, burst=burst)

        else:
            self.add_float_service(sensor_id=sensor_id)
            self._process(sensor_id=sensor_id, burst=burst)

    def _process(self, sensor_id: str, burst):
        if self.last_line_counters[sensor_id] + len(burst) + 1 <= fs.n_rows:
            self.input_buffers[sensor_id][self.last_line_counters[sensor_id] + 1:
                                         self.last_line_counters[sensor_id] + 1 + len(burst)] \
                = burst
            self.last_line_counters[sensor_id] += len(burst)
        else:
            self.last_line_counters[sensor_id] = len(burst) - 1
            self.input_buffers[sensor_id][0:len(burst)] = burst

        self.float_services[sensor_id].process(number_of_rows=len(burst))

    def add_float_service(self, sensor_id: str):
        self.input_buffers[sensor_id] = np.zeros(shape=[fs.n_rows, 6], dtype=float)
        self.output_buffers[sensor_id] = np.zeros(shape=[fs.n_rows, 3], dtype=float)
        self.last_line_counters[sensor_id] = -1
        self.sensor_ids.append(sensor_id)
        new_float_service = fs.FloatService(name=sensor_id,
                                            input=self.input_buffers[sensor_id],
                                            output=self.output_buffers[sensor_id],
                                            dev_mode=True)

        self.float_services[sensor_id] = new_float_service


class SensorDataRow:
    def __init__(self, row_string: str = None, sep: str = '\t', order: str = 'its'):
        """
        Order rules:
        i - ID
        t - Timestamp
        s - IMU data
        p - skip single element
        """
        self.sensor_id = None
        self.timestamp = None
        self.imu_data = None

        self.successful_data_extraction = True

        self.char_to_data_method = \
            {'i': self.set_id,
             't': self.attempt_parse_timestamp,
             's': self.attempt_parse_imu,
             'p': self.skip_element}
        self.char_to_num_of_elements = \
            {'i': 1,
             't': 1,
             's': 6,
             'p': 1}

        if row_string is not None:
            self.extract_data_row_from_string(data_string=row_string,
                                              sep=sep,
                                              order=order)

    def set_id(self, sequence: list):
        self.sensor_id = sequence[0]

    def attempt_parse_timestamp(self, sequence: list):
        try:
            self.timestamp = int(sequence[0])
        except ValueError:
            try:
                self.timestamp = float(sequence[0])
            except ValueError:
                self.successful_data_extraction = False

    def attempt_parse_imu(self, sequence: list):
        try:
            self.imu_data = [float(el) for el in sequence]
        except ValueError:
            self.successful_data_extraction = False

    def skip_element(self, sequence: list):
        pass

    def extract_data_row_from_string(self, data_string: str, sep: str = '\t', order: str = 'its'):
        """
        Order rules:
        i - ID
        t - Timestamp
        s - IMU data
        p - skip single element
        """
        split_string = data_string.split(sep=sep)

        # The extraction should not go through if the number of elements in the given split string does noe match the
        # expectation given a set of operations
        expected_elements = 0
        for operation in order:
            expected_elements += self.char_to_num_of_elements[operation]
        if expected_elements != len(split_string):
            self.successful_data_extraction = False
            return

        progress = 0
        for operation in order:
            self.char_to_data_method[operation] \
                (sequence=split_string[progress: progress+self.char_to_num_of_elements[operation]])
            progress += self.char_to_num_of_elements[operation]

        #
        # try:
        #     sensor_id = int(split_string[0])
        # except ValueError as e:
        #     return None
        # try:

    def clear_data(self):
        self.sensor_id = None
        self.timestamp = None
        self.imu_data = None
        self. successful_data_extraction = True


class SensorDataBurst:
    def __init__(self):
        self.sensor_id = None
        self.imu_data = []
        self.timestamps = []

    def add_data_row(self, data_row: SensorDataRow):
        if self.sensor_id is None:
            self.sensor_id = data_row.sensor_id

        if data_row.sensor_id == self.sensor_id:
            self.timestamps.append(data_row.timestamp)
            self.imu_data.append(data_row.imu_data)
            return True
        else:
            return False

    def clear_data(self):
        self.sensor_id = None
        self.imu_data = []
        self.timestamps = []


class SensorDataset:
    """
    Input read order rules:
    i - ID
    t - Timestamp
    s - IMU data
    p - skip single element
    """
    def __init__(self):
        self.imu_data = {}
        self.timestamps = {}

        self.output_dict = {}

        self.max_sensors = 256

        self.max_sensors_exeeded = False

    def add_burst(self, burst: SensorDataBurst):
        if burst.sensor_id in list(self.imu_data.keys()):
            self.imu_data[burst.sensor_id] += burst.imu_data
            self.timestamps[burst.sensor_id] += burst.timestamps
        else:
            self.imu_data[burst.sensor_id] = []
            self.timestamps[burst.sensor_id] = []
            self.add_burst(burst=burst)

    def read_input_from_csv_file(self, file_path: str, sep: str = '\t', order: str = 'its'):
        with open(file=file_path, mode='r') as data_file:
            line_str = data_file.readline()
            burst = SensorDataBurst()
            data_row = SensorDataRow()
            while line_str != '':
                data_row.extract_data_row_from_string(data_string=line_str, sep=sep, order=order)
                if data_row.successful_data_extraction:
                    if not burst.add_data_row(data_row=data_row):
                        self.add_burst(burst)
                        burst.clear_data()
                        burst.add_data_row(data_row=data_row)
                data_row.clear_data()
                line_str = data_file.readline()

                if not self.confirm_validity_of_internal_data():
                    self.print_invalidity_of_internal_data()
                    self.reset_dataset()
                    return

            self.add_burst(burst=burst)

        for key in list(self.imu_data.keys()):
            self.imu_data[key] = np.array(self.imu_data[key])
            print(f'Read {len(self.imu_data[key])} from sensor {key}.')

    def read_input_from_hdf5_file(self, file_path: str):
        with h5py.File(name=file_path, mode='r') as hdf5_file:
            for key in list(hdf5_file.keys()):
                if key not in list(self.imu_data.keys()):
                    self.imu_data[key] = hdf5_file[key]['data'][:]
                    if key in list(self.timestamps.keys()):
                        self.timestamps.pop(key)
                    if 'timestamps' in list(hdf5_file[key].keys()):
                        self.timestamps[key] = hdf5_file[key]['timestamps']
                    else:
                        print(f'No timestamp data was found for sensor {key}')
                    print(f'Data from sensor {key} was read.')
                else:
                    print(f'Data from sensor {key} is already registered. New data rejected.')

    def write_input_to_hdf5_file(self, file_path: str):
        if not self.imu_data:
            print('write_input_to_hdf5_file():\n'
                  'There is no input to write to an .hdf5-file. Aborting')
            return
        elif max([len(self.imu_data[key]) for key in list(self.imu_data.keys())]) == 0:
            print('write_input_to_hdf5_file():\n'
                  'All input kept in this dataset is empty. Aborting')
            return

        with h5py.File(name=file_path, mode='w') as new_file:
            for key in list(self.imu_data.keys()):
                sensor_specific_group = new_file.create_group(name=key)
                sensor_specific_group.create_dataset(name='data', data=self.imu_data[key])

                # In order to save timestamps in .hdf5-format, None's are replaced by 0's
                temp_timestamps = [val or 0 for val in self.timestamps[key]]
                # If it turned out that all timestamps were None, do nothing
                if not max(temp_timestamps) == 0:
                    sensor_specific_group.create_dataset(name='timestamps', data=temp_timestamps)

    def write_output_to_hdf5_file(self, output_path: str):
        if not self.output_dict:
            return

        with h5py.File(name=output_path, mode='w') as output_file:
            for key in list(self.output_dict.keys()):
                sensor_specific_group = output_file.create_group(key)
                sensor_specific_group.create_dataset(name='data', data=self.output_dict[key])
                sensor_specific_group.create_dataset(name='timestamps', data=self.output_dict[key])

    def interpolate_input_imu_data_to_length_of_longest_dataset(self):
        if not self.imu_data:
            print('interpolate_input_to_length_of_longest_dataset():\n'
                  'There is no input IMU data. Aborting')
            return
        end_lengths = []
        for key in list(self.imu_data.keys()):
            end_lengths.append(len(self.imu_data[key]))
        end_length = max(end_lengths)

        if end_length == 0:
            print('interpolate_input_to_length_of_longest_dataset():\n'
                  'Longest IMU dataset contains 0 rows. Aborting')
            return

        for key in list(self.imu_data.keys()):
            new_imu_data_temp = np.zeros(shape=[end_length, np.shape(self.imu_data[key])[1]])
            for i in range(np.shape(self.imu_data[key])[1]):
                new_imu_data_temp[:, i] = self.interpolate_array_to_specific_length(
                    array=self.imu_data[key][:, i],
                    length=end_length
                )
            self.imu_data[key] = new_imu_data_temp

        print(f'All IMU data was interpolated to a length of {end_length} rows')

    @staticmethod
    def interpolate_array_to_specific_length(array, length: int):
        original_length = len(array)
        x_series = np.linspace(0.0, length, original_length, dtype=float)
        func = interpolate.interp1d(x=x_series, y=array)
        x_series_new = np.arange(length, dtype=float)
        new_array = func(x_series_new)

        return new_array

    def partition_input_by_id_selection(self, ids_to_keep: list):
        if not self.imu_data:
            print('partition_input_by_id_selection():\n'
                  'There is no input IMU data. Aborting')
            return
        for key in list(self.imu_data.keys()):
            if key in ids_to_keep:
                self.imu_data.pop(key)
                self.timestamps.pop(key)
                ids_to_keep.remove(key)

        if ids_to_keep:
            print(f'partition_input_by_id_selection():\n'
                  f'The following requested ID\'s were not found:\n'
                  f'{ids_to_keep}')

    def map_10th_milli_timestamps_to_total_time(self):
        if not self.timestamps:
            print('convert_timestamps_to_total_time_in_secs():\n'
                  'No timestamp data available. Aborting')
            return
        roll_over_time = 65536
        for key in list(self.timestamps.keys()):
            n_roll_overs = 0
            new_temp_time = np.copy(self.timestamps[key])
            for i in range(1, len(self.timestamps[key])):
                if self.timestamps[key][i-1] > self.timestamps[key][i]:
                    n_roll_overs += 1
                new_temp_time[i] += roll_over_time * n_roll_overs

            self.timestamps[key] = new_temp_time

    def convert_10th_milli_timestamps_to_secs(self):
        if not self.timestamps:
            print('convert_10th_milli_timestamps_to_secs():\n'
                  'No timestamp data available. Aborting')
            return
        for key in list(self.timestamps.keys()):
            self.timestamps[key] = self.timestamps[key] / 10_000

    def start_timestamps_at_zero_time(self):
        if not self.timestamps:
            print('start_timestamps_at_zero_time():\n'
                  'No timestamp data available. Aborting')
            return
        for key in list(self.timestamps.keys()):
            self.timestamps[key] -= self.timestamps[key][0]

    def generate_float_service_output(self):
        if not self.imu_data:
            print('generate_float_service_output():\n'
                  'There is no input IMU data. Aborting')
            return

        for key in list(self.imu_data.keys()):
            self.output_dict[key] = np.zeros(shape=[len(self.imu_data[key]), 3],
                                             dtype=float)
            fs_temp = fs.FloatService(name='temp',
                                      input=self.imu_data[key],
                                      output=self.output_dict[key])
            fs.n_rows = len(self.imu_data[key])
            fs_temp.process(number_of_rows=len(self.imu_data[key]))

    def confirm_validity_of_internal_data(self):
        if len(self.imu_data) > self.max_sensors:
            self.max_sensors_exeeded = True
            return False
        else:
            self.max_sensors_exeeded = False

        return True

    def print_invalidity_of_internal_data(self):
        if self.max_sensors_exeeded:
            print(f'Max number of sensors - {self.max_sensors} - exeeded.')

    def reset_dataset(self):
        self.imu_data = {}
        self.timestamps = {}
        self.output_dict = {}

        self.max_sensors_exeeded = False

        print('All data wiped and variables reset.')
