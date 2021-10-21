import math
import random
import sys

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import typing
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

feature_name_white_list = ['/Plugins/Magnetometer_Z', '/Plugins/Gyroscope_Z', '/Plugins/Accelerometer_X',
                           '/CAN/WheelSpeed_RL', '/Plugins/GNSS_status', '/CAN/WheelSpeed_RR', '/CAN/WheelSpeed_FR',
                           '/CAN/EngineSpeed_CAN', '/GPS/Direction', '/CAN/ENG_Trq_ZWR', '/CAN/AccPedal',
                           '/Plugins/Slip_angle', '/CAN/Yawrate1', '/CAN/VehicleSpeed', '/GPS/Used satellites',
                           '/CAN/Trq_FrictionLoss', '/Plugins/Magnetometer_Y', '/GPS/Z', '/CAN/Trq_Indicated',
                           '/CAN/OilTemperature1', '/CAN/WheelSpeed_FL', '/Plugins/Body_acceleration_X',
                           '/CAN/ENG_Trq_DMD', '/Plugins/Magnetometer_X', '/CAN/EngineTemperature', '/Plugins/Pitch',
                           '/GPS/Acceleration', '/Plugins/Gyroscope_X', '/Plugins/Velocity_X', '/Plugins/Gyroscope_Y',
                           '/GPS/Velocity', '/CAN/ENG_Trq_m_ex', '/GPS/Distance', '/CAN/SteerAngle1',
                           '/CAN/AirIntakeTemperature']
feature_name_grey_list = ['/CAN/Engine_02_BZ', '/CAN/Engine_02_CHK', '/CAN/SCS_01_BZ', '/CAN/SCS_01_CHK']


def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = f'{prefix}/{key}'
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield path, item
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from h5py_dataset_iterator(item, path)


def traverse_datasets(hdf_file):
    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


class DatasetReader:
    def __init__(self, file_names: typing.Iterable):
        self.files = {}
        for file_name in file_names:
            self.files[file_name] = h5py.File(os.path.join(ROOT_DIR, "data", file_name), 'r')

    def print_structure(self, file_names: list):
        for target_file in file_names:
            print("=====\n", target_file)
            file = self.files[target_file]
            for dset in traverse_datasets(file):
                print('\tPath:', dset)
                print('\t\tShape:', file[dset].shape)
                print('\t\t type:', file[dset].dtype)

    def _concatenate_data(self, file_names: list, feature_names: list):
        time_step_size_array = []  # record time step size per file
        # concatenate all features from all files in one matrix of shape [sum_of_time_steps_of_all_files, feature_num]
        all_data_time_serials = []
        for file_name in file_names:
            file = self.files[file_name]
            data_time_serials = []
            size = 0
            for feature_name in feature_names:
                time_serial = np.array(file[feature_name][:, 0]).reshape([-1, 1])
                size = len(time_serial)
                data_time_serials.append(time_serial)
            data_time_serials = np.hstack(data_time_serials)  # shape: [time_step_length, feature_num]
            time_step_size_array.append(size)
            all_data_time_serials.append(data_time_serials)
        all_data_time_serials = np.vstack(all_data_time_serials)
        return all_data_time_serials, time_step_size_array

    def _de_concatenate_data(self, time_serials, size_array, feature_weights, time_steps=100, sample_interval=5,
                             target_sequence=False,
                             ignore_all_zero=True, target_features_fusion=False, target_skip_steps=0):
        input_time_serials = []
        target_time_serials = []

        accumulated_size = 0
        for size in size_array:
            data_time_serials = time_serials[accumulated_size: accumulated_size + size]
            accumulated_size += size
            for i in range(0, size - time_steps - target_skip_steps - 1, sample_interval):
                a_sample = data_time_serials[i:i + time_steps]
                if ignore_all_zero and not np.all(a_sample == 0):
                    continue
                input_time_serials.append(a_sample)
                if target_sequence:
                    if target_features_fusion:
                        target_time_serials.append(
                            np.dot(data_time_serials[i + target_skip_steps + 1: i + target_skip_steps + time_steps + 1],
                                   feature_weights.T))
                    else:
                        target_time_serials.append(
                            data_time_serials[i + target_skip_steps + 1: i + target_skip_steps + time_steps + 1])
                else:
                    if target_features_fusion:
                        target_time_serials.append(
                            [[np.dot(data_time_serials[i + target_skip_steps + time_steps], feature_weights.T)]])
                    else:
                        target_time_serials.append([data_time_serials[i + target_skip_steps + time_steps]])
        input_time_serials = np.array(input_time_serials)
        target_time_serials = np.array(target_time_serials)
        # if random_order:
        #     input_time_serials, target_time_serials = shuffle(input_time_serials, target_time_serials)
        return input_time_serials, target_time_serials

    def check_features(self, feature_names):
        for file_name in self.files.keys():
            file = self.files[file_name]
            for feature_name in feature_names:
                a = file[feature_name]

    def sample(self, feature_names=None, time_steps=100, sample_interval=5, target_sequence=False,
               ignore_all_zero=True, target_features_fusion=False, test_on_file=False, test_files=None,
               target_skip_steps=0):

        if not feature_names or len(feature_names) == 0:
            feature_names = self.get_all_features()

        train_files = self.files.keys()

        if not test_files or len(test_files) == 0:
            test_file_name = ''
            test_file_size = sys.maxsize
            for file_name in self.files.keys():
                file_size = os.path.getsize(os.path.join(ROOT_DIR, "data", file_name))
                if file_size < test_file_size:
                    test_file_size = file_size
                    test_file_name = file_name
            # test_files = [test_file_name]
            train_files_list = list(train_files)
            # print(train_files_list)
            ranIdx = random.randint(0, len(train_files_list)-1)
            # print(ranIdx)
            # print(train_files_list[ranIdx])
            test_files = [train_files_list[ranIdx]]
        train_files = list(set(self.files.keys()) - set(test_files))

        train_data_time_serials, train_time_step_size_array = self._concatenate_data(file_names=train_files,
                                                                                     feature_names=feature_names)

        test_data_time_serials, test_time_step_size_array = self._concatenate_data(file_names=test_files,
                                                                                   feature_names=feature_names)
        assert len(test_time_step_size_array) == 1

        if not test_on_file:
            append_train_size = test_time_step_size_array[0]//2
            # print(append_train_size)
            # print(train_data_time_serials.shape)
            # print(test_data_time_serials.shape)
            # exit(0)
            train_data_time_serials = np.vstack([train_data_time_serials, test_data_time_serials[:append_train_size]])
            # np.concatenate(train_data_time_serials, test_data_time_serials[:append_train_size])
            train_time_step_size_array.append(append_train_size)
            test_data_time_serials = test_data_time_serials[append_train_size:]
            test_time_step_size_array[0] -= append_train_size

        scaler = MinMaxScaler()
        # feature_scalers = []
        normalized_train_data_time_serials = []
        normalized_test_data_time_serials = []

        for i in range(len(feature_names)):
            train_time_serial = train_data_time_serials[:, i].reshape([-1, 1])
            scaler.fit(train_time_serial)
            normalized_train_data_time_serials.append(scaler.transform(train_time_serial))
            test_time_serial = test_data_time_serials[:, i].reshape([-1, 1])
            normalized_test_data_time_serials.append(scaler.transform(test_time_serial))

        normalized_train_data_time_serials = np.hstack(normalized_train_data_time_serials)
        normalized_test_data_time_serials = np.hstack(normalized_test_data_time_serials)

        train_feature_vars = np.var(normalized_train_data_time_serials, axis=0)
        train_feature_weights = train_feature_vars / (sum(train_feature_vars) + 1)

        train_input_time_serials, train_target_time_serials = self._de_concatenate_data(
            normalized_train_data_time_serials,
            train_time_step_size_array,
            train_feature_weights,
            time_steps=time_steps,
            sample_interval=sample_interval,
            target_sequence=target_sequence,
            ignore_all_zero=ignore_all_zero,
            target_features_fusion=target_features_fusion,
            target_skip_steps=target_skip_steps)

        test_input_time_serials, test_target_time_serials = self._de_concatenate_data(
            normalized_test_data_time_serials,
            test_time_step_size_array,
            train_feature_weights,
            time_steps=time_steps,
            sample_interval=1,
            target_sequence=target_sequence,
            ignore_all_zero=False,
            target_features_fusion=target_features_fusion,
            target_skip_steps=target_skip_steps)

        train_input_time_serials, train_target_time_serials = shuffle(train_input_time_serials, train_target_time_serials)
        return train_input_time_serials, train_target_time_serials, test_input_time_serials, test_target_time_serials, feature_names, normalized_test_data_time_serials

    def draw_one_path(self, file_name, feature_name):
        file = self.files[file_name]
        data_time_serial = np.array(file[feature_name])
        print(data_time_serial.shape)
        plt.plot(data_time_serial[-500:, 1], data_time_serial[-500:, 0])
        plt.title(feature_name)
        plt.show()
        return

    def get_all_features(self, size=10):
        # feature_names_set = set()
        #
        # for file_name, file in self.files.items():
        #     feature_names_in_one_set = set()
        #     for dset in traverse_datasets(file):
        #         data = file[dset]
        #         if len(data.shape) < 2:
        #             continue
        #         var = np.var(data[:, 0], axis=-1)
        #         if var > 1:
        #             feature_names_in_one_set.add(dset)
        #
        #     if len(feature_names_set) == 0:
        #         feature_names_set = feature_names_in_one_set
        #     else:
        #         feature_names_set = feature_names_set.intersection(feature_names_in_one_set)
        #
        # feature_names = shuffle(list(feature_names_set))
        np.random.shuffle(feature_name_white_list)
        return feature_name_white_list[:size]


if __name__ == '__main__':
    data_reader = DatasetReader(["20181117_Driver1_Trip7.hdf"])

    # file = data_reader.files["20181117_Driver1_Trip7.hdf"]
    # feature_names_in_one_set = set()
    # for dset in traverse_datasets(file):
    #     data = file[dset]
    #     if len(data.shape) < 2:
    #         continue
    #     var = np.var(data[:, 0], axis=-1)
    #     if var > 1:
    #         feature_names_in_one_set.add(dset)
    #         data_reader.draw_one_path("20181117_Driver1_Trip7.hdf", dset)
    #
    # print(feature_names_in_one_set)
    # data_reader.print_structure(["20181113_Driver1_Trip1.hdf"])
    # exit(0)
    data_reader.draw_one_path("20181117_Driver1_Trip7.hdf", "/GPS/Distance")
