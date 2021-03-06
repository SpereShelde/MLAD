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
            self.files[file_name] = h5py.File(os.path.join(ROOT_DIR, "../data", 'canbus', file_name), 'r')

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

    def _de_concatenate_data(self, time_serials, size_array, time_steps=100, sample_interval=5,
                             target_sequence=False, target_skip_steps=0):
        input_time_serials = []
        target_time_serials = []

        accumulated_size = 0
        for size in size_array:
            data_time_serials = time_serials[accumulated_size: accumulated_size + size]
            accumulated_size += size
            for i in range(0, size - time_steps - target_skip_steps - 1, sample_interval):
                a_sample = data_time_serials[i:i + time_steps]
                input_time_serials.append(a_sample)
                if target_sequence:
                    target_time_serials.append(
                        data_time_serials[i + target_skip_steps + 1: i + target_skip_steps + time_steps + 1])
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

    def collect(self, file_name, feature_names=None):
        pass

    def sample_from_np(self):
        pass

    def get_scalers(self, file_names=None, feature_names=None):
        if not feature_names or len(feature_names) == 0:
            feature_names = self.get_all_features()
        data_time_serials, time_step_size_array = self._concatenate_data(file_names=file_names, feature_names=feature_names)
        scalers = []
        for i in range(len(feature_names)):
            train_time_serial = data_time_serials[:, i].reshape([-1, 1])
            scaler = MinMaxScaler()
            scaler.fit(train_time_serial)
            scalers.append(scaler)
        return scalers

    def sample(self, feature_names=None, time_steps=100, sample_interval=5, target_sequence=False, target_skip_steps=0,
               split_test_from_train=False, test_files=None):

        if not feature_names or len(feature_names) == 0:
            feature_names = self.get_all_features()

        train_files = self.files.keys()

        if not test_files or len(test_files) == 0:
            test_file_name = ''
            test_file_size = sys.maxsize
            for file_name in self.files.keys():
                file_size = os.path.getsize(os.path.join(ROOT_DIR, "../data", file_name))
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

        if split_test_from_train:
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


        train_input_time_serials, train_target_time_serials = self._de_concatenate_data(
            normalized_train_data_time_serials,
            train_time_step_size_array,
            time_steps=time_steps,
            sample_interval=sample_interval,
            target_sequence=target_sequence,
            target_skip_steps=target_skip_steps)

        test_input_time_serials, test_target_time_serials = self._de_concatenate_data(
            normalized_test_data_time_serials,
            test_time_step_size_array,
            time_steps=time_steps,
            sample_interval=1,
            target_sequence=target_sequence,
            target_skip_steps=target_skip_steps)

        train_input_time_serials, train_target_time_serials = shuffle(train_input_time_serials, train_target_time_serials)
        return train_input_time_serials, train_target_time_serials, test_input_time_serials, test_target_time_serials, feature_names, normalized_test_data_time_serials

    def draw_one_path(self, file_name, feature_name):
        file = self.files[file_name]
        data_time_serial = np.array(file[feature_name])
        print(data_time_serial.shape)
        plt.plot(data_time_serial[:, 1], data_time_serial[:, 0])
        plt.show()
        return

    def get_all_features(self, size=10):
        feature_names_set = set()

        for file_name, file in self.files.items():
            feature_names_in_one_set = set()
            for dset in traverse_datasets(file):
                data = file[dset]
                if len(data.shape) < 2:
                    continue
                var = np.var(data[:, 0], axis=-1)
                # if dset == "/CAN/EngineSpeed_CAN":
                #     print(dset)
                #     print(file_name)
                #     print(data.shape)
                #     print(var)
                if var > 1:
                    feature_names_in_one_set.add(dset)

            if len(feature_names_set) == 0:
                feature_names_set = feature_names_in_one_set
            else:
                feature_names_set = feature_names_set.intersection(feature_names_in_one_set)

        feature_names = shuffle(list(feature_names_set))

        return feature_names[:size]


if __name__ == '__main__':
    data_reader = DatasetReader(["20181113_Driver1_Trip1.hdf", "20181117_Driver1_Trip7.hdf"])
    #  /CAN/BoostPressure /CAN/AccPedal /CAN/EngineSpeed_CAN

    # print(data_reader.get_all_features())
    # exit(0)
    #
    # data_reader.print_structure(["20181203_Driver1_Trip10.hdf"])
    # exit(0)
    data_reader.print_structure(["20181113_Driver1_Trip1.hdf"])
    exit(0)
    data_reader.draw_one_path("20181117_Driver1_Trip7.hdf", "/CAN/SCS_Tip_Restart")
    exit(0)
    data_reader.draw_one_path("20181113_Driver1_Trip1.hdf", "/Plugins/Velocity_X")
    data_reader.draw_one_path("20181113_Driver1_Trip1.hdf", "CAN/EngineSpeed_CAN")
    data_reader.draw_one_path("20181113_Driver1_Trip1.hdf", "CAN/VehicleSpeed")
