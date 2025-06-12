# -*- coding: utf-8 -*-
"""
EventDataset classes
"""

from torch.utils.data import Dataset
from os.path import join
import numpy as np
from utils.util import first_element_greater_than, last_element_less_than
import random
import torch
import glob


class EventDataset(Dataset):
    def __init__(self, base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True):
        self.base_folder = base_folder
        self.event_folder = join(self.base_folder, event_folder)
        self.transform = transform
        self.start_time = start_time
        self.stop_time = stop_time
        self.normalize = normalize

        self.use_mvsec = True
        self.read_timestamps()

        self.parse_event_folder()

    def read_timestamps(self):
        # Load the timestamps file
        raw_stamps = np.loadtxt(join(self.event_folder, 'timestamps.txt'))
        if raw_stamps.size == 0:
            raise IOError('Dataset is empty')

        if len(raw_stamps.shape) == 1:
            raw_stamps = raw_stamps.reshape((1, 2))

        self.stamps = raw_stamps[:, 1]
        if self.stamps is None:
            raise IOError('Unable to read timestamp file: '.format(join(self.event_folder, 'timestamps.txt')))
        # Check that the timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)), "timestamps are not unique and monotonically increasing"

        self.initial_stamp = self.stamps[0]
        self.stamps = self.stamps - self.initial_stamp  # offset the timestamps so they start at 0
        if self.start_time <= 0.0:
            self.first_valid_idx, self.first_stamp = 0, self.stamps[0]
        else:
            self.first_valid_idx, self.first_stamp = first_element_greater_than(self.stamps, self.start_time)
        assert(self.first_stamp is not None)
        # Find the index of the last event tensor whose timestamp <= end_time
        # If there is None, throw an error
        if self.stop_time <= 0.0:
            self.last_valid_idx, self.last_stamp = len(self.stamps) - 1, self.stamps[-1]
        else:
            self.last_valid_idx, self.last_stamp = last_element_less_than(self.stamps, self.stop_time)
        assert(self.last_stamp is not None)


        assert(self.first_stamp <= self.last_stamp)
        self.length = self.last_valid_idx - self.first_valid_idx + 1
        assert(self.length > 0)

    def parse_event_folder(self):
        raise NotImplementedError

    def __len__(self):
        return self.length

    def get_last_stamp(self):

        return self.stamps[self.last_valid_idx]

    def num_channels(self):
        raise NotImplementedError

    def get_index_at(self, i):
        return self.first_valid_idx + i

    def get_stamp_at(self, i):
        return self.stamps[self.get_index_at(i)]

    def __getitem__(self, i):
        raise NotImplementedError


class VoxelGridDataset(EventDataset):

    def parse_event_folder(self):

        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        # event_tensor will be a [num_bins x H x W] floating point array
        if self.use_mvsec:
            event_tensor = np.load(join(self.event_folder, 'event_tensor_{:010d}.npy'.format(self.first_valid_idx + i)))
        else:
            path_event = glob.glob(self.event_folder + '/*_{:04d}_voxel.npy'.format(self.first_valid_idx + i))
            event_tensor = np.load(path_event[0])
        if self.normalize:
            # normalize the event tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
            # in the tensor are equal to (0.0, 1.0)
            mask = np.nonzero(event_tensor)
            if mask[0].size > 0:
                mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
                if stddev > 0:
                    event_tensor[mask] = (event_tensor[mask] - mean) / stddev

        self.num_bins = event_tensor.shape[0]

        events = torch.from_numpy(event_tensor)# [C x H x W]
        if self.transform:
            random.seed(transform_seed)
            events = self.transform(events)

        return {'events': events}  # [num_bins x H x W] tensor


