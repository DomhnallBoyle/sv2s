import os
import random

import numpy as np
import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, sample_pool_location):
        self.sample_pool_location = sample_pool_location
        # self.sample_pool_size = len(self.samples())
        self.sample_pool_size = 32
        self._samples = ['grid_sample.npz' for _ in range(self.sample_pool_size)]

    def samples(self):
        # return [f'{r}/{f}' for r, ds, fs in os.walk(self.sample_pool_location)
        #         for f in fs if f[-4:] == '.npz']
        return self._samples

    def __len__(self):
        return self.sample_pool_size

    def __getitem__(self, idx):  # random index
        while True:
            sample_path = self.samples()[idx]
            try:
                _, frames, mel_spec, speaker_embedding, _ = np.load(str(sample_path), allow_pickle=True)['sample']
                break
            except Exception as e:
                print('Failed to load sample:', e)
                idx = random.randint(0, self.sample_pool_size - 1)  # inclusive

        # TODO: remove this, converts 25 - 20 FPS
        frames = np.asarray([frames[int(i - 1)] for i in np.linspace(1, 25, num=20)])

        return (frames, speaker_embedding), mel_spec
