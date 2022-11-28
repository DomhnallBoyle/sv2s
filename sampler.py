import random

from torch.utils.data import Sampler


class BucketSampler(Sampler):
    """
    Based on https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/utils/data/text/dataset.py
    """

    def __init__(self, dataset, batch_size, force_batch_size=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.force_batch_size = force_batch_size
        self.num_batches = 0
        self._generate_batch_list()

    def _generate_batch_list(self):
        batch_map = {}  # length, indices lookup
        for i, sample in enumerate(self.dataset.samples):
            length = self.dataset.lengths[sample]

            if self.dataset.slicing:
                if self.dataset.get_duration(sample) > self.dataset.max_sample_duration:
                    length = self.dataset.get_random_duration_num_frames()
                i = [i, length]

            indices = batch_map.get(length, [])
            indices.append(i)
            batch_map[length] = indices

        for length, indices in batch_map.items(): 
            random.shuffle(indices)  # indices put into different batches

        min_key = min(batch_map.items(), key=lambda x: len(x[1]))[0]
        max_key = max(batch_map.items(), key=lambda x: len(x[1]))[0]

        print('Min samples per length:', min_key, len(batch_map[min_key]))
        print('Max samples per length:', max_key, len(batch_map[max_key]))

        batch_list = [indices[i: (i + self.batch_size)]
                      for indices in batch_map.values()
                      for i in range(0, len(indices), self.batch_size)]

        if self.force_batch_size:
            batch_list = [b for b in batch_list if len(b) == self.batch_size]

        random.shuffle(batch_list)  # aren't ordered by bucket size

        self.num_batches = len(batch_list)

        return batch_list

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # yields a batch of indices every iteration
        batch_list = self._generate_batch_list()  # this part runs once

        for batch in batch_list:
            if self.force_batch_size:
                assert len(batch) == self.batch_size
            yield batch
