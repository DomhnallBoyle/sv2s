import random

from torch.utils.data import Sampler


# https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/utils/data/text/dataset.py
class BucketSampler(Sampler):

    def __init__(self, batch_size, lengths, force_batch_size=False):
        self.batch_size = batch_size
        self.lengths = lengths
        self.force_batch_size = force_batch_size
        self.num_batches = 0

    def _generate_batch_list(self):
        batch_map = {}  # length, indices lookup
        for i, length in enumerate(self.lengths):
            indices = batch_map.get(length, [])
            indices.append(i)
            random.shuffle(indices)  # indices put into different batches
            batch_map[length] = indices

        batch_list = [indices[i: (i + self.batch_size)]
                      for indices in batch_map.values()
                      for i in range(0, len(indices), self.batch_size)]
        random.shuffle(batch_list)  # aren't ordered by bucket size

        return batch_list

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # yields a batch of indices every iteration
        batch_list = self._generate_batch_list()  # this part runs once
        self.num_batches = len(batch_list)

        for batch in batch_list:
            if self.force_batch_size and len(batch) != self.batch_size:
                continue  # force batch to be correct size
            yield batch
