import random

from torch.utils.data import Sampler


class BucketSampler(Sampler):
    """
    Based on https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/utils/data/text/dataset.py
    """

    def __init__(self, dataset, batch_size, force_batch_size=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.force_batch_size = force_batch_size
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.num_batches = 0
        self.batch_map = self._generate_batch_map()
        self._generate_batch_list_wrapper()

    def _generate_batch_map(self):
        batch_map = {}  # length, indices lookup
        for i, sample in enumerate(self.dataset.samples):
            length = self.dataset.lengths[sample]

            index = [i, None]
            if self.dataset.slicing and self.dataset.get_duration(sample) > self.dataset.max_sample_duration:
                length = self.dataset.get_random_duration_num_frames()
                index[1] = length

            indices = batch_map.get(length, [])
            indices.append(index)
            batch_map[length] = indices
        
        return batch_map

    def _generate_batch_list_wrapper(self):
        batch_list = self._generate_batch_list()

        if self.force_batch_size:
            batch_list = [b for b in batch_list if len(b) == self.batch_size]
        random.shuffle(batch_list)  # aren't ordered by bucket size
        self.num_batches = len(batch_list)

        return batch_list

    def _generate_batch_list(self):
        batch_map = self.batch_map.copy()

        for indices in batch_map.values(): 
            random.shuffle(indices)  # indices put into different batches

        min_key = min(batch_map.items(), key=lambda x: len(x[1]))[0]
        max_key = max(batch_map.items(), key=lambda x: len(x[1]))[0]

        print('Min samples per length:', min_key, len(batch_map[min_key]))
        print('Max samples per length:', max_key, len(batch_map[max_key]))

        return [indices[i: (i + self.batch_size)]
                for indices in batch_map.values()
                for i in range(0, len(indices), self.batch_size)]

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # yields a batch of indices every iteration
        batch_list = self._generate_batch_list_wrapper()  # this part runs once at start

        for batch in batch_list:
            if self.force_batch_size:
                assert len(batch) == self.batch_size
            yield batch


class BalancedBatchSampler(BucketSampler): 
    """
    Balanced batches based on unique users and phrases
    """

    def __init__(self, dataset, batch_size, force_batch_size=False): 
        super().__init__(dataset, batch_size, force_batch_size, max_attempts=100)

    def _generate_batch_list(self):
        batch_map = self.batch_map.copy()

        batch_list = []
        for indices in batch_map.values():  # ensuring samples in batch have same length
            samples = [self.dataset.samples[i[0]] for i in indices]
            video_path_pool = [self.dataset.video_paths[s] for s in samples]
            indices_d = {video_path: i for video_path, i in zip(video_path_pool, indices)}
            while not len(video_path_pool) < self.batch_size:
                batch = []
                num_attempts = 0
                while len(batch) < self.batch_size and num_attempts < self.max_attempts:
                    video_path = random.choice(video_path_pool)
                    user, phrase = self.dataset.get_user(video_path), self.dataset.get_phrase(video_path)
                    if all([u != user and p != phrase for u, p, _ in batch]):  # ensure unique users/phrases in batch
                        batch.append([user, phrase, indices_d[video_path]])
                        video_path_pool = list(set(video_path_pool) - {video_path})
                    num_attempts += 1
                if len(batch) == self.batch_size:
                    batch_list.append(batch)

        for i, batch in enumerate(batch_list):
            _, _, indices = zip(*batch)
            video_paths = [self.dataset.video_paths[self.dataset.samples[j[0]]] for j in indices]
            users = [self.dataset.get_user(vp) for vp in video_paths]
            phrases = [self.dataset.get_phrase(vp) for vp in video_paths]
            assert len(users) == len(set(users))  # ensure no duplicate users or phrases
            assert len(phrases) == len(set(phrases))
            batch_list[i] = indices

        return batch_list
