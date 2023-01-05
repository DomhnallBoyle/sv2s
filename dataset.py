import argparse
import math
import os
import pickle
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from playsound import playsound
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import CenterCrop, RandomErasing, Normalize
from tqdm import tqdm

from hparams import hparams
from transform import HorizontalFlipping, IntensityAugmentation, RandomCrop, Slicing, TimeMasking
from utils import plot_spectrogram, save_wav
from vocoder import griffin_lim


class CustomCollate:

    def __init__(self, last_frame_padding=False, skip_assert=False):
        self.last_frame_padding = last_frame_padding
        self.skip_assert = skip_assert

    def __call__(self, batch):
        video_paths = [x[0] for x in batch]
        windows = [torch.Tensor(x[1]) for x in batch]
        speaker_embeddings = torch.from_numpy(np.asarray([x[2] for x in batch]))
        gt_mel_specs = [torch.Tensor(x[3]) for x in batch]
        weights = torch.Tensor([x[4] for x in batch])
        lengths = torch.Tensor([x[1].shape[0] for x in batch]).int()
        target_lengths = torch.Tensor([x.shape[0] for x in gt_mel_specs]).int()

        # padding with the last frame from the window
        if self.last_frame_padding:
            max_length = int(lengths.max())
            max_audio_length = max_length * 4
            new_windows, new_gt_mel_specs = [], []
            for window, gt_mel_spec in zip(windows, gt_mel_specs):
                window_length = window.shape[0]
                if window_length == max_length:
                    new_windows.append(window)
                    new_gt_mel_specs.append(gt_mel_spec)
                    continue
                last_frame = window[-1]
                new_window = torch.zeros((max_length, *window.shape[1:3]))
                new_window[:window_length, :, :] = window
                new_window[window_length:max_length, :, :] = last_frame  # finished talking - silence

                mel_spec_length = gt_mel_spec.shape[0]
                new_gt_mel_spec = torch.zeros((max_audio_length, hparams.num_mels))
                new_gt_mel_spec[:mel_spec_length, :] = gt_mel_spec
                new_gt_mel_spec[mel_spec_length:max_audio_length, :] = gt_mel_spec.min()  # silence

                new_windows.append(new_window)
                new_gt_mel_specs.append(new_gt_mel_spec)
            windows = torch.stack(new_windows)
            gt_mel_specs = torch.stack(new_gt_mel_specs)
            lengths = torch.Tensor([x.shape[0] for x in windows]).int()
            target_lengths = torch.Tensor([x.shape[0] for x in gt_mel_specs]).int()

        if not self.skip_assert:
            for window, length, gt_mel_spec, target_length in zip(windows, lengths, gt_mel_specs, target_lengths):
                assert window.shape[0] == int(length)
                assert (window.shape[0] / hparams.fps) == (gt_mel_spec.shape[0] / hparams.num_mels)
                assert gt_mel_spec.shape[0] == target_length
                # assert PAD_VALUE not in gt_mel_spec

        # # calculate sorted indices of seq length in desc order
        # index_lengths = [[i, w.shape[0]] for i, w in enumerate(windows)]
        # index_lengths_sorted = sorted(index_lengths, key=lambda x: x[1], reverse=True)  # sort based on sequence length desc
        # sorted_indices = [x[0] for x in index_lengths_sorted]

        # pad sequences
        windows = pad_sequence(windows, batch_first=True, padding_value=hparams.pad_value)
        gt_mel_specs = pad_sequence(gt_mel_specs, batch_first=True, padding_value=hparams.pad_value)

        # # sort data
        # video_paths = [video_paths[i] for i in sorted_indices]
        # windows = windows[sorted_indices]
        # speaker_embeddings = speaker_embeddings[sorted_indices]
        # gt_mel_specs = gt_mel_specs[sorted_indices]
        # lengths = lengths[sorted_indices]

        # ([B, 1, T, H, W], [B, 256, 1])
        windows = windows.unsqueeze(1)
        speaker_embeddings = speaker_embeddings.unsqueeze(-1)

        return video_paths, (windows, speaker_embeddings, lengths), gt_mel_specs, target_lengths, weights


class CustomDataset(torch.utils.data.Dataset):

    # TODO: default min sample duration? e.g. 1 or 2 seconds
    #  changes to get_random_duration_num_frames()
    def __init__(self, location, horizontal_flipping=False, intensity_augmentation=False, time_masking=False,
                 erasing=False, random_cropping=False, wait_ms=hparams.fps, num_samples=None, use_class_weights=False,
                 min_sample_duration=None, max_sample_duration=None, slicing=False, time_mask_by_frame=False, debug=False, 
                 **kwargs):
        self.location = Path(location)
        assert self.location.exists()
        self.augmentation_prob = 0.5
        self.horizontal_flipping = HorizontalFlipping(p=self.augmentation_prob) if horizontal_flipping else None
        self.intensity_augmentation = IntensityAugmentation(p=self.augmentation_prob) if intensity_augmentation else None
        self.time_masking = TimeMasking(debug=debug) if time_masking else None
        self.erasing = RandomErasing(p=self.augmentation_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3)) if erasing else None
        self.cropping = RandomCrop(size=(hparams.height, hparams.width)) if random_cropping else CenterCrop(size=(hparams.height, hparams.width))
        self.slicing = Slicing() if slicing else None
        self.wait_ms = wait_ms
        self.num_samples = num_samples
        self.use_class_weights = use_class_weights
        self.min_sample_duration = min_sample_duration
        self.max_sample_duration = max_sample_duration
        self.time_masking_axis = 0 if time_mask_by_frame else None  # can be frame (0 = vsrml) or pixel value (None = sv2s)
        self.debug = debug

        self.lengths_location = self.location.joinpath('lengths.pkl')
        self.video_paths_location = self.location.joinpath('video_paths.pkl')
        self.sample_paths = [f'{r}/{f}' for r, ds, fs in os.walk(self.location) for f in fs if f[-4:] == '.npz']
        self.lengths = self.get_stats(location=self.lengths_location, stats_f=self.get_length)
        self.video_paths = self.get_stats(location=self.video_paths_location, stats_f=self.get_video_path)
        self.samples = self.get_samples()
        self.user_weights = self.get_class_weights(_key='get_user') if use_class_weights else None
        self.phrase_weights = self.get_class_weights(_key='get_phrase') if use_class_weights else None

        # https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/14b33789c40f931860994eafdef18d05136ef8ef/dataloader/dataloader.py#L86
        self.normalise_1 = Normalize(mean=0.0, std=255.0)
        self.normalise_2 = Normalize(mean=0.421, std=0.165)

        if self.slicing: 
            assert self.max_sample_duration is not None

    def get_samples(self):
        print(f'Getting samples from {self.location}...')
        samples = self.sample_paths.copy()

        if self.min_sample_duration:
            samples = [s for s in samples if self.get_duration(s) >= self.min_sample_duration]

        if self.max_sample_duration and not self.slicing:
            samples = [s for s in samples if self.get_duration(s) <= self.max_sample_duration]

        if self.num_samples is not None:
            random.shuffle(samples)
            samples = samples[:self.num_samples]

        return samples

    def get_duration(self, sample_path): 
        return self.lengths[sample_path] / hparams.fps

    def get_random_duration_num_frames(self): 
        return random.randint(
            self.min_sample_duration * hparams.fps, 
            self.max_sample_duration * hparams.fps
        )  # inclusive 

    def get_length(self, sample_path):
        _, frames, _, _ = np.load(str(sample_path), allow_pickle=True)['sample']

        return frames.shape[0]

    def get_video_path(self, sample_path):
        return np.load(str(sample_path), allow_pickle=True)['sample'][0]

    def get_stats(self, location, stats_f): 
        print(f'Getting dataset stats for {stats_f.__name__}...')

        if location.exists():
            with location.open('rb') as f:
                return pickle.load(f)

        stats = {}
        for sample_path in tqdm(self.sample_paths):
            stats[sample_path] = stats_f(sample_path=sample_path)

        with location.open('wb') as f:
            pickle.dump(stats, f)

        return stats

    def get_user(self, video_path): 
        return Path(video_path).parts[-2]

    def get_phrase(self, video_path):
        return Path(video_path).stem.split('_')[0].lower()

    def get_class_weights(self, _key):
        """
        https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4

        Alternative:
            https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
            e.g. {phrase: 1 - (count / len(self)) for phrase, count in class_counts.items()}
        """
        print(f'Getting class weights for "{_key}"...')
        class_counts = {}
        for sample_path in tqdm(self.samples):
            video_path = self.get_video_path(sample_path)
            _class = getattr(self, _key)(video_path)
            class_counts[_class] = class_counts.get(_class, 0) + 1

        num_classes = len(class_counts)
        classes, num_samples_per_class = zip(*class_counts.items())
        weights = 1 / np.asarray(num_samples_per_class)
        weights = (weights / np.sum(weights) * num_classes).tolist()  # sum(weights) == no. of classes

        return {_class: weight for _class, weight in zip(classes, weights)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # random index
        if type(idx) is list:
            idx, length = idx  # from bucketing
        else:
            length = None

        sample_path = self.samples[idx]
        video_path, frames, speaker_embedding, mel_spec = np.load(str(sample_path), allow_pickle=True)['sample']
        
        if self.slicing and self.get_duration(sample_path) > self.max_sample_duration:
            # randomly select min duration frames <= x <= max duration frames and crop the mel-spec to size
            if length is None: 
                duration_num_frames = self.get_random_duration_num_frames()
            else:
                duration_num_frames = length
            
            frames, mel_spec = self.slicing(frames, mel_spec, duration_num_frames)

        frames = frames.astype(np.float32)

        if self.debug:
            for frame in frames:
                cv2.imshow('Before:', frame / 255.)
                cv2.waitKey(self.wait_ms)

        mean_time_mask_value = np.mean(frames, axis=self.time_masking_axis)  # can be frame (0 = vsrml) or pixel value (None = sv2s)

        if self.horizontal_flipping:
            frames = self.horizontal_flipping(frames)
        
        if self.intensity_augmentation: 
            frames = self.intensity_augmentation(frames)

        # perform random cropping or centre crop to 88x88
        frames = self.cropping(torch.from_numpy(frames.copy())).numpy()

        if self.erasing:
            frames = self.erasing(torch.from_numpy(frames.copy())).numpy()

        if self.time_masking:
            frames = self.time_masking(frames, mean_time_mask_value)
            
        # normalise frames
        frames = self.normalise_2(self.normalise_1(torch.from_numpy(frames.copy()))).numpy()

        # weight by phrase and user
        weight = self.user_weights[self.get_user(video_path)] + self.phrase_weights[self.get_phrase(video_path)] if self.use_class_weights else 1

        if self.debug:
            for frame in frames:
                cv2.imshow('After:', frame)
                cv2.waitKey(self.wait_ms)

        return video_path, frames, speaker_embedding, mel_spec, weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('location')
    parser.add_argument('--horizontal_flipping', action='store_true')
    parser.add_argument('--intensity_augmentation', action='store_true')
    parser.add_argument('--time_masking', action='store_true')
    parser.add_argument('--erasing', action='store_true')
    parser.add_argument('--random_cropping', action='store_true')
    parser.add_argument('--last_frame_padding', action='store_true')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--time_mask_by_frame', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wait_ms', type=int, default=hparams.fps)

    args = parser.parse_args()

    dataset = CustomDataset(**args.__dict__)
    collator = CustomCollate(last_frame_padding=args.last_frame_padding)

    batch = []
    for i in range(len(dataset)):
        video_path, frames, speaker_embedding, mel_spec, weight = dataset[i]
        if args.debug:
            print(i, video_path, frames.shape, speaker_embedding.shape, mel_spec.shape, weight)
        batch.append([video_path, frames, speaker_embedding, mel_spec, weight])
        if len(batch) == 5:
            video_paths, (windows, speaker_embeddings, lengths), gt_mel_specs, target_lengths, weights = collator(batch)
            for video_path, window, speaker_embedding, length, gt_mel_spec, target_length, weight in \
                    zip(video_paths, windows, speaker_embeddings, lengths, gt_mel_specs, target_lengths, weights):

                print(video_path, window.shape, speaker_embedding.shape, length, gt_mel_spec.shape, target_length, weight)
                print(window.max(), window.min())

                for frame in window[0]:
                    frame = cv2.normalize(frame.numpy(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow('Frame', frame)
                    cv2.waitKey(args.wait_ms)

                gt_mel_spec = gt_mel_spec.numpy()
                fig = plot_spectrogram(gt_mel_spec)
                plt.show()

                audio_path = '/tmp/audio.wav'
                save_wav(griffin_lim([gt_mel_spec])[0], save_path=audio_path, sr=hparams.sample_rate)
                playsound(audio_path, block=True)

            batch = []

    # show frequency of frame length across dataset
    data = np.asarray(dataset.get_lengths())
    plt.hist(data, bins=np.arange(data.min(), data.max() + 1))
    plt.show()
