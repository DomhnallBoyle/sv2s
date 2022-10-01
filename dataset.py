import argparse
import math
import os
import random

random.seed(1234)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision.transforms import CenterCrop, RandomErasing, Normalize
from tqdm import tqdm

from utils import plot_spectrogram

FPS = 20
HEIGHT = 88
WIDTH = 88
PAD_VALUE = 0.0


class CustomCollate:

    def __init__(self, last_frame_padding=False):
        self.last_frame_padding = last_frame_padding

    def __call__(self, batch):
        video_paths = [x[0] for x in batch]
        windows = [torch.Tensor(x[1]) for x in batch]
        speaker_embeddings = torch.from_numpy(np.asarray([x[2] for x in batch]))
        gt_mel_specs = [torch.Tensor(x[3]) for x in batch]
        lengths = torch.Tensor([x[1].shape[0] for x in batch])
        target_lengths = torch.Tensor([x.shape[0] for x in gt_mel_specs])

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
                new_gt_mel_spec = torch.zeros((max_audio_length, 80))
                new_gt_mel_spec[:mel_spec_length, :] = gt_mel_spec
                new_gt_mel_spec[mel_spec_length:max_audio_length, :] = gt_mel_spec.min()  # silence

                new_windows.append(new_window)
                new_gt_mel_specs.append(new_gt_mel_spec)
            windows = torch.stack(new_windows)
            gt_mel_specs = torch.stack(new_gt_mel_specs)
            lengths = torch.Tensor([x.shape[0] for x in windows])
            target_lengths = torch.Tensor([x.shape[0] for x in gt_mel_specs])

        for window, length, gt_mel_spec, target_length in zip(windows, lengths, gt_mel_specs, target_lengths):
            assert window.shape[0] == int(length)
            assert (window.shape[0] / FPS) == (gt_mel_spec.shape[0] / 80)
            assert gt_mel_spec.shape[0] == target_length
            # assert PAD_VALUE not in gt_mel_spec

        # # calculate sorted indices of seq length in desc order
        # index_lengths = [[i, w.shape[0]] for i, w in enumerate(windows)]
        # index_lengths_sorted = sorted(index_lengths, key=lambda x: x[1], reverse=True)  # sort based on sequence length desc
        # sorted_indices = [x[0] for x in index_lengths_sorted]

        # pad sequences
        windows = pad_sequence(windows, batch_first=True, padding_value=PAD_VALUE)
        gt_mel_specs = pad_sequence(gt_mel_specs, batch_first=True, padding_value=PAD_VALUE)

        # # sort data
        # video_paths = [video_paths[i] for i in sorted_indices]
        # windows = windows[sorted_indices]
        # speaker_embeddings = speaker_embeddings[sorted_indices]
        # gt_mel_specs = gt_mel_specs[sorted_indices]
        # lengths = lengths[sorted_indices]

        # ([B, 1, T, H, W], [B, 256, 1])
        windows = windows.unsqueeze(1)
        speaker_embeddings = speaker_embeddings.unsqueeze(-1)

        return video_paths, (windows, speaker_embeddings, lengths), gt_mel_specs, target_lengths


class RandomCrop:

    def __init__(self, size):
        self.height, self.width = size

    def __call__(self, frames):
        t, h, w = frames.shape
        delta_w = random.randint(0, w - self.width)
        delta_h = random.randint(0, h - self.height)

        return frames[:, delta_h:delta_h + self.height, delta_w:delta_w + self.width]


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, location, horizontal_flipping=False, intensity_augmentation=False, time_masking=False,
                 erasing=False, random_cropping=False, wait_ms=FPS, num_samples=None, debug=False, **kwargs):
        self.location = location
        self.augmentation_prob = 0.5
        self.horizontal_flipping = horizontal_flipping
        self.intensity_augmentation = intensity_augmentation
        self.time_masking = time_masking
        self.erasing = RandomErasing(p=self.augmentation_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3)) if erasing else None
        self.cropping = RandomCrop(size=(HEIGHT, WIDTH)) if random_cropping else CenterCrop(size=(HEIGHT, WIDTH))
        self.wait_ms = wait_ms
        self.num_samples = num_samples
        self.debug = debug
        self.samples = self.get_samples()

        # https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/14b33789c40f931860994eafdef18d05136ef8ef/dataloader/dataloader.py#L86
        self.normalise_1 = Normalize(mean=0.0, std=255.0)
        self.normalise_2 = Normalize(mean=0.421, std=0.165)

    def get_samples(self):
        samples = [f'{r}/{f}' for r, ds, fs in os.walk(self.location) for f in fs if f[-4:] == '.npz']
        if self.num_samples is not None:
            random.shuffle(samples)
            samples = samples[:self.num_samples]

        return samples

    def get_lengths(self):
        lengths = []
        print('Getting dataset lengths...')
        for sample_path in tqdm(self.samples):
            _, frames, _, _ = np.load(str(sample_path), allow_pickle=True)['sample']
            lengths.append(frames.shape[0])

        return lengths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # random index
        while True:
            sample_path = self.samples[idx]
            try:
                video_path, frames, speaker_embedding, mel_spec = np.load(str(sample_path), allow_pickle=True)['sample']
                break
            except Exception as e:
                print('Failed to load sample:', e, sample_path)
                self.samples = self.get_samples()
                idx = random.randint(0, len(self) - 1)  # inclusive

        if self.debug:
            for frame in frames:
                cv2.imshow('Before:', frame / 255.)
                cv2.waitKey(self.wait_ms)

        if self.time_masking:
            mean_frame = np.mean(frames, axis=0)
            num_masks = len(frames) // FPS
            for j in range(num_masks):
                mask_duration_secs = np.random.uniform(0, 0.4)
                mask_duration_frames = math.ceil(FPS * mask_duration_secs)  # choose num frames to mask
                frame_index = random.randint(j * FPS, ((j * FPS) + FPS) - mask_duration_frames)  # choose start index of mask
                frames[frame_index:frame_index + mask_duration_frames, :, :] = mean_frame
                if self.debug:
                    print(f'Mask {j + 1}:', mask_duration_secs, mask_duration_frames, frame_index, frame_index + mask_duration_frames, len(frames))

        apply_flipping = self.horizontal_flipping and random.random() < self.augmentation_prob
        apply_intensity_augmentation = self.intensity_augmentation and random.random() < self.augmentation_prob

        if apply_flipping:
            frames = frames[..., :, ::-1]  # flip along vertical axis
        
        if apply_intensity_augmentation: 
            intensity = np.random.randint(-30, 30)  # inclusive
            frames += intensity
            frames = np.where(frames > 255, 255, frames)
            frames = np.where(frames < 0, 0, frames)  # capping frame values

        # perform random cropping or centre crop to 88x88
        frames = self.cropping(torch.from_numpy(frames.copy())).numpy()

        if self.erasing:
            frames = self.erasing(torch.from_numpy(frames.copy())).numpy()

        # TODO: this altogether gives range -2 -> 2, ask about this
        # frames /= 255.  # min-max normalisation between 0-1
        # frames = self.normalise(torch.from_numpy(frames.copy())).numpy()
        # print(frames.max(), frames.min())

        # frames = self.normalise_1(torch.from_numpy(frames.copy()))
        # frames = self.normalise_2(frames).numpy()
        # frames = self.normalise_2(torch.from_numpy(frames.copy())).numpy()
        # print(frames.max(), frames.min())

        # normalise frames
        frames = self.normalise_2(self.normalise_1(torch.from_numpy(frames.copy()))).numpy()

        if self.debug:
            for frame in frames:
                cv2.imshow('After:', frame)
                cv2.waitKey(self.wait_ms)

        return video_path, frames, speaker_embedding, mel_spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('location')
    parser.add_argument('--horizontal_flipping', action='store_true')
    parser.add_argument('--intensity_augmentation', action='store_true')
    parser.add_argument('--time_masking', action='store_true')
    parser.add_argument('--erasing', action='store_true')
    parser.add_argument('--random_cropping', action='store_true')
    parser.add_argument('--last_frame_padding', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wait_ms', type=int, default=FPS)

    args = parser.parse_args()

    dataset = CustomDataset(**args.__dict__)
    collator = CustomCollate(last_frame_padding=args.last_frame_padding)
    batch = []
    for i in range(len(dataset)):
        video_path, frames, speaker_embedding, mel_spec = dataset[i]
        if args.debug:
            print(i, video_path, frames.shape, speaker_embedding.shape, mel_spec.shape)
        batch.append([video_path, frames, speaker_embedding, mel_spec])
        if len(batch) == 5:
            video_paths, (windows, speaker_embeddings, lengths), gt_mel_specs, target_lengths = collator(batch)
            for video_path, window, speaker_embedding, length, gt_mel_spec, target_length in \
                    zip(video_paths, windows, speaker_embeddings, lengths, gt_mel_specs, target_lengths):
                print(video_path, window.shape, speaker_embedding.shape, length, gt_mel_spec.shape, target_length)
                for frame in window[0]:
                    frame = cv2.normalize(frame.numpy(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow('Frame', frame)
                    cv2.waitKey(args.wait_ms)
                gt_mel_spec = gt_mel_spec.numpy()
                fig = plot_spectrogram(gt_mel_spec)
                plt.show()
            batch = []

    # show frequency of frame length across dataset
    data = np.asarray(dataset.get_lengths())
    plt.hist(data, bins=np.arange(data.min(), data.max() + 1))
    plt.show()
