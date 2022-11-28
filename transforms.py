import math
import random

import numpy as np

from hparams import hparams


class HorizontalFlipping:

    def __init__(self, p): 
        self.p = p

    def __call__(self): 
        if random.random() < self.p:
            frames = frames[..., :, ::-1]  # flip along vertical axis

        return frames


class IntensityAugmentation:
    
    def __init__(self, p):
        self.p = p

    def __call__(self, frames): 
        if random.random() < self.p: 
            intensity = np.random.randint(-30, 30)  # inclusive
            frames += intensity
            frames = np.where(frames > 255, 255, frames)
            frames = np.where(frames < 0, 0, frames)  # capping frame values

        return frames


class RandomCrop:

    def __init__(self, size):
        self.height, self.width = size

    def __call__(self, frames):
        t, h, w = frames.shape
        delta_w = random.randint(0, w - self.width)
        delta_h = random.randint(0, h - self.height)

        return frames[:, delta_h:delta_h + self.height, delta_w:delta_w + self.width]


class TimeMasking:
    
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, frames, mean_time_mask_value): 
        num_masks = len(frames) // hparams.fps  # 1 mask per second
        for j in range(num_masks):
            mask_duration_secs = np.random.uniform(0, 0.4)  # a mask can have a max of 8 frames
            mask_duration_frames = math.ceil(hparams.fps * mask_duration_secs)  # choose num frames to mask
            frame_index = random.randint(j * hparams.fps, ((j * hparams.fps) + hparams.fps) - mask_duration_frames)  # choose start index of mask
            frames[frame_index:frame_index + mask_duration_frames, :, :] = mean_time_mask_value
            if self.debug:
                print(f'Mask {j + 1}:', mask_duration_secs, mask_duration_frames, frame_index, frame_index + mask_duration_frames, len(frames))

        return frames

    
class Slicing: 
    
    def __init__(self):
        pass

    def __call__(self, frames, mel_spec, duration_num_frames): 
        duration = duration_num_frames / hparams.fps

        start_index = random.randint(0, (len(frames) - duration_num_frames) - 1)  # inclusive
        end_index = start_index + duration_num_frames
        assert end_index < len(frames)

        mel_start_index = int(len(mel_spec) * start_index / len(frames))
        mel_end_index = mel_start_index + (duration_num_frames * 4)

        frames = frames[start_index:end_index, ...]
        assert len(frames) / hparams.fps == duration

        mel_spec = mel_spec[mel_start_index:mel_end_index, ...]
        assert len(frames) * 4 == len(mel_spec)

        return frames, mel_spec
