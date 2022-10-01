import torch


class SpectralConvergenceLoss(torch.nn.Module):
    # https://github.com/kan-bayashi/ParallelWaveGAN/blob/1f7949f593cc5600478cd4cc23bbf34bbcb0bcff/parallel_wavegan/losses/stft_loss.py#L43

    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        return torch.norm(pred - gt, p='fro') / torch.norm(pred, p='fro')
