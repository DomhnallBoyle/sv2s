import torch


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = WeightedL1Loss()
        self.spectral_convergence_loss = WeightedSpectralConvergenceLoss()

        # self.l1_loss = torch.nn.L1Loss()
        # self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, gts, preds, weights):
        return self.l1_loss(gts, preds, weights) + self.spectral_convergence_loss(gts, preds, weights)

    # def forward(self, gts, preds, weights):
    #     return self.l1_loss(gts, preds) + self.spectral_convergence_loss(gts, preds)


class SpectralConvergenceLoss(torch.nn.Module):
    # https://github.com/kan-bayashi/ParallelWaveGAN/blob/1f7949f593cc5600478cd4cc23bbf34bbcb0bcff/parallel_wavegan/losses/stft_loss.py#L43

    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        return torch.norm(pred - gt, p='fro') / torch.norm(pred, p='fro')


class WeightedL1Loss(torch.nn.Module):
    # weigh loss computed for different samples based on whether they belong to the majority/minority class
    # i.e. assign a higher weight to the loss encountered by samples in the minor class

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, gts, preds, weights):
        batch_loss = 0
        for gt, pred, weight in zip(gts, preds, weights):
            batch_loss += (self.l1_loss(gt, pred) * weight)  # higher weight = minority = high error

        return batch_loss / len(gts)


class WeightedSpectralConvergenceLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm_f = torch.linalg.norm

    def forward(self, gts, preds, weights):
        batch_loss = 0
        for gt, pred, weight in zip(gts, preds, weights):
            loss = (self.norm_f(pred - gt, ord='fro') / self.norm_f(pred, ord='fro')) * weight
            batch_loss += loss

        return batch_loss / len(gts)
