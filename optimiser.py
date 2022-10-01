import torch


class CustomOptimiser:
    """Based on https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
    Warm-up learning rate for first 10% of iterations and then decay
    """

    def __init__(self, params, target_lr, num_steps, warmup_rate, decay=False):
        self.params = params
        self._step = 0
        self.num_steps = num_steps
        self.warmup_steps = int(num_steps * warmup_rate)
        self.decay = decay
        self.target_lr = target_lr
        self._rate = 0
        self.optimiser = None
        self.cosine_scheduler = None

    def init(self):
        rate = self._rate if self._rate != 0 else self.target_lr
        self.optimiser = torch.optim.AdamW(self.params, lr=rate, betas=(0.9, 0.98), weight_decay=0.01)
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=self.num_steps)
        if self._rate != 0:
            for p in self.optimiser.param_groups:
                p['lr'] = self._rate
            self.cosine_scheduler._last_lr = [self._rate]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'params'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.optimiser._warned_capturable_if_run_uncaptured = False  # BUG: https://github.com/pytorch/pytorch/pull/80345

    def step(self):
        self.optimiser.step()
        self.cosine_scheduler.step()

        self._step += 1
        rate = self.rate()
        for p in self.optimiser.param_groups:
            p['lr'] = rate
        self._rate = rate

    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= self.warmup_steps:
            # https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
            return step * (self.target_lr / self.warmup_steps)
        elif not self.decay:
            return self.target_lr
        else:
            return self.cosine_scheduler.get_last_lr()[0]

    def zero_grad(self):
        self.optimiser.zero_grad()
