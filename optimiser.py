import torch


class CustomOptimiser:
    """Based on https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
    Warm-up learning rate for first 10% of iterations and then decay
    """

    def __init__(self, optimiser, target_lr, num_steps, warmup_rate):
        self.optimiser = optimiser
        self._step = 0
        self.warmup_steps = int(num_steps * warmup_rate)
        self.target_lr = target_lr
        self._rate = 0
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=num_steps)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimiser'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

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
        else:
            return self.cosine_scheduler.get_last_lr()[0]

    def zero_grad(self):
        self.optimiser.zero_grad()
