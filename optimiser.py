import matplotlib.pyplot as plt
import torch


class CustomOptimiser:
    """
    Based on https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
    Warm-up learning rate for first 10% of iterations and then decay
    """

    def __init__(self, params, target_lr, num_steps, warmup_rate, decay=False, num_restarts=None, num_first_restart_iters=None, restart_iteration_factor=1):
        self.params = params
        self.target_lr = target_lr
        self.num_steps = num_steps
        self.decay = decay
        self.num_restarts = num_restarts
        self.num_first_restart_iters = num_first_restart_iters 
        self.restart_iteration_factor = restart_iteration_factor
        self.step_counter = 0
        self.warmup_steps = int(num_steps * warmup_rate)
        self.rate = 0
        self.optimiser = None
        self.cosine_scheduler = None
        
        if self.num_restarts:
            self.num_first_restart_iters = int((self.num_steps - self.warmup_steps) / (self.num_restarts + 1))

    def init(self):
        rate = self.rate if self.rate != 0 else self.target_lr
        self.optimiser = torch.optim.AdamW(self.params, lr=rate, betas=(0.9, 0.98), weight_decay=0.01)

        # https://arxiv.org/pdf/1608.03983.pdf
        # the paper suggests that warm restarts leads to faster convergence with better results
        if self.num_first_restart_iters: 
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser, T_0=self.num_first_restart_iters, T_mult=self.restart_iteration_factor)
        else:
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=self.num_steps - self.warmup_steps)
        
        # continue training
        if self.rate != 0:
            for p in self.optimiser.param_groups:
                p['lr'] = self.rate
            self.cosine_scheduler._last_lr = [self.rate]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in ['params', 'decay']}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.optimiser._warned_capturable_if_run_uncaptured = False  # BUG: https://github.com/pytorch/pytorch/pull/80345

    def update_params(self, lr):
        for p in self.optimiser.param_groups:
            p['lr'] = lr

    def step(self):
        self.optimiser.step()

        if self.step_counter <= self.warmup_steps:
            # https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
            lr = self.step_counter * (self.target_lr / self.warmup_steps)
            self.update_params(lr=lr)
        elif not self.decay:
            lr = self.target_lr
            self.update_params(lr=lr)
        else:
            self.cosine_scheduler.step()  # this updates lr of params automatically
            lr = self.cosine_scheduler.get_last_lr()[0]

        self.step_counter += 1
        self.rate = lr

    def zero_grad(self):
        self.optimiser.zero_grad()

    def plot_lr_graph(self):
        x = list(range(self.num_steps))
        y = []
        while self.step_counter < self.num_steps: 
            self.step()
            y.append(self.rate)
        
        plt.plot(x, y)
        plt.show()
