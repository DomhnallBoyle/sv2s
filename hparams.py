class HParams:

    num_mels = 80

    # mel-spec
    n_fft = 800
    sample_rate = 16000
    win_size = 800
    hop_size = 200
    # n_fft = 2048
    # sample_rate = 24000
    # win_size = 1200
    # hop_size = 300

    # mel-spec norm and clipping
    signal_normalization = True
    allow_clipping_in_normalization = True
    max_abs_value = 4.0
    symmetric_mels = True

    # limits
    min_level_db = -100
    ref_level_db = 20
    fmin = 55
    fmax = 7600

    # griffin-lim
    power = 1.5
    griffin_lim_iters = 60
    # griffin_lim_iters = 30  # doesn't seem to impact that much between 30 and 60

    pre_emphasize = True
    pre_emphasis = 0.97

    height = 88
    width = 88
    fps = 20
    conformer_params = {
        's': {
            'blocks': 6,
            'att_dim': 256,
            'att_heads': 4,
            'conv_k': 31,
            'ff_dim': 2048,
            'total_params': 27.3
        },
        'm': {
            'blocks': 12,
            'att_dim': 256,
            'att_heads': 4,
            'conv_k': 31,
            'ff_dim': 2048,
            'total_params': 43.1
        },
        'l': {
            'blocks': 12,
            'att_dim': 512,
            'att_heads': 8,
            'conv_k': 31,
            'ff_dim': 2048,
            'total_params': 87.6
        }
    }
    seed = 1234
    pad_value = 0.0

    @property
    def __dict__(self): 
        return {a: getattr(self, a) for a in dir(self) if a[0] != '_'}


hparams = HParams()
