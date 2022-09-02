
class HParams:

    num_mels = 80

    # mel-spec
    n_fft = 800
    sample_rate = 16000
    win_size = 800
    hop_size = 200

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

    pre_emphasize = True
    pre_emphasis = 0.97
