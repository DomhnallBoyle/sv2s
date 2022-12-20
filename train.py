import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from jiwer import cer as calculate_cer, wer as calculate_wer
from pystoi import stoi
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from architecture import V2S
from asr import DeepSpeechASR
from dataset import CustomCollate, CustomDataset
from hparams import hparams
from loss import CustomLoss
from optimiser import CustomOptimiser
from sampler import BalancedBatchSampler, BucketSampler
from utils import log, norm_wav, plot_spectrogram, save_wav
from vocoder import griffin_lim, parallel_wavegan


def main(args):
    args.gradient_accumulation = args.gradient_accumulation_steps is not None

    # seed the randomisers
    torch.manual_seed(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)

    training_output_directory = Path(f'runs/{args.name}')
    checkpoint_directory = training_output_directory.joinpath('checkpoints')
    checkpoint_directory.mkdir(exist_ok=True, parents=True)
    utils.log_path = training_output_directory.joinpath(f'log_{args.run_index}.txt')
    with utils.log_path.open('w') as f:
        for arg in sys.argv:
            f.write(f'{arg} \\\n')
    log(f'\nHparams: {hparams.__dict__}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f'\nUsing: {device}')

    # create network
    net = V2S(
        conformer_type=args.conformer_size, 
        device=device, 
        pretrained_resnet_path=args.pretrained_resnet_path, 
        freeze_encoder=args.freeze_encoder, 
        group_norm=args.gradient_accumulation
    ).to(device)

    # create loss function
    criterion = CustomLoss()

    # create datasets
    train_dataset = CustomDataset(
        location=args.training_dataset_location,
        horizontal_flipping=args.horizontal_flipping,
        intensity_augmentation=args.intensity_augmentation,
        time_masking=args.time_masking,
        erasing=args.erasing,
        random_cropping=args.random_cropping,
        use_class_weights=args.use_class_weights,
        min_sample_duration=args.min_sample_duration,
        max_sample_duration=args.max_sample_duration,
        slicing=args.slicing,
        time_mask_by_frame=args.time_mask_by_frame,
        num_samples=args.num_dataset_samples
    )
    val_dataset = CustomDataset(
        location=args.val_dataset_location,
        min_sample_duration=args.min_sample_duration,
        max_sample_duration=args.max_sample_duration,
        num_samples=args.num_dataset_samples
    )

    # create custom collate fn
    collator = CustomCollate(last_frame_padding=args.last_frame_padding)

    # create data loaders
    if args.bucketing:
        # ORDER: (1) batch sampler, (2) dataset[i], (3) collate_fn
        if args.balanced_bucketing:
            sampler = BalancedBatchSampler(
                dataset=train_dataset,
                batch_size=args.batch_size,
                force_batch_size=args.force_train_batch_size
            )
        else:
            sampler = BucketSampler(
                dataset=train_dataset,
                batch_size=args.batch_size,
                force_batch_size=args.force_train_batch_size
            )
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            batch_sampler=sampler
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True
        )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True, 
        collate_fn=collator,
        pin_memory=True
    )

    if args.linear_scaling_lr:
        # batch size 16 = 0.001 LR
        args.learning_rate *= (args.batch_size / 16)
    num_epochs = args.num_epochs
    num_training_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    steps_per_epoch = len(train_data_loader) if args.bucketing else num_training_samples // args.batch_size
    num_steps = num_epochs * steps_per_epoch
    eval_every = steps_per_epoch
    asr_every_num_epochs = 5  # every 5 epochs, run ASR
    training_stats = f'\nLearning Rate: {args.learning_rate}\n' \
        f'Num epochs: {num_epochs}\n' \
        f'Num training samples: {num_training_samples}\n' \
        f'Num val samples: {num_val_samples}\n' \
        f'Steps per epoch: {steps_per_epoch}\n' \
        f'Num steps: {num_steps / 1000}k\n' \
        f'Epoch/eval every {eval_every} steps\n' \
        f'ASR every {asr_every_num_epochs} epochs'
    log(training_stats)

    # create tensorboard
    writer = SummaryWriter(training_output_directory)

    # training loop
    epoch, total_iterations, best_val_loss = 0, 0, np.inf

    # resume training from specific checkpoint
    optimiser_state = None
    if args.checkpoint_path:
        # if checkpoint contains GPU tensors, tensors loaded to GPU by default
        # use map_location='cpu' to avoid GPU usage here
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimiser_state = checkpoint['optimiser_state_dict']
        epoch = checkpoint['epoch']
        total_iterations = checkpoint['total_iterations']
        best_val_loss = checkpoint['best_val_loss']
        log(f'Loaded model at epoch {epoch}, iterations {total_iterations}')
        del checkpoint

    # create optimiser and lr scheduler
    # it is advised that when multiplying batch size by K, one should multiply LR by sqrt(K)
    # a rule of thumb is to multiply the LR by K
    optimiser = CustomOptimiser(
        params=net.parameters(),
        target_lr=args.learning_rate,
        num_steps=num_steps / args.gradient_accumulation_steps if args.gradient_accumulation else num_steps,
        warmup_rate=args.warmup_rate,
        decay=args.lr_decay
    )
    if optimiser_state and not args.reset_optimiser:
        optimiser.load_state_dict(optimiser_state)
        steps_left = num_steps - total_iterations
        log('Loaded optimiser state')
    else:
        steps_left = num_steps
        epoch = 0
    total_steps_left = steps_left
    optimiser.init()
    log(f'Warmup steps: {optimiser.warmup_steps}\nSteps left: {steps_left}\n')

    net.train()
    net.zero_grad()  # zero the param gradients - same as optim.zero_grad()

    deepspeech_asr = DeepSpeechASR(args.deepspeech_host) if args.deepspeech_host else None

    finished_first_eval, finished_training = False, False
    while not finished_training:
        running_loss, running_step_time = 0, 0

        for i, train_data in enumerate(train_data_loader):
            video_paths, (windows, speaker_embeddings, lengths), gt_mel_specs, target_lengths, weights = train_data  # batch
            windows = windows.to(device)
            speaker_embeddings = speaker_embeddings.to(device)
            lengths = lengths.to(device)
            gt_mel_specs = gt_mel_specs.to(device)

            # forward + backward + optimise
            start_time = time.time()
            outputs = net(windows, speaker_embeddings, lengths)  # expects ([B, 1, T, H, W], [B, 256, 1])

            # calculate loss
            loss = criterion(gt_mel_specs, outputs, target_lengths, weights)
            running_loss += loss.item()
            if args.gradient_accumulation:
                loss /= args.gradient_accumulation_steps
            loss.backward()  # accumulates the gradients from every forward pass

            if args.gradient_accumulation: 
                if (i + 1) % args.gradient_accumulation_steps == 0: 
                    optimiser.step()
                    optimiser.zero_grad()  # only zero the gradients after every update
            else:
                optimiser.step()
                optimiser.zero_grad()

            step_time = time.time() - start_time
            running_step_time += step_time

            if args.debug:
                log(f'Iteration took {step_time:.2f}s')
                log(f'GT vs. Pred: {gt_mel_specs[0]}\n{outputs[0]}')

            total_iterations += 1
            steps_left -= 1

            if (i + 1) % args.log_every == 0:
                running_loss /= args.log_every
                eta_days = ((running_step_time / args.log_every) * steps_left) / 86400  # in days
                epoch_progress = round((epoch / num_epochs) * 100, 1)
                log(f'[Epoch: {epoch}/{num_epochs} ({epoch_progress}%), Iteration: {i + 1}, Total Iteration: {total_iterations}, LR: {optimiser.rate}] Loss: {running_loss}, ETA: {eta_days:.2f} days')
                writer.add_scalar('Loss/train', running_loss, global_step=total_iterations)
                writer.add_scalar('LR', optimiser.rate, global_step=total_iterations)
                running_loss, running_step_time = 0, 0

            run_eval = False
            run_first_eval = not finished_first_eval and i + 1 == 100
            if args.eval_n_times:
                if steps_left in [int(total_steps_left * x) for x in np.arange(0, 1, 1 / args.eval_n_times)]:
                    run_eval = True
            elif run_first_eval or (i + 1) == len(train_data_loader) or steps_left == 0:
                # eval at 100 steps, at the end of every epoch or at the end of training
                run_eval = True
        
            if run_eval:
                # run validation data
                log('Running evaluation...')
                net.eval()  # switch to evaluation mode
                run_asr = deepspeech_asr and epoch % asr_every_num_epochs == 0

                with torch.no_grad():  # turn off gradients computation
                    running_val_loss, running_val_gl_stats, running_val_pwgan_stats = 0, [], []
                    for j, val_data in enumerate(tqdm(val_data_loader, total=args.num_eval_samples)):  # loaders reset from beginning
                        val_video_paths, (val_windows, val_speaker_embeddings, val_lengths), val_gt_mel_specs, val_target_lengths, val_weights = val_data  # batch
                        val_windows = val_windows.to(device)
                        val_speaker_embeddings = val_speaker_embeddings.to(device)
                        val_lengths = val_lengths.to(device)
                        val_gt_mel_specs = val_gt_mel_specs.to(device)

                        val_outputs = net(val_windows, val_speaker_embeddings, val_lengths)  # expects ([B, 1, T, H, W], [B, 256, 1])

                        val_loss = criterion(val_gt_mel_specs, val_outputs, val_target_lengths, val_weights)
                        running_val_loss += val_loss.item()

                        val_gt_mel = val_gt_mel_specs[0].cpu().numpy()  # NOTE: only 1 sample per validation batch
                        val_pred_mel = val_outputs[0].cpu().numpy()

                        # calculate stois and wer/cer of a sample using different vocoders
                        for vocoder in args.vocoders:

                            if vocoder == 'gl': 
                                val_gt_wav, val_pred_wav = griffin_lim([val_gt_mel, val_pred_mel])
                            else:
                                val_gt_wav, val_pred_wav = parallel_wavegan([val_gt_mel, val_pred_mel], args.pwgan_checkpoint)

                            if len(val_gt_wav) > len(val_pred_wav):
                                val_gt_wav = val_gt_wav[:val_pred_wav.shape[0]]
                            else:
                                val_pred_wav = val_pred_wav[:val_gt_wav.shape[0]]

                            _stoi = stoi(val_gt_wav, val_pred_wav, fs_sig=hparams.sample_rate)
                            _estoi = stoi(val_gt_wav, val_pred_wav, fs_sig=hparams.sample_rate, extended=True)

                            _wer, _cer = None, None
                            if run_asr:
                                save_wav(val_gt_wav, '/tmp/gt.wav', hparams.sample_rate)
                                save_wav(val_pred_wav, '/tmp/pred.wav', hparams.sample_rate)
                                gt_prediction = deepspeech_asr.run('/tmp/gt.wav')[0]
                                pred_prediction = deepspeech_asr.run('/tmp/pred.wav')[0]
                                try:
                                    _wer = calculate_wer(gt_prediction, pred_prediction)
                                    _cer = calculate_cer(gt_prediction, pred_prediction)
                                except ValueError:  # usually caused by an empty string in gt or pred
                                    log(f'WER/CER failed: {gt_prediction}, {pred_prediction}')
                                    continue        

                            stats = [_stoi, _estoi, _wer, _cer]

                            if vocoder == 'gl': 
                                running_val_gl_stats.append([*stats])
                            else:
                                running_val_pwgan_stats.append([*stats])

                        if j == args.num_eval_samples - 1:
                            break

                    # write stats to tensorboard
                    writer.add_scalar('Loss/val', running_val_loss / args.num_eval_samples, global_step=total_iterations)
                    gl_stois, gl_estois, gl_wers, gl_cers = zip(*running_val_gl_stats)
                    stats_d = {'STOI/val': {'gl': gl_stois}, 'ESTOI/val': {'gl': gl_estois}}
                    if run_asr:
                        stats_d.update({'WER/val': {'gl': gl_wers}, 'CER/val': {'gl': gl_cers}})
                    if running_val_pwgan_stats:
                        pwgan_stois, pwgan_estois, pwgan_wers, pwgan_cers = zip(*running_val_pwgan_stats)
                        stats_d['STOI/val']['pwgan'] = pwgan_stois
                        stats_d['ESTOI/val']['pwgan'] = pwgan_estois
                        if run_asr:
                            stats_d['WER/val']['pwgan'] = pwgan_wers
                            stats_d['CER/val']['pwgan'] = pwgan_cers
                    for label, stats in stats_d.items():
                        writer.add_scalars(label, {k: np.mean(v) for k, v in stats.items()}, global_step=total_iterations)

                    # save checkpoint if new best val loss, every 5 epochs or at the end of training
                    # don't save at 100 iterations
                    save_checkpoint = (running_val_loss < best_val_loss or epoch % 5 == 0 or steps_left == 0) and total_iterations != 100
                    if save_checkpoint:
                        checkpoint_path = checkpoint_directory.joinpath(f'model_checkpoint_{epoch}_{total_iterations}.pt')
                        torch.save({
                            'epoch': epoch,
                            'total_iterations': total_iterations,
                            'best_val_loss': best_val_loss,
                            'model_state_dict': net.state_dict(),
                            'optimiser_state_dict': optimiser.state_dict(),
                        }, checkpoint_path)

                        # update best val loss
                        if running_val_loss < best_val_loss:
                            best_val_loss = running_val_loss

                    # save out validation and training samples
                    for video_path, window, window_length, gt, pred, target_length, weight, name in zip(
                        [video_paths[0], val_video_paths[0]],
                        [windows[0], val_windows[0]],
                        [lengths[0], val_lengths[0]],
                        [gt_mel_specs[0], val_gt_mel_specs[0]],
                        [outputs[0], val_outputs[0]],
                        [target_lengths[0], val_target_lengths[0]],
                        [weights[0], val_weights[0]],
                        ['train', 'val']
                    ):
                        window_length = int(window_length.item())
                        target_length = int(target_length)

                        window = window.cpu().numpy()[:, :window_length]  # random window, de-pad with the length
                        window = np.asarray([cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                             for frame in window])  # for displaying purposes

                        gt = gt.cpu().numpy()[:target_length]  # remove padding
                        pred = pred.cpu().numpy()[:target_length]

                        assert window.shape[1] == window_length
                        assert (gt.shape[0], pred.shape[0]) == (target_length, target_length)

                        writer.add_images(f'Window/{name}', window, global_step=total_iterations, dataformats='CNHW')
                        writer.add_figure(f'Figure/{name}', plot_spectrogram(
                            pred,
                            title='GT vs. Pred Mel-specs',
                            target_spectrogram=gt,
                            loss=criterion([torch.tensor(gt)], [torch.tensor(pred)], [torch.tensor(target_length)], [torch.tensor(weight)]), 
                            video_path=video_path
                        ), global_step=total_iterations)
                        for _type, mel in zip(['gt', 'pred'], [gt, pred]):
                            writer.add_audio(
                                f'Audio/{name}_{_type}_gl', 
                                norm_wav(griffin_lim([mel])[0]), 
                                global_step=total_iterations,
                                sample_rate=hparams.sample_rate
                            )
                            if args.pwgan_checkpoint:
                                writer.add_audio(
                                    f'Audio/{name}_{_type}_pwgan', 
                                    norm_wav(parallel_wavegan([mel], args.pwgan_checkpoint)[0]), 
                                    global_step=total_iterations,
                                    sample_rate=hparams.sample_rate
                                )

                    # save out the 4 lrs3 v2s samples if applicable
                    if args.v2s_samples_dataset_location:
                        assert args.pwgan_checkpoint is not None
                        test_loader = torch.utils.data.DataLoader(
                            CustomDataset(args.v2s_samples_dataset_location),
                            batch_size=1,
                            num_workers=1,
                            collate_fn=collator,
                            pin_memory=True
                        )
                        for test_data in tqdm(test_loader):
                            test_video_paths, (test_windows, test_speaker_embeddings, test_lengths), _, _, _ = test_data
                            test_outputs = net(test_windows, test_speaker_embeddings, test_lengths)
                            test_pred_mel = test_outputs[0].cpu().numpy()
                            writer.add_audio(
                                f'V2S_Samples:{test_video_paths[0]}', 
                                norm_wav(parallel_wavegan([test_pred_mel], args.pwgan_checkpoint)[0]),
                                global_step=total_iterations,
                                sample_rate=hparams.sample_rate
                            )

                net.train()  # switch back to training mode
                finished_first_eval = True

            if steps_left == 0:
                log('Training complete...')
                finished_training = True
                writer.close()
                break

            writer.flush()
    
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('run_index', type=int)
    parser.add_argument('training_dataset_location')
    parser.add_argument('val_dataset_location')
    parser.add_argument('--v2s_samples_dataset_location')  # the 4 LRS3 samples
    parser.add_argument('--conformer_size', choices=['s', 'm', 'l'], default='s')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--num_eval_samples', type=int, default=100)
    parser.add_argument('--bucketing', action='store_true')
    parser.add_argument('--balanced_bucketing', action='store_true')
    parser.add_argument('--last_frame_padding', action='store_true')
    parser.add_argument('--horizontal_flipping', action='store_true')
    parser.add_argument('--intensity_augmentation', action='store_true')
    parser.add_argument('--time_masking', action='store_true')
    parser.add_argument('--erasing', action='store_true')
    parser.add_argument('--random_cropping', action='store_true')
    parser.add_argument('--min_sample_duration', type=int)
    parser.add_argument('--max_sample_duration', type=int)
    parser.add_argument('--slicing', action='store_true')
    parser.add_argument('--deepspeech_host')
    parser.add_argument('--reset_optimiser', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--eval_n_times', type=int)
    parser.add_argument('--force_train_batch_size', action='store_true')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--linear_scaling_lr', action='store_true')
    parser.add_argument('--num_dataset_samples', type=int)
    parser.add_argument('--pretrained_resnet_path')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--time_mask_by_frame', action='store_true')
    parser.add_argument('--pwgan_checkpoint')
    parser.add_argument('--vocoders', type=lambda s: s.split(','), default=['gl'])
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
