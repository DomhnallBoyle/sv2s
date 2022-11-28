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

from architecture import V2S
from asr import DeepSpeechASR
from dataset import CustomCollate, CustomDataset
from hparams import hparams
from loss import CustomLoss
from optimiser import CustomOptimiser
from sampler import BucketSampler
from utils import plot_spectrogram, save_wav, spec_2_wav

log_path = None


def log(s):
    print(s)
    with log_path.open('a') as f:
        f.write(f'{s}\n')


def main(args):
    global log_path

    # seed the randomisers
    torch.manual_seed(hparams.seed)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)

    training_output_directory = Path(f'runs/{args.name}')
    checkpoint_directory = training_output_directory.joinpath('checkpoints')
    checkpoint_directory.mkdir(exist_ok=True, parents=True)
    log_path = training_output_directory.joinpath(f'log_{args.run_index}.txt')
    with log_path.open('w') as f:
        for arg in sys.argv:
            f.write(f'{arg} \\\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f'\nUsing: {device}')

    # create network
    net = V2S(
        conformer_type=args.conformer_size, 
        device=device, 
        pretrained_resnet_path=args.pretrained_resnet_path, 
        freeze_encoder=args.freeze_encoder
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
        use_duration_range=args.train_use_duration_range,
        slicing=args.slicing,
        time_mask_by_frame=args.time_mask_by_frame,
        num_samples=args.num_samples
    )
    val_dataset = CustomDataset(
        location=args.val_dataset_location,
        min_sample_duration=args.min_sample_duration,
        max_sample_duration=args.max_sample_duration,
        use_duration_range=args.val_use_duration_range,
        num_samples=args.num_samples
    )

    # create custom collate fn
    collator = CustomCollate(last_frame_padding=args.last_frame_padding)

    # create data loaders
    if args.bucketing:
        # ORDER: (1) batch sampler, (2) dataset[i], (3) collate_fn
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            batch_sampler=BucketSampler(
                dataset=train_dataset,
                batch_size=args.batch_size,
                force_batch_size=args.force_train_batch_size
            )
        )
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=1,
            collate_fn=collator,
            pin_memory=True,
            batch_sampler=BucketSampler(
                dataset=val_dataset,
                batch_size=args.batch_size
            )
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
            batch_size=args.batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True
        )

    args.gradient_accumulation = args.gradient_accumulation_steps is not None
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
    total_steps_left = steps_left
    optimiser.init()
    log(f'Warmup steps: {optimiser.warmup_steps}\nSteps left: {steps_left}\n')

    net.train()
    net.zero_grad()  # zero the param gradients - same as optim.zero_grad()

    deepspeech_asr = DeepSpeechASR(args.deepspeech_host) if args.deepspeech_host else None

    finished_training = False
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
                log(f'[Epoch: {epoch}/{num_epochs} ({epoch_progress}%), Iteration: {i + 1}, Total Iteration: {total_iterations}, LR: {optimiser._rate}] Loss: {running_loss}, ETA: {eta_days:.2f} days')
                writer.add_scalar('Loss/train', running_loss, global_step=total_iterations)
                writer.add_scalar('LR', optimiser._rate, global_step=total_iterations)
                running_loss, running_step_time = 0, 0

            run_eval = False
            if args.eval_n_times:
                if steps_left in [int(total_steps_left * x) for x in np.arange(0, 1, 1 / args.eval_n_times)]:
                    run_eval = True
            elif total_iterations == 100 or (i + 1) == len(train_data_loader) or steps_left == 0:
                # eval at 100 steps, at the end of every epoch or at the end of training
                run_eval = True
        
            if run_eval:
                log('Running evaluation...')
                # run validation data
                net.eval()  # switch to evaluation mode
                with torch.no_grad():  # turn off gradients computation
                    running_val_loss, running_val_stoi, running_val_estoi, running_val_wer, running_val_cer = 0, 0, 0, 0, 0
                    for j, val_data in enumerate(val_data_loader):
                        val_video_paths, (val_windows, val_speaker_embeddings, val_lengths), val_gt_mel_specs, val_target_lengths, val_weights = val_data  # batch
                        val_windows = val_windows.to(device)
                        val_speaker_embeddings = val_speaker_embeddings.to(device)
                        val_lengths = val_lengths.to(device)
                        val_gt_mel_specs = val_gt_mel_specs.to(device)

                        val_outputs = net(val_windows, val_speaker_embeddings, val_lengths)  # expects ([B, 1, T, H, W], [B, 256, 1])

                        val_loss = criterion(val_gt_mel_specs, val_outputs, val_target_lengths, val_weights)

                        running_val_loss += val_loss.item()

                        # calculate stoi and estoi over a batch
                        av_stoi, av_estoi, av_wer, av_cer = 0, 0, 0, 0
                        for val_gt_mel, val_pred_mel in tqdm(zip(val_gt_mel_specs, val_outputs)):
                            val_gt_mel = val_gt_mel.cpu().numpy()
                            val_pred_mel = val_pred_mel.cpu().numpy()
                            
                            val_gt_wav = spec_2_wav(val_gt_mel.T, hparams)
                            val_pred_wav = spec_2_wav(val_pred_mel.T, hparams)
   
                            if len(val_gt_wav) > len(val_pred_wav):
                                val_gt_wav = val_gt_wav[:val_pred_wav.shape[0]]
                            else:
                                val_pred_wav = val_pred_wav[:val_gt_wav.shape[0]]
 
                            av_stoi += stoi(val_gt_wav, val_pred_wav, fs_sig=hparams.sample_rate)
                            av_estoi += stoi(val_gt_wav, val_pred_wav, fs_sig=hparams.sample_rate, extended=True)

                            if deepspeech_asr and epoch % asr_every_num_epochs == 0:
                                save_wav(val_gt_wav, '/tmp/gt.wav', hparams.sample_rate)
                                save_wav(val_pred_wav, '/tmp/pred.wav', hparams.sample_rate)
                                gt_prediction = deepspeech_asr.run('/tmp/gt.wav')[0]
                                pred_prediction = deepspeech_asr.run('/tmp/pred.wav')[0]
                                try:
                                    av_wer += calculate_wer(gt_prediction, pred_prediction)
                                    av_cer += calculate_cer(gt_prediction, pred_prediction)
                                except ValueError:
                                    log(f'WER/CER failed: {gt_prediction}, {pred_prediction}')

                        val_batch_size = len(val_video_paths)
                        av_stoi /= val_batch_size
                        av_estoi /= val_batch_size
                        av_wer /= val_batch_size
                        av_cer /= val_batch_size

                        running_val_stoi += av_stoi
                        running_val_estoi += av_estoi
                        running_val_wer += av_wer
                        running_val_cer += av_cer

                        if j == args.num_eval_batches - 1:
                            break

                    running_val_loss /= args.num_eval_batches
                    running_val_stoi /= args.num_eval_batches
                    running_val_estoi /= args.num_eval_batches
                    running_val_wer /= args.num_eval_batches
                    running_val_cer /= args.num_eval_batches

                    writer.add_scalar('Loss/val', running_val_loss, global_step=total_iterations)
                    writer.add_scalar('Stoi/val', running_val_stoi, global_step=total_iterations)
                    writer.add_scalar('Estoi/val', running_val_estoi, global_step=total_iterations)
                    if deepspeech_asr and epoch % asr_every_num_epochs == 0:
                        writer.add_scalar('WER/val', running_val_wer, global_step=total_iterations)
                        writer.add_scalar('CER/val', running_val_cer, global_step=total_iterations)

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
                    for _video_paths, _windows, _lengths, gts, preds, _target_lengths, name in zip(
                        [video_paths, val_video_paths],
                        [windows, val_windows],
                        [lengths, val_lengths],
                        [gt_mel_specs, val_gt_mel_specs],
                        [outputs, val_outputs],
                        [target_lengths, val_target_lengths],
                        ['train', 'val']
                    ):
                        video_path = _video_paths[0]                        

                        window_length = int(_lengths[0].item())
                        target_length = int(_target_lengths[0])

                        window = _windows[0].cpu().numpy()[:, :window_length]  # random window, de-pad with the length
                        window = np.asarray([cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                             for frame in window])  # for displaying purposes

                        gt = gts[0].cpu().numpy()[:target_length]  # remove padding
                        pred = preds[0].cpu().numpy()[:target_length]

                        assert window.shape[1] == window_length
                        assert (gt.shape[0], pred.shape[0]) == (target_length, target_length)

                        writer.add_text(f'Video Path/{name}', video_path, global_step=total_iterations)
                        writer.add_images(f'Window/{name}', window, global_step=total_iterations, dataformats='CNHW')
                        writer.add_figure(f'Figure/{name}', plot_spectrogram(
                            pred,
                            title='GT vs. Pred Mel-specs',
                            target_spectrogram=gt,
                            loss=criterion([torch.tensor(gt)], [torch.tensor(pred)], [torch.tensor(target_length)], [torch.tensor(1)])
                        ), global_step=total_iterations)
                        for _type, mel in zip(['gt', 'pred'], [gt, pred]):
                            wav = spec_2_wav(mel.T, hparams)
                            writer.add_audio(f'Audio/{name}_{_type}', wav, global_step=total_iterations,
                                             sample_rate=hparams.sample_rate)

                net.train()  # switch back to training mode

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
    parser.add_argument('--conformer_size', choices=['s', 'm', 'l'], default='s')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--num_eval_batches', type=int, default=16)
    parser.add_argument('--bucketing', action='store_true')
    parser.add_argument('--last_frame_padding', action='store_true')
    parser.add_argument('--horizontal_flipping', action='store_true')
    parser.add_argument('--intensity_augmentation', action='store_true')
    parser.add_argument('--time_masking', action='store_true')
    parser.add_argument('--erasing', action='store_true')
    parser.add_argument('--random_cropping', action='store_true')
    parser.add_argument('--min_sample_duration', type=int)
    parser.add_argument('--max_sample_duration', type=int)
    parser.add_argument('--train_use_duration_range', action='store_true')
    parser.add_argument('--val_use_duration_range', action='store_true')
    parser.add_argument('--slicing', action='store_true')
    parser.add_argument('--deepspeech_host')
    parser.add_argument('--reset_optimiser', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--eval_n_times', type=int)
    parser.add_argument('--force_train_batch_size', action='store_true')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--linear_scaling_lr', action='store_true')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--pretrained_resnet_path')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--time_mask_by_frame', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
