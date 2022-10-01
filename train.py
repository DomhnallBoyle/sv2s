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
from torchinfo import summary
from tqdm import tqdm

# seed the randomisers
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from architecture import V2S
from asr import DeepSpeechASR
from dataset import CustomCollate, CustomDataset, PAD_VALUE
from hparams import HParams
from loss import SpectralConvergenceLoss
from optimiser import CustomOptimiser
from sampler import BucketSampler
from utils import plot_spectrogram, save_wav, spec_2_wav


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using:', device)

    # create network
    net = V2S(conformer_type=args.conformer_size, device=device).to(device)
    # if args.debug:
    #     summary(net, input_size=[(32, 1, 20, 88, 88), (32, 256, 1), (32,)], device=device)

    # create losses
    l1_loss = torch.nn.L1Loss()
    # l1_loss = torch.nn.L1Loss(reduction='none')
    spectral_convergence_loss = SpectralConvergenceLoss()

    # create datasets
    train_dataset = CustomDataset(
        location=args.training_dataset_location,
        horizontal_flipping=args.horizontal_flipping,
        intensity_augmentation=args.intensity_augmentation,
        time_masking=args.time_masking,
        erasing=args.erasing,
        random_cropping=args.random_cropping
    )
    val_dataset = CustomDataset(location=args.val_dataset_location)

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
                batch_size=args.batch_size,
                lengths=train_dataset.get_lengths()
            )
        )
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=1,
            collate_fn=collator,
            pin_memory=True,
            batch_sampler=BucketSampler(
                batch_size=args.batch_size,
                lengths=val_dataset.get_lengths(),
                force_batch_size=args.force_eval_batch_size
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
        )
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True
        )

    num_epochs = args.num_epochs
    num_training_samples = len(train_dataset)
    steps_per_epoch = num_training_samples // args.batch_size
    num_steps = num_epochs * steps_per_epoch
    eval_every = steps_per_epoch
    asr_every_num_epochs = 5  # every 5 epochs, run ASR
    print(f'Num epochs: {num_epochs}\n'
          f'Num training samples: {num_training_samples}\n'
          f'Steps per epoch: {steps_per_epoch}\n'
          f'Num steps: {num_steps / 1000}k\n'
          f'Epoch/eval every {eval_every} steps\n'
          f'ASR every {asr_every_num_epochs} epochs')

    training_output_directory = Path(f'runs/{args.name}')
    checkpoint_directory = training_output_directory.joinpath('checkpoints')
    checkpoint_directory.mkdir(exist_ok=True, parents=True)
    with training_output_directory.joinpath('command.txt').open('w') as f:
        for arg in sys.argv:
            f.write(f'{arg} \\\n')

    # create tensorboard
    writer = SummaryWriter(training_output_directory)

    # training loop
    epoch, total_iterations, best_val_loss = 0, 0, np.inf

    # resume training from specific checkpoint
    optimiser_state = None
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimiser_state = checkpoint['optimiser_state_dict']
        epoch = checkpoint['epoch']
        total_iterations = checkpoint['total_iterations']
        best_val_loss = checkpoint['best_val_loss']
        print(f'Loaded model at epoch {epoch}, iterations {total_iterations}')

    # create optimiser and lr scheduler
    # it is advised that when multiplying batch size by K, one should multiply LR by sqrt(K)
    # a rule of thumb is to multiply the LR by K
    optimiser = CustomOptimiser(
        params=net.parameters(),
        target_lr=args.learning_rate,
        num_steps=num_steps,
        warmup_rate=args.warmup_rate,
        decay=args.lr_decay
    )
    if optimiser_state and not args.reset_optimiser:
        optimiser.load_state_dict(optimiser_state)
        steps_left = num_steps - total_iterations
        print('Loaded optimiser state')
    else:
        steps_left = num_steps
    total_steps_left = steps_left
    optimiser.init()
    print(f'Warmup steps: {optimiser.warmup_steps}\n'
          f'Steps left: {steps_left}\n')

    net.train()
    net.zero_grad()  # zero the param gradients - same as optim.zero_grad()

    hparams = HParams()
    deepspeech_asr = DeepSpeechASR(args.deepspeech_host) if args.deepspeech_host else None

    finished_training = False
    while not finished_training:
        running_loss = 0

        for i, train_data in enumerate(train_data_loader):
            _, (windows, speaker_embeddings, lengths), gt_mel_specs, target_lengths = train_data  # batch
            windows = windows.to(device)
            speaker_embeddings = speaker_embeddings.to(device)
            lengths = lengths.to(device)
            gt_mel_specs = gt_mel_specs.to(device)

            # forward + backward + optimise
            start_time = time.time()
            outputs = net(windows, speaker_embeddings, lengths)  # expects ([B, 1, T, H, W], [B, 256, 1])

            if args.debug:
                print(f'Iteration took {time.time() - start_time:.2f}s')
                print('GT vs. Pred:', gt_mel_specs[0], outputs[0])

            # calculate loss
            loss = l1_loss(gt_mel_specs, outputs) + spectral_convergence_loss(gt_mel_specs, outputs)

            # # compute the training loss over a batch
            # # padding should not be used in computation of loss - mask it out
            # batch_loss = 0
            # for gt_mel_spec, output, mask in zip(gt_mel_specs, outputs, masks):
            #     mask = mask.to(device)
            #     gt_mel_spec, output = gt_mel_spec.masked_select(mask).reshape(-1, 80), output.masked_select(mask).reshape(-1, 80)
            #     batch_loss += (l1_loss(gt_mel_spec, output) + spectral_convergence_loss(gt_mel_spec, output))
            # loss = batch_loss / len(gt_mel_specs)

            # # compute the training loss over a batch
            # # padding should not be used in computation of loss - mask it out
            # batch_loss = 0
            # outputs = torch.nn.utils.rnn.unpad_sequence(outputs, target_lengths, batch_first=True)
            # for gt_mel_spec, output in zip(gt_mel_specs, outputs):
            #     gt_mel_spec = gt_mel_spec.to(device)
            #     assert gt_mel_spec.shape == output.shape
            #     batch_loss += (l1_loss(gt_mel_spec, output) + spectral_convergence_loss(gt_mel_spec, output))
            # loss = batch_loss / len(gt_mel_specs)

            # calculate loss
            # https://discuss.pytorch.org/t/ignore-padding-area-in-loss-computation/95804
            # loss = l1_loss(gt_mel_specs, outputs) + spectral_convergence_loss(gt_mel_specs, outputs)  # unreduced loss
            # loss_mask = (gt_mel_specs != PAD_VALUE).to(device)  # mask out loss values where there's padding
            # loss_masked = loss.where(loss_mask, torch.tensor(PAD_VALUE).to(device))  # this has been tested
            # loss = loss_masked.sum() / loss_mask.sum()  # reduce loss (mean of non-padded values)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total_iterations += 1
            steps_left -= 1

            running_loss += loss.item()
            if (i + 1) % args.log_every == 0:
                running_loss /= args.log_every
                print(f'[Epoch: {epoch}, Iteration: {i + 1}, Total Iteration: {total_iterations}, LR: {optimiser._rate}] loss: {running_loss}')
                writer.add_scalar('Loss/train', running_loss, global_step=total_iterations)
                writer.add_scalar('LR', optimiser._rate, global_step=total_iterations)
                running_loss = 0

            run_eval = False
            if args.eval_n_times:
                if steps_left in [int(total_steps_left * x) for x in np.arange(0, 1, 1 / args.eval_n_times)]:
                    run_eval = True
            elif total_iterations == 100 or (i + 1) == len(train_data_loader):
                # eval at 100 steps or at the end of every epoch
                run_eval = True

            if run_eval:
                print('Running evaluation...')
                # run validation data
                net.eval()  # switch to evaluation mode
                with torch.no_grad():  # turn off gradients computation
                    running_val_loss, running_val_stoi, running_val_estoi, running_val_wer, running_val_cer = 0, 0, 0, 0, 0
                    for j, val_data in enumerate(val_data_loader):
                        _, (val_windows, val_speaker_embeddings, val_lengths), val_gt_mel_specs, val_target_lengths = val_data  # batch
                        val_windows = val_windows.to(device)
                        val_speaker_embeddings = val_speaker_embeddings.to(device)
                        val_lengths = val_lengths.to(device)
                        val_gt_mel_specs = val_gt_mel_specs.to(device)

                        val_outputs = net(val_windows, val_speaker_embeddings, val_lengths)  # expects ([B, 1, T, H, W], [B, 256, 1])

                        val_loss = l1_loss(val_gt_mel_specs, val_outputs) + spectral_convergence_loss(val_gt_mel_specs, val_outputs)

                        # val_batch_loss = 0
                        # val_outputs = torch.nn.utils.rnn.unpad_sequence(val_outputs, val_target_lengths, batch_first=True)
                        # for val_gt_mel_spec, val_output in zip(val_gt_mel_specs, val_outputs):
                        #     val_gt_mel_spec = val_gt_mel_spec.to(device)
                        #     assert val_gt_mel_spec.shape == val_output.shape
                        #     val_batch_loss += (l1_loss(val_gt_mel_spec, val_output) + spectral_convergence_loss(val_gt_mel_spec, val_output))
                        # val_loss = val_batch_loss / len(val_gt_mel_specs)

                        # val_loss = l1_loss(val_gt_mel_specs, val_outputs) + spectral_convergence_loss(val_gt_mel_specs, val_outputs)
                        # val_loss_mask = (val_gt_mel_specs != PAD_VALUE).to(device)
                        # val_loss_masked = val_loss.where(val_loss_mask, torch.tensor(PAD_VALUE).to(device))
                        # val_loss = val_loss_masked.sum() / val_loss_mask.sum()

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
                                    print(f'WER/CER failed: {gt_prediction}, {pred_prediction}')

                        av_stoi /= args.batch_size
                        av_estoi /= args.batch_size
                        av_wer /= args.batch_size
                        av_cer /= args.batch_size

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

                    # save checkpoint if new best val loss or every 5 epochs
                    # don't save at 100 iterations
                    save_checkpoint = (running_val_loss < best_val_loss or epoch % 5 == 0) and total_iterations != 100
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
                    for _windows, _lengths, gts, preds, _target_lengths, name in zip(
                        [windows, val_windows],
                        [lengths, val_lengths],
                        [gt_mel_specs, val_gt_mel_specs],
                        [outputs, val_outputs],
                        [target_lengths, val_target_lengths],
                        ['train', 'val']
                    ):
                        window_length = int(_lengths[0].item())
                        target_length = int(_target_lengths[0])

                        window = _windows[0].cpu().numpy()[:, :window_length]  # random window, de-pad with the length
                        window = np.asarray([cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                             for frame in window])  # for displaying purposes

                        gt = gts[0].cpu().numpy()[:target_length]  # remove padding
                        pred = preds[0].cpu().numpy()[:target_length]

                        assert window.shape[1] == window_length
                        assert (gt.shape[0], pred.shape[0]) == (target_length, target_length)

                        writer.add_images(f'Window/{name}', window, global_step=total_iterations, dataformats='CNHW')
                        writer.add_figure(f'Figure/{name}', plot_spectrogram(
                            pred,
                            title='GT vs. Pred Mel-specs',
                            target_spectrogram=gt,
                        ), global_step=total_iterations)
                        for _type, mel in zip(['gt', 'pred'], [gt, pred]):
                            wav = spec_2_wav(mel.T, hparams)
                            writer.add_audio(f'Audio/{name}_{_type}', wav, global_step=total_iterations,
                                             sample_rate=hparams.sample_rate)

                net.train()  # switch back to training mode

            if steps_left == 0:
                print('Training complete...')
                finished_training = True
                writer.close()
                break

            writer.flush()
    
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('training_dataset_location')
    parser.add_argument('val_dataset_location')
    parser.add_argument('--conformer_size', choices=['s', 'm', 'l'], default='s')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--deepspeech_host')
    parser.add_argument('--reset_optimiser', action='store_true')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--eval_n_times', type=int)
    parser.add_argument('--force_eval_batch_size', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
