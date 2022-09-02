import argparse
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torchinfo import summary

from architecture import V2S
from dataset import CustomDataset
from hparams import HParams
from loss import SpectralConvergenceLoss
from optimiser import CustomOptimiser
from utils import plot_spectrogram, save_wav, spec_2_wav

# TODO: Use tensorboard for graphs, images and audio


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    # create network
    net = V2S(conformer_type='s', device=device).to(device)
    if args.debug:
        summary(net, input_size=[(32, 1, 20, 88, 88), (32, 256, 1)], device=device)

    # create losses
    l1_loss = torch.nn.L1Loss()
    spectral_convergence_loss = SpectralConvergenceLoss()

    # create optimiser and lr scheduler
    optimiser = CustomOptimiser(
        optimiser=torch.optim.AdamW(net.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay),
        target_lr=args.learning_rate,
        num_steps=args.num_steps,
        warmup_rate=args.warmup_rate
    )

    # create data loaders
    train_data_loader = torch.utils.data.DataLoader(
        CustomDataset(sample_pool_location=args.training_sample_pool_location),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        CustomDataset(sample_pool_location=args.val_sample_pool_location),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    # training loop
    epoch, total_iterations = 0, 0
    training_losses, val_losses = [], []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # resume training from specific checkpoint
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        epoch = checkpoint['epoch']
        total_iterations = checkpoint['total_iterations']

    def animate(k):
        ax.clear()

        # plot training
        xs = [args.log_every * j for j in range(len(training_losses))]
        ys = training_losses.copy()
        ax.plot(xs, ys, label='Training')

        # plot val
        xs = [args.eval_every * j for j in range(len(val_losses))]
        ys = val_losses.copy()
        ax.plot(xs, ys, label='Val')

        ax.set_xlabel('Num Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(f'Epoch {epoch}, Total Iterations {total_iterations}')
        ax.legend()

    if args.live_plot:
        ani = animation.FuncAnimation(fig, animate, interval=5000)
        plt.draw()  # non-blocking

    net.train()
    net.zero_grad()  # zero the param gradients - same as optim.zero_grad()

    hparams = HParams()

    finished_training = False
    while not finished_training:
        running_loss = 0

        for i, train_data in enumerate(train_data_loader):
            (windows, speaker_embeddings), gt_mel_specs = train_data  # batch
            if windows.shape[0] != args.batch_size:
                continue
            windows = windows.unsqueeze(1).to(device)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).to(device)
            gt_mel_specs = gt_mel_specs.to(device)

            # forward + backward + optimise
            start_time = time.time()
            outputs = net(windows, speaker_embeddings)  # expects ([B, 1, T, H, W], [B, 256, 1])
            loss = l1_loss(gt_mel_specs, outputs) + spectral_convergence_loss(gt_mel_specs, outputs)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total_iterations += 1

            if args.debug:
                print(f'Iteration took {time.time() - start_time:.2f}s')
                print('GT vs. Pred:', gt_mel_specs[0], outputs[0])

            running_loss += loss.item()
            if (i + 1) % args.log_every == 0:
                running_loss /= args.log_every
                print(f'[Epoch: {epoch}, Iteration: {i + 1}, Total Iteration: {total_iterations}, LR: {optimiser._rate}] loss: {running_loss}')
                training_losses.append(running_loss)
                running_loss = 0
                if args.live_plot:
                    plt.pause(0.1)

            if total_iterations % args.eval_every == 0:
                # run validation data
                net.eval()  # switch to evaluation mode
                with torch.no_grad():  # turn off gradients computation
                    running_val_loss = 0
                    for j, val_data in enumerate(val_data_loader):
                        (val_windows, val_speaker_embeddings), val_gt_mel_specs = val_data  # batch
                        if val_windows.shape[0] != args.batch_size:
                            continue
                        val_windows = val_windows.unsqueeze(1).to(device)
                        val_speaker_embeddings = val_speaker_embeddings.unsqueeze(-1).to(device)
                        val_gt_mel_specs = val_gt_mel_specs.to(device)

                        val_outputs = net(val_windows, val_speaker_embeddings)  # expects ([B, 1, T, H, W], [B, 256, 1])
                        val_loss = l1_loss(val_gt_mel_specs, val_outputs) + spectral_convergence_loss(val_gt_mel_specs, val_outputs)
                        running_val_loss += val_loss.item()

                        if j == args.num_eval_batches - 1:
                            break
                    val_losses.append(running_val_loss / args.num_eval_batches)

                net.train()  # switch back to training mode

                # save out validation and training samples
                for gts, preds, name in zip([gt_mel_specs, val_gt_mel_specs], [outputs, val_outputs], ['training', 'val']):
                    gt, pred = gts[0].detach().cpu().numpy(), preds[0].detach().cpu().numpy()  # random gt, pred
                    plot_spectrogram(
                        pred,
                        f'{name}_mel_spec.png',
                        title='GT vs. Pred Mel-specs',
                        target_spectrogram=gt,
                        max_len=hparams.num_mels
                    )
                    save_wav(spec_2_wav(pred.T, hparams), f'{name}.wav', hparams.sample_rate)

            if total_iterations % args.checkpoint_every == 0:
                # save model checkpoint
                checkpoint_path = f'model_checkpoint_{epoch}_{total_iterations}.pt'
                torch.save({
                    'epoch': epoch,
                    'total_iterations': total_iterations + 1,
                    'model_state_dict': net.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                }, checkpoint_path)

            if total_iterations == args.num_steps:
                print('Training complete...')
                finished_training = True
                break

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_sample_pool_location')
    parser.add_argument('val_sample_pool_location')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay',  type=float, default=0.01)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.98))
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--live_plot', action='store_true')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--num_eval_batches', type=int, default=16)
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
