import argparse
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torchinfo import summary

from architecture import V2S
from dataset import CustomDataset
from loss import SpectralConvergenceLoss
from utils import plot_spectrogram

# TODO: Conformer is returning same output every time
#  removing conformer makes the network overfit as expected (with small dataset)
#  tried init weights of conformer but didn't work
#  I think I might be using a conformer for encoding which means it is learning a representation
#  there are conformer decoders available
#  LSTM on own also works
#  fix conformer code


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

    # create optimiser
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=0.01)

    # create data loaders
    training_data = CustomDataset(sample_pool_location=args.training_sample_pool_location)
    train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                                    shuffle=True)
    # val_data_loader = torch.utils.data.DataLoader()

    # training loop
    epoch, total_iterations = 0, 0
    training_losses = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(k):
        xs = [args.log_every * j for j in range(len(training_losses))]
        ys = training_losses.copy()
        ax.clear()
        ax.plot(xs, ys)
        # ax.set_ylim([0, max(ys) + 1])
        ax.set_xlabel('Num Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(f'Epoch {epoch}')

    if args.live_plot:
        ani = animation.FuncAnimation(fig, animate, interval=5000)
        plt.draw()  # non-blocking

    while True:
        running_loss = 0
        for i, data in enumerate(train_data_loader):
            (windows, speaker_embeddings), gt_mel_specs = data  # batch
            if windows.shape[0] != args.batch_size:
                continue
            windows = windows.unsqueeze(1).to(device)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).to(device)
            gt_mel_specs = gt_mel_specs.to(device)

            # zero the param gradients
            optimiser.zero_grad()

            # forward + backward + optimise
            start_time = time.time()
            outputs = net(windows, speaker_embeddings)  # expects ([B, 1, T, H, W], [B, 256, 1])
            loss = l1_loss(gt_mel_specs, outputs) + spectral_convergence_loss(gt_mel_specs, outputs)
            loss.backward()
            optimiser.step()
            if args.debug:
                print(f'Iteration took {time.time() - start_time:.2f}s')
                print('GT vs. Pred:', gt_mel_specs[0], outputs[0])

            running_loss += loss.item()
            if (i + 1) % args.log_every == 0:
                running_loss /= args.log_every
                print(f'[Epoch: {epoch}, Iteration: {i + 1}] loss: {running_loss}')
                training_losses.append(running_loss)
                running_loss = 0
                if args.live_plot:
                    plt.pause(0.1)

            total_iterations += 1

            if (total_iterations + 1) % args.eval_every == 0:
                # run validation data
                # save out validation and training samples

                # training_sample = outputs[0].detach().cpu().numpy()
                # wav = librosa.feature.inverse.mel_to_audio(training_sample, sr=16000)
                # wavfile.write('sample.wav', 16000, wav)

                training_gt, training_pred = gt_mel_specs[0].detach().cpu().numpy(), \
                                             outputs[0].detach().cpu().numpy()
                plot_spectrogram(
                    training_pred,
                    f'mel_spec.png',
                    title='GT vs. Pred Mel-specs',
                    target_spectrogram=training_gt,
                    max_len=80
                )

            if (total_iterations + 1) % args.checkpoint_every == 0:
                # save model checkpoint
                pass

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_sample_pool_location')
    # parser.add_argument('val_sample_pool_location')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--live_plot', action='store_true')
    parser.add_argument('--debug', action='store_true')

    main(parser.parse_args())
