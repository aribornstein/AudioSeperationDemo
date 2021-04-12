import os
from argparse import ArgumentParser
# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet
# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix
# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--download', type=bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gpus', type=int, default=None)
    args = parser.parse_args()

    # This will automatically download MiniLibriMix from Zenodo on the first run.
    train_loader, val_loader = LibriMix.loaders_from_mini(task="sep_clean", batch_size=16)

    # Tell DPRNN that we want to separate to 2 sources.
    model = DPRNNTasNet(n_src=2)

    # PITLossWrapper works with any loss function.
    loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    system = System(model, optimizer, loss, train_loader, val_loader)

    # Train for 1 epoch using a single GPU. If you're running this on Google Colab,
    # be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
    trainer = Trainer(max_epochs=args.max_epochs, gpus=args.gpus)
    trainer.fit(system)