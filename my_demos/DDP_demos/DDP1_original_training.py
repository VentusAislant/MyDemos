####################################################
## a single model trained on Single GPU
## date: Mar 26 2024
## author: Aislant Ventus (张浩杰)
####################################################
import time

import torch
from torch.utils.data import DataLoader
from torch import nn
from alexnet import AlexNet
from datasets import load_dataloader
import argparse


# Trainer for training the model
class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,  # ferquency of saving
    ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, sources, targets):
        self.optimizer.zero_grad()
        output = self.model(sources)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print('=' * 60)
        print(f'GPU: {self.gpu_id}, # of Epoch: {epoch + 1}, Batch size: {b_sz} | Steps: {len(self.train_data)}')
        print('=' * 60)
        for b_id, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            if b_id % 1000 == 0 or b_id == 0 or b_id == len(self.train_data) - 1:
                print(f'Step {b_id}, loss: {loss:.3f}')

    def _save_checkpoint(self, epoch):
        print('Saving Model Checkpoint ...')
        ckp = self.model.state_dict()
        torch.save(ckp, 'checkpoint.pt')
        print(f'Epoch: {epoch + 1}, Training checkpoint saved at checkpoint.pt')

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)


def main(args):
    train_dataloader, test_dataloader = load_dataloader(args.batch_size)
    model = AlexNet()
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    trainer = Trainer(model, train_dataloader, optimizer, args.device, args.save_every)
    trainer.train(max_epochs=args.total_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('For training model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of training loader')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--device', type=str,
                        default='cpu', help='device')
    parser.add_argument('--save_every', type=int, default=10,
                        help='frequency of saving model')
    parser.add_argument('--total_epochs', type=int, default=10,
                        help='num of training epochs')

    args = parser.parse_args()

    start_time = time.time()  # strat
    main(args)
    end_time = time.time()  # end
    duration = end_time - start_time

    # Convert duration to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Training completed in {hours} hours, {minutes} minutes, and {seconds} seconds.")
    # Training completed in 0 hours, 11 minutes, and 59 seconds.
