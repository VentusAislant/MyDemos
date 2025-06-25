####################################################
## a model trained on Multi-GPU with DDP
## but without torchrun
##
## date: Mar 26 2024
## author: Aislant Ventus (张浩杰)
####################################################
import time

import torch
from torch.utils.data import DataLoader
from torch import nn
from alexnet import AlexNet
from datasets import load_dataset
import argparse

# extra import
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(rank, world_size):
    """
    :param rank: Unique identifier of each process
    :param world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    # to get a process for each GPU
    init_process_group(backend='nccl', world_size=world_size, rank=rank)


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
        # Add DDP
        self.model = DDP(module=model.to(gpu_id), device_ids=[self.gpu_id])
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
        # DDP
        for b_id, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            if b_id % 10 == 0 or b_id == 0 or b_id == len(self.train_data) - 1:
                print(f'Step {b_id}, loss: {loss:.3f}')

    def _save_checkpoint(self, epoch):
        # Get model from ddp-wrapped-object
        ckp = self.model.module.state_dict()
        torch.save(ckp, 'checkpoint.pt')
        print(f'Epoch: {epoch + 1}, Training checkpoint saved at checkpoint.pt')

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and (epoch + 1) % self.save_every == 0:
                # only save the ckp on rank 0, because all the rank has same ckp
                self._save_checkpoint(epoch)


def main(rank: int, world_size: int, args):
    try:
        # setup ddp
        ddp_setup(rank, world_size)

        train_dataset, _ = load_dataset()
        # Distributed Sampler for DataLoder
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            # pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(dataset=train_dataset)
        )
        model = AlexNet()
        optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
        trainer = Trainer(model, train_dataloader, optimizer, rank, args.save_every)
        trainer.train(max_epochs=args.total_epochs)
    except Exception as e:
        print(e)
    finally:
        # destroy ddp processes
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('For training model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of training loader')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--save_every', type=int, default=10,
                        help='frequency of saving model')
    parser.add_argument('--total_epochs', type=int, default=10,
                        help='num of training epochs')

    args = parser.parse_args()
    world_size = torch.cuda.device_count()

    start_time = time.time()  # strat
    mp.spawn(main, args=(world_size, args,), nprocs=world_size)

    end_time = time.time()  # end
    duration = end_time - start_time

    # Convert duration to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    print(f"Training completed in {hours} hours, {minutes} minutes, and {seconds} seconds.")
    # Training completed in 1 hours, 56 minutes, and 22 seconds.
