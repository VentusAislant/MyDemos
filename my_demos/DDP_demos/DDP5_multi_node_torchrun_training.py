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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup():
    init_process_group(backend='nccl')


# Trainer for training the model
class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,  # ferquency of saving
            snapshot_path: str,
    ):
        # torch run
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])

        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print('Loading snapshot')
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        print(f'Resuming training from snapshot at epoch {self.epochs_run}')

    def _run_batch(self, sources, targets):
        self.optimizer.zero_grad()
        output = self.model(sources)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        if self.local_rank == 0:
            print('=' * 90)
            print(f'global rank: {self.global_rank}, local rank: {self.local_rank}, # of Epoch: {epoch + 1},'
                  f'Batch size: {b_sz} , Steps: {len(self.train_data)}')
            print('=' * 90)
        # DDP
        for b_id, (source, targets) in enumerate(self.train_data):
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            loss = self._run_batch(source, targets)
            if self.local_rank == 0 and (b_id % 1000 == 0 or b_id == len(self.train_data) - 1):
                print(f'Step {b_id}, loss: {loss:.3f}')

    def _save_checkpoint(self, epoch):
        # Get model from ddp-wrapped-object
        snapshot = {}
        snapshot['MODEL_STATE'] = self.model.module.state_dict()
        snapshot['EPOCHS_RUN'] = epoch
        torch.save(snapshot, 'snapshot.pt')
        print(f'Epoch: {epoch + 1}, Training checkpoint saved at snapshot.pt')

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0 and (epoch + 1) % self.save_every == 0:
                # only save the ckp on rank 0, because all the rank has same ckp
                self._save_checkpoint(epoch)

            torch.distributed.barrier()


def main(args):
    try:
        # setup ddp
        ddp_setup()

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
        trainer = Trainer(model, train_dataloader, optimizer, args.save_every, args.snapshot_path)
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
    parser.add_argument('--snapshot_path', type=str, default='snapshot.pt',
                        help='path to snapshot')

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
    #
