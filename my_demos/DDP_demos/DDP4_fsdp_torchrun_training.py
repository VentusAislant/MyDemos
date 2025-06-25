####################################################
## a model trained on Multi-GPU with DDP
## but without torchrun
##
## date: Mar 26 2024
## author: Aislant Ventus (张浩杰)
####################################################

import argparse
import time
from typing import Type

from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, FullStateDictConfig, \
    StateDictType

import alexnet
from policies import fp32_policy, fpSixteen, bfSixteen, get_wrapper
from torch.distributed import init_process_group, destroy_process_group
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os
from tqdm import tqdm
from datasets import load_dataset


def myprint(*str, color='\033[35m'):
    print(color, end='')
    print(*str, end='\033[0m\n')


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_data: DataLoader,
            optimizer_cls: Type[torch.optim.Optimizer],
            lr: float,
            # fsdp config
            auto_wrap_policy,
            mixed_precision_policy,
            sharding_strategy,
            save_every: int,  # default setting of saving strategy is epoch
            snapshot_path: str,
            log_every: int = 100,  # log frequency (how many steps)
            log_strategy: str = 'epoch',  # ['epoch', 'step]
    ):
        # trochrun
        self.local_rank = int(os.environ['LOCAL_RANK'])  # locally gpu rank
        self.rank = int(os.environ['RANK'])  # global gpu rank

        self.train_data = train_data
        self.save_every = save_every
        self.log_every = log_every
        self.log_strategy = log_strategy

        os.makedirs(snapshot_path, exist_ok=True)
        self.snapshot_path = os.path.join(snapshot_path, 'snapshot.pt')

        self.epochs_run = 0
        self.model = model
        if os.path.exists(self.snapshot_path):
            myprint('Loading snapshot')
            self._load_snapshot(self.snapshot_path)

        torch.cuda.set_device(self.local_rank)
        self.model = FSDP(
            module=self.model,
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy
        )

        # optimize must be initialized after the model was wrapped with FSDP
        self.optimizer = optimizer_cls(params=model.parameters(), lr=lr)

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        myprint(f'Resuming training from snapshot at epoch {self.epochs_run}')

    def _run_batch(self, sources, targets):
        self.optimizer.zero_grad()
        output = self.model(sources)
        loss = nn.CrossEntropyLoss(reduction='sum')(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])

        if self.local_rank == 0:
            myprint('=' * 90)
            myprint(f'world size: {self.model.world_size}, print rank: {self.rank},',
                    f'# of Epoch: {epoch + 1}, Batch size: {b_sz} | Steps: {len(self.train_data)}')
            myprint('=' * 90)
            inner_pbar = tqdm(
                range(len(self.train_data)), colour="MAGENTA", desc="rank 0 Training Epoch"
            )

        epoch_loss = torch.zeros(2).to(self.local_rank)
        for b_id, (sources, targets) in enumerate(self.train_data):
            sources, targets = sources.to(self.local_rank), targets.to(self.local_rank)
            loss = self._run_batch(sources, targets)

            epoch_loss[0] += loss
            epoch_loss[1] += len(sources)

            if self.local_rank == 0:
                inner_pbar.update(1)

            if self.log_strategy == 'step' and (b_id % self.log_every == 0
                                                or b_id == len(self.train_data) - 1):
                myprint('{', f'"Step": {b_id + 1}, "loss": {loss / len(sources):.3f}', '}')

        # sum up loss on all processes
        torch.distributed.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.SUM)
        return epoch_loss[0] / epoch_loss[1]

    def _save_checkpoint(self, epoch):

        ckp = self.model.state_dict()

        if self.rank == 0:  # only save one ckp on master machine
            snapshot = {}
            snapshot['EPOCHS_RUN'] = epoch + 1

            snapshot['MODEL_STATE'] = ckp
            myprint('Saving Model Checkpoint ...')
            torch.save(snapshot, self.snapshot_path)
            myprint(f'Epoch: {epoch + 1}, Training Checkpoint saved at {self.snapshot_path}')

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            loss = self._run_epoch(epoch)
            if self.log_strategy == 'epoch' and self.local_rank == 0:
                myprint('{', f'"Epoch": {epoch + 1}, "Total Loss": {loss:.3f}', '}')
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)

            # make sure all ranks is done
            torch.distributed.barrier()


def ddp_steup():
    init_process_group(backend='nccl')


def ddp_cleanup():
    destroy_process_group()


def get_sharding_strategy(policy: str) -> ShardingStrategy:
    if policy == 'zero2':
        return ShardingStrategy.SHARD_GRAD_OP
    elif policy == 'zero3':
        return ShardingStrategy.SHARD_GRAD_OP
    elif policy == 'zero1':
        return ShardingStrategy.NO_SHARD


def get_mixed_precision(policy: str) -> MixedPrecision:
    if policy == 'fp16':
        return fpSixteen
    elif policy == 'bf16':
        return bfSixteen
    elif policy == 'fp32':
        return fp32_policy


def main(args):
    ddp_steup()

    model = alexnet.AlexNet()

    optimizer_cls = torch.optim.Adam

    dataset, _ = load_dataset()

    # Distributed Sampler for DataLoader
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset=dataset),
        num_workers=2,
    )

    # fsdp_config
    sharding_strategy = get_sharding_strategy(args.sharding_strategy)
    mixed_precision = get_mixed_precision(args.mixed_precision)
    auto_wrap_policy = get_wrapper()

    trainer = Trainer(
        model=model,
        train_data=train_dataloader,
        optimizer_cls=optimizer_cls,
        lr=args.lr,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision_policy=mixed_precision,
        sharding_strategy=sharding_strategy,
        save_every=args.save_every,
        snapshot_path=args.output_dir,
        log_every=args.log_every,
        log_strategy=args.log_strategy,
    )

    trainer.train(max_epochs=args.num_epochs)

    ddp_cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser('For training model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size of training loader')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--save_every', type=int, default=3,
                        help='frequency of saving model (epochs)')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='number of training epochs')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logging during training (steps)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/',
                        help='directory to save trained model')
    parser.add_argument('--sharding_strategy', type=str, choices=['zero1', 'zero2', 'zero3'],
                        default='zero3', help='sharding strategy for FSDP')
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'bf16', 'fp32'],
                        default='fp32', help='mixed precision training')
    parser.add_argument('--log_strategy', type=str, choices=['step', 'epoch'],
                        default='epoch',
                        help='Logging strategy for training progress. Choose between "step" to log at'
                             ' <save_every> training step, or "epoch" to log at <save_every> each epoch. ')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    start_time = time.time()  # strat

    main(args)

    end_time = time.time()  # end
    duration = end_time - start_time

    # Convert duration to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    myprint(f"Training completed in {hours} hours, {minutes} minutes, and {seconds} seconds.")
    # Training completed in 0 hours, 3 minutes, and 13 seconds.
    myprint('Completed!')
