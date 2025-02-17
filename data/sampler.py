import torch
from torch.utils.data.sampler import Sampler
import linklink as link
import math
import numpy as np


class DistributedSampler(Sampler):

    def __init__(self, dataset, world_size=None, rank=None, round_up=True):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_up = round_up
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        if self.round_up:
            self.total_size = self.num_samples * self.world_size
            self.length = self.num_samples
        else:
            self.total_size = len(self.dataset)

        if self.rank < self.world_size - 1:
            self.length = self.num_samples
        else:
            self.length = self.total_size - (self.world_size - 1) * self.num_samples

    def __iter__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=0):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[self.last_iter * self.batch_size])
        else:
            raise RuntimeError("thid sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size


class DistributedEpochSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=0):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.all_size_single = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[self.last_iter * self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def get_one_epoch_self_part(self):
        num = len(self.dataset)
        indices = np.arange(num)
        extra_indices = np.random.choice(num, self.extra_per_epoch, )

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        self.all_num = self.total_iter * self.batch_size * self.world_size
        iter_per_epoch = (len(self.dataset) - 1) // (self.batch_size * self.world_size) + 1
        self.num_per_epoch = iter_per_epoch * self.batch_size * self.world_size
        self.extra_per_epoch = self.num_per_epoch - len(self.dataset)
        repeat = (self.all_num - 1) // self.num_per_epoch + 1
        indices = []
        for i in range(repeat):
            indice = self.get_one_epoch_self_part()
            indices.append(indice)

        indices = np.concatenate(indices)
        indices = indices[:self.all_size_single]

        assert len(indices) == self.all_size_single

        return indices

    def __len__(self):
        return self.all_size_single
