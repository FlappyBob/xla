import args_parse

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--test_only_at_end': {
        'action': 'store_true',
    },
    '--persistent_workers': {
        'action': 'store_true',
    },
    '--prefetch_factor': {
        'type': int,
    },
    '--loader_prefetch_size': {
        'type': int,
    },
    '--device_prefetch_size': {
        'type': int,
    },
    '--host_to_device_transfer_threads': {
        'type': int,
    },
    '--sharding': {
        'choices': ['batch', 'spatial', 'conv', 'linear'],
        'nargs': '+',
        'default': [],
    },
    '--use_gradient_checkpointing': {
        'action': 'store_true',
    }
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
    opts=MODEL_OPTS.items(),
)

import os
import schedulers
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp.wrap import (recursive_wrap,
                                             transformer_auto_wrap_policy)
from torch_xla.distributed.fsdp.utils import checkpoint_module
from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.spmd as xs

xr.use_spmd(auto=FLAGS.auto_spmd)

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
    persistent_workers=False,
    prefetch_factor=16,
    loader_prefetch_size=8,
    device_prefetch_size=4,
    num_workers=8,
    host_to_device_transfer_threads=1,
)