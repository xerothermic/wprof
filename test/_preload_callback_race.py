"""Forkserver preload module for test_pytorch_callback_race.

Initializes CUDA in the forkserver process so it shows up in nvidia-smi
and behaves like the production icvr_launcher forkserver, where torch is
preloaded and CUDA is initialized before workers fork off.
"""
import torch
torch.zeros(1, device='cuda')
