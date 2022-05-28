# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""
import os
import subprocess
import time

import torch.cuda


def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)


def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    gpu_count = torch.cuda.device_count()
    # gpu_count = 5
    n_gpus = os.getenv("CUDA_VISIBLE_DEVICES").split(',')
    # n_gpus = ['3', '4', '5', '6', '7']
    print('*' * 80)
    print(n_gpus)
    print('*' * 80)
    procs_by_gpu = [None] * gpu_count

    while len(commands) > 0:
        for idx in range(gpu_count):
            cur_gpu = n_gpus[idx]
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                print(f'CUDA_VISIBLE_DEVICES={cur_gpu} {cmd}')
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={cur_gpu} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}

try:
    from domainbed import facebook

    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
