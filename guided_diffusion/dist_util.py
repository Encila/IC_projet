"""
Helpers for distributed training.
"""

import io
import os
import socket
import sys
import platform

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    
    # On Windows, skip distributed initialization for single-GPU training
    # as libuv is not available in standard PyTorch builds
    if platform.system() == "Windows" and MPI.COMM_WORLD.Get_size() == 1:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        return
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def get_rank():
    """
    Get the rank of the current process.
    Returns 0 if distributed is not initialized (single GPU training).
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    For local paths, uses standard file I/O. For cloud paths, uses blobfile.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    rank = MPI.COMM_WORLD.Get_rank()
    
    # Use standard file I/O for local paths, blobfile for cloud paths
    is_local = not (path.startswith("gs://") or path.startswith("s3://") or path.startswith("az://"))
    
    if rank == 0:
        try:
            if is_local:
                with open(path, "rb") as f:
                    data = f.read()
            else:
                with bf.BlobFile(path, "rb") as f:
                    data = f.read()
        except Exception as e:
            # If local open fails, try blobfile as fallback
            if is_local:
                with bf.BlobFile(path, "rb") as f:
                    data = f.read()
            else:
                raise
                
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    # Skip if distributed is not initialized (single GPU training)
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
