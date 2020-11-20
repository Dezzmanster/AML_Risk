import time
import numpy as np

# Standard Libraries
import os
import glob
import shutil
import nvidia_smi

# External Dependencies
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
import dask
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

from pathlib import Path
import pandas as pd

print('Dask Version:', dask.__version__)
print('Dask cuDF Version:', dask_cudf.__version__)
print()

# NVTabular
import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io import Shuffle
from nvtabular.utils import device_mem_size

def set_cluster_client(n_gpus=-1, device_spill_frac=0.8):
        # TODO: Check for any solution. If user calls this function, for the second call the correct recreation will fail. 
        # New cluster can be created after 'kernel restart' procedure.
        '''
        device_spill_frac: Spill GPU-Worker memory to host at this limit. Reduce if spilling fails to prevent device memory errors.
        '''
        # Deploy a Single-Machine Multi-GPU Cluster
        if n_gpus == -1:
            nvidia_smi.nvmlInit()
            n_gpus_avail = nvidia_smi.nvmlDeviceGetCount()
            print('\n n_gpus_avail: {}'.format(n_gpus_avail))
            n_gpus = n_gpus_avail
        # Delect devices to place workers
        visible_devices = [i for i in list(range(n_gpus))]
        visible_devices = str(visible_devices)[1:-1]
        #print('visible_devices: {}'.format(visible_devices))
        
        
        #TODO: how to reinitialzed cluster
        cluster = LocalCUDACluster(
            protocol = "tcp", # "tcp" or "ucx"
            CUDA_VISIBLE_DEVICES = visible_devices,
            device_memory_limit = device_spill_frac * device_mem_size(kind="total"),
        )
        
        try:
            # Create the distributed client
            client = Client(cluster)
            display(client)
            print('\n Dashboard avail: http://localhost:8888/proxy/8787/status')
            # Initialize RMM pool on ALL workers
            def _rmm_pool():
                rmm.reinitialize(
                    pool_allocator=True,
                    initial_pool_size=None, # Use default size
                )         
            client.run(_rmm_pool)
            return client
        
        except MemoryError:
            print('\n The client is already initialized')
            return None