"""Util file for Equinv scripts"""

import math
import resource
from pathlib import Path

import numpy as np
import torch
#from openbabel import pybel
from rdkit import RDLogger
from torchmetrics import MetricCollection
from tqdm import tqdm
from ..comparm import GP

def disable_lib_stdout():
    #pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except Exception:
            print("Limit change unsuccessful. Using torch file_system file sharing strategy instead.")

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


