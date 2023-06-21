
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm
import utils

from utils import thread_wrapped_func


def producer(q, ):

#### Entry point

def run(q, args, device):
    th.cuda.set_device(device)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--mps', type=str, default='0')
    args = argparser.parse_args()

    device = th.device('cuda:%d' % args.gpu)
    mps = list(map(str, args.mps.split(',')))

    # If MPS values are given, then setup MPS
    if float(mps[0]) != 0:
        user_id = utils.mps_get_user_id()
        utils.mps_daemon_start()
        utils.mps_server_start(user_id)
        server_pid = utils.mps_get_server_pid()
        time.sleep(4)

    ctx = mp.get_context('spawn')

    if float(mps[0]) != 0:
        utils.mps_set_active_thread_percentage(server_pid, mps[0])
        # Just in case we add a timer to make sure MPS setup is done before we launch producer
        time.sleep(4)

    # TODO: shared structure declarations can be futher simplified
    q = ctx.SimpleQueue()

    # Synchornization signals
    event1 = ctx.Event()
    event2 = ctx.Event()

    print("Producer Start")
    producer_inst = ctx.Process(target=producer,
                    args=(q, device))
    producer_inst.start()

    if float(mps[0]) != 0:
        # Just in case we add timers to make sure MPS setup is done before we launch training
        time.sleep(8)
        utils.mps_set_active_thread_percentage(server_pid, mps[1])
        time.sleep(4)

    print("Run Start")
    p = mp.Process(target=thread_wrapped_func(run),
                    args=(q, args, device))
    p.start()

    p.join()
    producer_inst.join()
