import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
import tqdm
import utils

from load_graph import load_reddit, inductive_split,SAGE


def run(args, device,g):
    th.cuda.set_device(device)
    
    train_g, val_g, test_g = inductive_split(g)
    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    train_g = train_g.to(device)
    train_nid = train_nid.to(device)
    train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
    train_labels = val_labels = test_labels = g.ndata.pop('labels')
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    model = SAGE(train_nfeat.shape[1], args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop dataloader to sample the computation dependency graph as a list of blocks.
        for step, (input_nodes, seeds, blocks_next) in enumerate(dataloader):
            tic_step = time.time()

            blocks_temp = blocks_next # 训练的同时采样. 训练完了获得采样的结果. 
            blocks = [block.int().to(device) for block in blocks_temp]
            batch_feats =  train_nfeat[input_nodes].to(device)
            batch_labels =  train_labels[seeds].to(device)
            batch_pred = model(blocks, batch_feats)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            # epochtime = (time.time() - tic_step)
            # iter_tput.append(len(seeds) / epochtime)
            # if step % args.log_every == 0:
            #     print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
            #         epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

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
    g, n_classes = load_reddit()

    print("Run Start")
    run(args, device,g)
    # p = mp.Process(target=run,
    #                 args=(args, device,g))
    # p.start()

    # p.join()
