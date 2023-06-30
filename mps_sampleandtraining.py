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
import functools
from load_graph import load_reddit, inductive_split,SAGE,compute_acc



def run(args, device,g):
    th.cuda.set_device(device)
    
    train_g, val_g, test_g = inductive_split(g)
    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    val_nid = val_mask.nonzero(as_tuple=False).squeeze()
    train_nid = train_mask.nonzero(as_tuple=False).squeeze()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    train_g = train_g.to(device)
    train_nid = train_nid.to(device)
    nfeat = g.ndata.pop('features')
    labels = g.ndata.pop('labels')
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    
    nfeat = nfeat.to(device="unified")
    fanout_max = functools.reduce(lambda x, y: x * int(y), args.fan_out.split(','), 1)

    model = SAGE(nfeat.shape[1], args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    feat_dimension = [args.batch_size * fanout_max, nfeat.shape[1]]
    in_feat1 = th.zeros(feat_dimension, device=device)
    for epoch in range(args.num_epochs):
        model.train()
        tic = time.time()

        for step, (input_nodes, seeds, blocks_next) in enumerate(dataloader):
            idxf1_len = len(input_nodes)
            blocks_temp = blocks_next 
            blocks = [block.int().to(device) for block in blocks_temp]
            batch_feats = th.index_select(nfeat, 0, input_nodes.to(device=device), out=in_feat1[0:idxf1_len])
            batch_labels = labels[seeds].to(device)
            batch_pred = model(blocks, batch_feats)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        model.eval()
        with th.no_grad():
            pred = model.inference(val_g, nfeat, device,args)
        eval_acc = compute_acc(pred[val_nid], labels[val_nid])
        print('Eval Acc {:.4f}'.format(eval_acc)) 


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=5)
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
