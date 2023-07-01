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


from utils import thread_wrapped_func
from load_graph import load_reddit, inductive_split,load_ogb,SAGE

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def producer(q, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2, train_nfeat, train_labels, feat_dimension, label_dimension, device):
    th.cuda.set_device(device)

    # Map input tensors into GPU address
    # train_nfeat = train_nfeat.to(device="unified")
    # train_labels = train_labels.to(device="unified")
    feat_handler = dgl.utils.pin_memory_inplace(train_nfeat)
    labels_handler = dgl.utils.pin_memory_inplace(train_labels)

    # Create GPU-side ping pong buffers
    in_feat1 = th.zeros(feat_dimension, device=device)
    in_feat2 = th.zeros(feat_dimension, device=device)
    in_label1 = th.zeros(label_dimension, dtype=th.long, device=device)
    in_label2 = th.zeros(label_dimension, dtype=th.long, device=device)

    # Termination signal
    running = th.ones(1, dtype=th.bool)

    # Share with the training process
    q.put((in_feat1, in_feat2, in_label1, in_label2, running))
    print("Allocation done")

    flag = 1

    with th.no_grad():
        while(1):
            event1.wait()
            event1.clear()
            if not running:
                break
            if flag:
                # th.index_select(train_nfeat, 0, idxf1[0:idxf1_len].to(device=device), out=in_feat1[0:idxf1_len])
                # th.index_select(train_labels, 0, idxl1[0:idxl1_len].to(device=device), out=in_label1[0:idxl1_len])
                in_feat1 =  dgl.utils.gather_pinned_tensor_rows(train_nfeat,  idxf1[0:idxf1_len].to(device=device))
                in_label1 = dgl.utils.gather_pinned_tensor_rows(train_labels, idxl1[0:idxl1_len].to(device=device))
            else:
                # th.index_select(train_nfeat, 0, idxf2[0:idxf2_len].to(device=device), out=in_feat2[0:idxf2_len])
                # th.index_select(train_labels, 0, idxl2[0:idxl2_len].to(device=device), out=in_label2[0:idxl2_len])
                in_feat2 =  dgl.utils.gather_pinned_tensor_rows(train_nfeat,  idxf2[0:idxf2_len].to(device=device))
                in_label2 = dgl.utils.gather_pinned_tensor_rows(train_labels, idxl2[0:idxl2_len].to(device=device))
            flag = (flag == False)
            # print("one mini batch gather done")
            th.cuda.synchronize()
            event2.set()


def run(q, args, device, data, in_feats, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2):
    th.cuda.set_device(device)
    n_classes, train_g, val_g, test_g = data

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
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    in_feat1, in_feat2, in_label1, in_label2, finish = q.get()

    # A prologue for the pipelining purpose, just for the first minibatch of the first epoch
    # ------------------------------------------------------
    flag = True
    input_nodes, seeds, blocks_next = next(iter(dataloader))

    # Send node indices for the next minibatch to the producer
    if flag:
        idxf1[0:len(input_nodes)].copy_(input_nodes)
        idxl1[0:len(seeds)].copy_(seeds)
        idxf1_len.fill_(len(input_nodes))
        idxl1_len.fill_(len(seeds))
    else:
        idxf2[0:len(input_nodes)].copy_(input_nodes)
        idxl2[0:len(seeds)].copy_(seeds)
        idxf2_len.fill_(len(input_nodes))
        idxl2_len.fill_(len(seeds))
    event1.set()
    time.sleep(1)

    input_nodes_n = len(input_nodes)
    seeds_n = len(seeds)
    flag = (flag == False)
    blocks_temp = blocks_next
    # ------------------------------------------------------
    # Prologue done


    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop dataloader to sample the computation dependency graph as a list of blocks.
        for step, (input_nodes, seeds, blocks_next) in enumerate(dataloader):
            tic_step = time.time()

            # Send node indices for the next minibatch to the producer
            if flag:
                idxf1[0:len(input_nodes)].copy_(input_nodes)
                idxl1[0:len(seeds)].copy_(seeds)
                idxf1_len.fill_(len(input_nodes))
                idxl1_len.fill_(len(seeds))
            else:
                idxf2[0:len(input_nodes)].copy_(input_nodes)
                idxl2[0:len(seeds)].copy_(seeds)
                idxf2_len.fill_(len(input_nodes))
                idxl2_len.fill_(len(seeds))

            event1.set()

            event2.wait() # wait index select,gather features to GPU
            event2.clear()

            # Load the input features as well as output labels
            if not flag:
                batch_feats = in_feat1[0:input_nodes_n]
                batch_labels = in_label1[0:seeds_n]
            else:
                batch_feats = in_feat2[0:input_nodes_n]
                batch_labels = in_label2[0:seeds_n]

            blocks = [block.int().to(device) for block in blocks_temp]

            batch_pred = model(blocks, batch_feats)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flag = (flag == False)
            input_nodes_n = len(input_nodes)
            seeds_n = len(seeds)

            blocks_temp = blocks_next # 训练的同时采样. 训练完了获得采样的结果. 
            epochtime = (time.time() - tic_step)
            iter_tput.append(len(seeds) / epochtime)
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))

        # A prologue for the next epoch
        # ------------------------------------------------------
        input_nodes, seeds, blocks_next = next(iter(dataloader))

        if flag:
            idxf1[0:len(input_nodes)].copy_(input_nodes)
            idxl1[0:len(seeds)].copy_(seeds)
            idxf1_len.fill_(len(input_nodes))
            idxl1_len.fill_(len(seeds))
        else:
            idxf2[0:len(input_nodes)].copy_(input_nodes)
            idxl2[0:len(seeds)].copy_(seeds)
            idxf2_len.fill_(len(input_nodes))
            idxl2_len.fill_(len(seeds))
        event1.set()

        event2.wait()
        event2.clear()

        # Load the input features as well as output labels
        if not flag:
            batch_feats = in_feat1[0:input_nodes_n]
            batch_labels = in_label1[0:seeds_n]
        else:
            batch_feats = in_feat2[0:input_nodes_n]
            batch_labels = in_label2[0:seeds_n]

        blocks = [block.int().to(device) for block in blocks_temp]
        batch_pred = model(blocks, batch_feats)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        flag = (flag == False)
        input_nodes_n = len(input_nodes)
        seeds_n = len(seeds)
        blocks_temp = blocks_next
        # ------------------------------------------------------
        # Prologue done

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(
                model, val_g, feat, labels, val_nid, device,args)
            print('Eval Acc {:.4f}'.format(eval_acc))
            # test_acc = evaluate(
            #     model, test_g, feat, labels, test_nid, device,args)
            # print('Test Acc: {:.4f}'.format(test_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    # Send a termination signal to the producer
    finish.copy_(th.zeros(1, dtype=th.bool))
    event1.set()

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
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--mps', type=str, default='0')
    args = argparser.parse_args()

    device = th.device('cuda:0') 
    mps = list(map(str, args.mps.split(',')))

    # If MPS values are given, then setup MPS
    if float(mps[0]) != 0:
        user_id = utils.mps_get_user_id()
        utils.mps_daemon_start()
        utils.mps_server_start(user_id)
        server_pid = utils.mps_get_server_pid()
        time.sleep(4)
    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    data = n_classes, train_g, val_g, test_g

    feat = val_nfeat = test_nfeat = g.ndata.pop('features').share_memory_()
    labels = val_labels = test_labels = g.ndata.pop('labels').share_memory_()
    in_feats = feat.shape[1]

    fanout_max = 1
    for fanout in args.fan_out.split(','):
        fanout_max = fanout_max * int(fanout)

    feat_dimension = [args.batch_size * fanout_max, feat.shape[1]]
    label_dimension = [args.batch_size]

    ctx = mp.get_context('spawn')

    if float(mps[0]) != 0:
        utils.mps_set_active_thread_percentage(server_pid, mps[0])
        # Just in case, we make sure MPS setup is done before we launch producer
        time.sleep(4)

    # TODO: shared structure declarations can be futher simplified
    q = ctx.SimpleQueue()

    # Synchornization signals
    event1 = ctx.Event()
    event2 = ctx.Event()
    # idxf1 and idxf2 are used for two batch pipeline
    # Indices and the their lengths shared between the producer and the training processes
    idxf1 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxf2 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxl1 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxl2 = th.zeros([args.batch_size * fanout_max], dtype=th.long).share_memory_()
    idxf1_len = th.zeros([1], dtype=th.long).share_memory_()
    idxf2_len = th.zeros([1], dtype=th.long).share_memory_()
    idxl1_len = th.zeros([1], dtype=th.long).share_memory_()
    idxl2_len = th.zeros([1], dtype=th.long).share_memory_()

    print("Producer Start")
    producer_inst = ctx.Process(target=producer,
                    args=(q, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2, feat, labels, feat_dimension, label_dimension, device))
    producer_inst.start()

    if float(mps[0]) != 0:
        # Just in case we add timers to make sure MPS setup is done before we launch training
        time.sleep(8)
        utils.mps_set_active_thread_percentage(server_pid, mps[1])
        time.sleep(4)

    print("Run Start")
    p = ctx.Process(target=run,
                    args=(q, args, device, data, in_feats, idxf1, idxf2, idxl1, idxl2, idxf1_len, idxf2_len, idxl1_len, idxl2_len, event1, event2))
    p.start()

    p.join()
    producer_inst.join()
