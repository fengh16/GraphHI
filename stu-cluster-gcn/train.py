#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
import time

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from models import GCN
import torch.nn as nn

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)
import torch
import dgl

import random


def st_loss(logits_S, logits_T, temperature=1.0):
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1)
    temp_multiplier = temperature ** 2
    if isinstance(temperature, float) or isinstance(temperature, int):
        return loss.mean() * (temperature ** 2)
    else:
        temp_multiplier = temp_multiplier.reshape(-1)
        return (loss * temp_multiplier).mean()


def st_loss_yelp(logits_S, logits_T, temperature=1.0):
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = beta_logits_T.sigmoid()
    loss = nn.BCEWithLogitsLoss(reduction='none')(beta_logits_S, p_T).mean(dim=-1)
    temp_multiplier = temperature ** 2
    if isinstance(temperature, float) or isinstance(temperature, int):
        return loss.mean() * (temperature ** 2)
    else:
        temp_multiplier = temp_multiplier.reshape(-1)
        return (loss * temp_multiplier).mean()


def cl_loss(emb_S, emb_T, GraphHI_CL_sample_num=10, temperature=1.0, delta=0):
    out_CL = F.normalize(emb_S)
    t_out_CL = F.normalize(emb_T).detach()
    num_out_nodes = out_CL.shape[0]
    neg_sample_ids = torch.multinomial(torch.ones(num_out_nodes), GraphHI_CL_sample_num * num_out_nodes, replacement=True).reshape(num_out_nodes, GraphHI_CL_sample_num)
    similarity_self = (out_CL * t_out_CL).sum(dim=-1)
    neg_data = t_out_CL[neg_sample_ids]
    neg_similarity = (out_CL.unsqueeze(1) * neg_data).sum(-1)
    similarity_self /= temperature
    neg_similarity /= temperature
    CL_result = -torch.log(torch.exp(similarity_self) / (torch.exp(similarity_self) + torch.exp(neg_similarity).sum(-1)))
    
    neg_sample_ids2 = torch.multinomial(torch.ones(num_out_nodes), GraphHI_CL_sample_num * num_out_nodes, replacement=True).reshape(num_out_nodes, GraphHI_CL_sample_num)
    similarity_self2 = (out_CL * out_CL).sum(dim=-1)
    neg_data2 = out_CL[neg_sample_ids2]
    neg_similarity2 = (out_CL.unsqueeze(1) * neg_data2).sum(-1)
    similarity_self2 /= temperature
    neg_similarity2 /= temperature
    CL_result2 = -torch.log(torch.exp(similarity_self2) / (torch.exp(similarity_self2) + torch.exp(neg_similarity2).sum(-1)))
    return CL_result.mean() + CL_result2.mean() * delta


def gen_model(args, CL_dim=None):
    if args.use_labels:
        model = GCN(
            in_feats + n_classes, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear, CL_dim=CL_dim
        )
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear, CL_dim=CL_dim)
    return model


def cross_entropy(x, labels):
    if labels.dim() != 1 and labels.shape[1] != 1:  # for Yelp
        return nn.BCEWithLogitsLoss()(x, labels)
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator=None):
    if evaluator is None:
        y_pred = pred > 0
        y_true = labels > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    if labels.shape[1] != 1: ## multi-class label like Yelp
        onehot = th.zeros([feat.shape[0], n_classes]).to(device)
        onehot[idx] = labels[idx]
    else: ## arxiv
        onehot = th.zeros([feat.shape[0], n_classes]).to(device)
        onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels, nargs, *args):
    global alpha_ALC_rewards, alpha_chosen_index, alpha_CL, alpha_ST, alpha_CE 
    
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    
    if nargs.role == 'GraphHI':
        tea_logits, tea_emb, loss_dis, epoch = args
        label_loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        modify_epoch = nargs.GraphHI_dynamic_coefficient_after_epoches
        if modify_epoch is not None:
            if 0 < modify_epoch < 1:
                modify_epoch = int(nargs.n_epochs * nargs.GraphHI_dynamic_coefficient_after_epoches)
            if epoch <= modify_epoch:
                alpha_CL, alpha_ST, alpha_CE = alpha_base
        logits = pred
        with torch.no_grad():
            temperature = (nargs.GraphHI_attention_temp_range_max - (nargs.GraphHI_attention_temp_range_max - nargs.GraphHI_attention_temp_range_min) * ((logits.max(dim=-1)[0] * nargs.GraphHI_attention_temp_w + nargs.GraphHI_attention_temp_b).sigmoid())).view(-1, 1).detach()
            from dgl.nn.pytorch import GraphConv
            GNN_layer = GraphConv(in_feats=1, out_feats=1, norm='both', weight=False, bias=False, activation=None)
            temperature = (GNN_layer(graph, temperature) * nargs.GraphHI_attention_temp_smooth_addition_new_weight + temperature) / (nargs.GraphHI_attention_temp_smooth_addition_new_weight + 1)
            need_change_index = tea_logits.argmax(dim=-1).reshape(-1,1) != labels.reshape(-1,1)
            mask_not_train = torch.ones_like(need_change_index, dtype=torch.bool).to(feat.device)
            mask_not_train[train_pred_idx] = False
            need_change_index[mask_not_train] = False
            temperature[need_change_index] = nargs.GraphHI_attention_temp_range_max
        loss = alpha_CE * label_loss + alpha_ST * st_loss(pred, tea_logits, temperature=temperature)
        loss = loss + alpha_CL * cl_loss(model.CL_linear(model.emb), tea_emb, GraphHI_CL_sample_num=nargs.GraphHI_CL_sample_num, temperature=nargs.GraphHI_CL_temp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, pred
    else:
        loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        loss.backward()
        optimizer.step()
        return loss, pred


# for yelp and products
def cluster_train(model, cluster, feat, labels, mask, optimizer, args, *others):
    global alpha_ALC_rewards, alpha_chosen_index, alpha_CL, alpha_ST, alpha_CE 
    model.train()
    train_idx = torch.arange(cluster.num_nodes())[mask].to(feat.device)
    graph = cluster
    if args.use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate
        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate
        train_pred_idx = train_idx[mask]
    
    optimizer.zero_grad()
    pred = model(cluster, feat)
    if args.role == 'GraphHI':
        tea_logits, tea_emb, loss_dis, epoch = others
        label_loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        modify_epoch = args.GraphHI_dynamic_coefficient_after_epoches
        if modify_epoch is not None:
            if 0 < modify_epoch < 1:
                modify_epoch = int(args.n_epochs * args.GraphHI_dynamic_coefficient_after_epoches)
            if epoch <= modify_epoch:
                alpha_CL, alpha_ST, alpha_CE = alpha_base
        logits = pred
        nargs = args
        with torch.no_grad():
            if nargs.dataset == 'yelp':
                metric = torch.mean(torch.max(logits, 1-logits), dim=-1)
            else:
                metric = (logits.max(dim=-1)[0] * nargs.GraphHI_attention_temp_w + nargs.GraphHI_attention_temp_b).sigmoid()
            temperature = (nargs.GraphHI_attention_temp_range_max - (nargs.GraphHI_attention_temp_range_max - nargs.GraphHI_attention_temp_range_min) * metric).view(-1, 1).detach()
            from dgl.nn.pytorch import GraphConv
            GNN_layer = GraphConv(in_feats=1, out_feats=1, norm='both', weight=False, bias=False, activation=None)
            temperature = (GNN_layer(cluster, temperature) * nargs.GraphHI_attention_temp_smooth_addition_new_weight + temperature) / (nargs.GraphHI_attention_temp_smooth_addition_new_weight + 1)
            if nargs.dataset == 'yelp':
                need_change_index = torch.sum((tea_logits > 0) == (labels > 0), dim=-1) < (labels.shape[-1] + 1) // 2
            else:
                need_change_index = tea_logits.argmax(dim=-1).reshape(-1,1) != labels.reshape(-1,1)
            mask_not_train = torch.ones_like(need_change_index, dtype=torch.bool).to(feat.device)
            mask_not_train[train_pred_idx] = False
            need_change_index[mask_not_train] = False
            temperature[need_change_index] = nargs.GraphHI_attention_temp_range_max
        if args.dataset == 'yelp':
            loss = alpha_CE * label_loss + alpha_ST * st_loss_yelp(pred, tea_logits, temperature=temperature)
        else:
            loss = alpha_CE * label_loss + alpha_ST * st_loss(pred, tea_logits, temperature=temperature)
        loss = loss + alpha_CL * cl_loss(model.CL_linear(model.emb), tea_emb, GraphHI_CL_sample_num=args.GraphHI_CL_sample_num, temperature=args.GraphHI_CL_temp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, pred
    else:
        loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        loss.backward()
        optimizer.step()
        return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()
    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


@th.no_grad()
def cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, use_labels, evaluator, feat, data_name):
    model.eval()
    cuda = False if data_name == "ogbn-products" else True

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    perms = []
    preds = []
    for cluster in cluster_iterator:
        cluster = cluster.int().to(device)
        input_nodes = cluster.ndata['id']
        batch_feat = feat[input_nodes]
        pred = model(cluster, batch_feat) if cuda else model(cluster, batch_feat).cpu()
        perms.append(input_nodes)
        preds.append(pred)
    perm = th.cat(perms, dim=0)
    pred = th.cat(preds, dim=0)
    inv_perm=perm.sort()[1]
    if not cuda:
        inv_perm = inv_perm.cpu()
        pred = pred[inv_perm]
        labels = labels.cpu()
        train_idx, val_idx, test_idx = train_idx.cpu(), val_idx.cpu(), test_idx.cpu()
    else:
        pred = pred[inv_perm]
    

    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    global alpha_ALC_rewards, alpha_chosen_index, alpha_CL, alpha_ST, alpha_CE, alpha_base
    
    # Get teacher knowledge
    import os
    if args.dataset == 'ogbn-arxiv':
        inter_model_path = os.path.join(args.inter_model_dir, f'arxiv-knowledge.pth.tar')
    elif args.dataset == 'ogbn-products':
        inter_model_path = os.path.join(args.inter_model_dir, f'products-knowledge.pth.tar')
    elif args.dataset == 'yelp':
        inter_model_path = os.path.join(args.inter_model_dir, f'yelp-knowledge.pth.tar')
    assert os.path.isfile(inter_model_path), "Please download teacher knowledge first"
    knowledge = th.load(inter_model_path, map_location=device)
    tea_logits = knowledge['logits']
    tea_emb = knowledge['embedding']
    if 'perm' in knowledge.keys():
        perm = knowledge['perm']
        inv_perm = perm.sort()[1]
        tea_logits = tea_logits[inv_perm]
        tea_emb = tea_emb[inv_perm]
    print(f'Teacher Test ACC: {compute_acc(tea_logits[test_idx], labels[test_idx], evaluator)}')

    CL_dim = tea_emb.shape[-1]

    # define model and optimizer
    model = gen_model(args, CL_dim=CL_dim)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    loss_dis = torch.nn.BCELoss()

    if args.role == 'GraphHI':
        model_param_count = sum(param.numel() for param in model.parameters())
        if model.CL_dim == None:
            CL_transfer_param_count = 0
        else:
            CL_transfer_param_count = sum(param.numel() for param in model.CL_linear.parameters())
            model_param_count -= CL_transfer_param_count
        print("#Params in the student model:", model_param_count)


    #+++++++++++++++++++++++++++++++++++++++++++++++
    # Create DataLoader for constructing blocks
    from functools import partial
    from torch.utils.data import DataLoader
    from sampler import ClusterIter, subgraph_collate_fn
    if args.dataset in ["ogbn-products", "yelp"]:
        nfeat = graph.ndata.pop('feat').to(device)
        cluster_iter_data = ClusterIter(args.dataset, graph, args.num_partitions) #'ogbn-products'
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[train_idx] = True
        graph.ndata['train_mask'] = mask
        graph.ndata['id'] = th.arange(graph.num_nodes())
        cluster_iterator = DataLoader(cluster_iter_data, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=0,
                                    collate_fn=partial(subgraph_collate_fn, graph))
    #+++++++++++++++++++++++++++++++++++++++++++++++
    graph = graph.int().to(device)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    alpha_ST = args.GraphHI_alpha
    alpha_CE = 1
    alpha_CL = args.GraphHI_CL_weight
    
    alpha_ALC_rewards = [[1] for _ in range(3)]
    alpha_chosen_index = 0
    alpha_original_ = [alpha_CL, alpha_ST, alpha_CE]
    alpha_CL, alpha_ST, alpha_CE = [i / sum(alpha_original_) * 3 for i in alpha_original_]
    
    alpha_base = [alpha_CL, alpha_ST, alpha_CE]

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        adjust_learning_rate(optimizer, args.lr, epoch)

        ## (๑>؂<๑) full batch GCN train arxiv
        if args.dataset in ["ogbn-arxiv"]:
            loss, pred = train(model, graph, labels, train_idx, optimizer, args.use_labels, args, tea_logits, tea_emb, loss_dis, epoch)
            acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)
            #+++++++++++++++++
            t_start = time.time()
            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
                model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
            )
            if epoch == 1:
                t_end = time.time()
                test_time = t_end - t_start
                print(f'inference time: {test_time * 1e3} ms')
            
            alpha_ALC_rewards[alpha_chosen_index].append(train_acc)
            if random.random() < args.GraphHI_dynamic_coefficient_epsilon:
                # explore
                alpha_chosen_index = random.randint(0, 3 - 1)
            else:
                # exploit
                alpha_original_rewards = [sum(reward) for reward in alpha_ALC_rewards]
                alpha_chosen_index = alpha_original_rewards.index(max(alpha_original_rewards))
            alpha_original_ = [alpha_CL, alpha_ST, alpha_CE]
            alpha_original_[alpha_chosen_index] *= args.GraphHI_dynamic_coefficient_gamma
            if args.GraphHI_dynamic_coefficient_max > 1:
                if alpha_original_[alpha_chosen_index] > args.GraphHI_dynamic_coefficient_max * alpha_base[alpha_chosen_index]:
                    alpha_original_[alpha_chosen_index] = args.GraphHI_dynamic_coefficient_max * alpha_base[alpha_chosen_index]
            alpha_CL, alpha_ST, alpha_CE = [i / sum(alpha_original_) * 3 for i in alpha_original_]
            # print("New weights", alpha_CL, alpha_ST, alpha_CE)
            
            #+++++++++++++++++
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_acc = test_acc
        ## mini batch cluster-GCN train products and yelp
        elif args.dataset in ["ogbn-products", "yelp"]:
            for step, cluster in enumerate(cluster_iterator):
                mask = cluster.ndata.pop('train_mask')
                if mask.sum() == 0:
                    continue
                cluster.edata.pop(dgl.EID)
                cluster = cluster.int().to(device)
                input_nodes = cluster.ndata['id']
                batch_feat = nfeat[input_nodes]
                batch_labels = labels[input_nodes]

                if args.role == 'vani':
                    loss, pred = cluster_train(model, cluster, batch_feat, batch_labels, mask, optimizer, args)
                elif args.role in ['GraphHI']:
                    loss, pred = cluster_train(model, cluster, batch_feat, batch_labels, mask, optimizer, args, tea_logits[input_nodes], tea_emb[input_nodes], loss_dis, epoch)
                if step % args.log_every == 0:
                    acc = compute_acc(pred[mask], batch_labels[mask], evaluator)
                    print(
                    f"Epoch: {epoch}/{args.n_epochs} | Loss: {loss.item():.4f} | {step:3d}-th Cluster Train Acc: {acc:.4f}")
            #+++++++++++++++++ testing OOM
            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator, nfeat, args.dataset)
            
            alpha_ALC_rewards[alpha_chosen_index].append(train_acc)
            if random.random() < args.GraphHI_dynamic_coefficient_epsilon:
                # explore
                alpha_chosen_index = random.randint(0, 3 - 1)
            else:
                # exploit
                alpha_original_rewards = [sum(reward) for reward in alpha_ALC_rewards]
                alpha_chosen_index = alpha_original_rewards.index(max(alpha_original_rewards))
            alpha_original_ = [alpha_CL, alpha_ST, alpha_CE]
            alpha_original_[alpha_chosen_index] *= args.GraphHI_dynamic_coefficient_gamma
            if args.GraphHI_dynamic_coefficient_max > 1:
                if alpha_original_[alpha_chosen_index] > args.GraphHI_dynamic_coefficient_max * alpha_base[alpha_chosen_index]:
                    alpha_original_[alpha_chosen_index] = args.GraphHI_dynamic_coefficient_max * alpha_base[alpha_chosen_index]
            alpha_CL, alpha_ST, alpha_CE = [i / sum(alpha_original_) * 3 for i in alpha_original_]
            
            if epoch == 1:
                t_start = time.time()
                train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator, nfeat, args.dataset)
                t_end = time.time()
                test_time = t_end - t_start
                print(f'inference time: {test_time * 1e3} ms')
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f} s. \n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
                )
            #+++++++++++++++++
            if epoch % args.log_every == 0:
                train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator, nfeat, args.dataset)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    final_test_acc = test_acc
        ##############################################

        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f} s. \n"
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

            for l, e in zip(
                [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
                [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

    
    print("*" * 50)
    print(args)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("GCN on OGBN-*", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1000, help="number of epochs")
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--use-linear", action="store_true", help="Use linear layer.")
    argparser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=256, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--seed", type=int, default=2022, help="random seed")
    argparser.add_argument("-d", "--dataset", type=str, default='ogbn-arxiv', help="Dataset name ('ogbn-products', 'ogbn-arxiv', 'yelp').")
    # extra added
    argparser.add_argument("--num_partitions", type=int, default=200, help="num of subgraphs")
    argparser.add_argument("--batch-size", type=int, default=32, help="batch size")
    argparser.add_argument("--role", type=str, default="vani", choices=['vani', 'GraphHI'])
    argparser.add_argument("--inter_model_dir", type=str, default='../inter_model')
    argparser.add_argument("--data_dir", type=str, default='../../dataset')

    argparser.add_argument('--GraphHI_alpha', type=float, default=0.5)
    argparser.add_argument('--GraphHI_CL_weight', type=float, default=1)
    argparser.add_argument('--GraphHI_CL_temp', type=float, default=0.8)
    argparser.add_argument('--GraphHI_CL_sample_num', type=int, default=10)
    argparser.add_argument('--GraphHI_CL_delta', type=float, default=1.0)
    argparser.add_argument('--GraphHI_dynamic_coefficient_after_epoches', type=float, default=0.5)
    argparser.add_argument('--GraphHI_dynamic_coefficient_max', type=float, default=1.1)
    argparser.add_argument('--GraphHI_dynamic_coefficient_epsilon', type=float, default=0.9)
    argparser.add_argument('--GraphHI_dynamic_coefficient_gamma', type=float, default=1.001)
    argparser.add_argument('--GraphHI_attention_temp_w', type=float, default=1)
    argparser.add_argument('--GraphHI_attention_temp_b', type=float, default=0)
    argparser.add_argument('--GraphHI_attention_temp_range_min', type=float, default=1)
    argparser.add_argument('--GraphHI_attention_temp_range_max', type=float, default=20)
    argparser.add_argument('--GraphHI_attention_temp_smooth_addition_new_weight', type=float, default=1, help="final T = (GraphHI_attention_temp_smooth_addition_new_weight*smoothed T + T)/(GraphHI_attention_temp_smooth_addition_new_weight + 1)")

    args = argparser.parse_args()
    
    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)
    
    # run
    val_accs = []
    test_accs = []

    for i in range(args.n_runs):
        # load data
        if args.dataset == 'yelp':
            from dgl.data import YelpDataset
            data = YelpDataset()
            graph = data[0]
            n_classes = data.num_classes # # multi-label classification 100
            labels = graph.ndata['label'].float()
            idx = torch.arange(graph.num_nodes())
            train_idx, val_idx, test_idx = idx[graph.ndata['train_mask'].bool()], idx[graph.ndata['val_mask'].bool()], idx[graph.ndata['test_mask'].bool()]
            evaluator = None
        else:  # arxiv or products
            data = DglNodePropPredDataset(name=args.dataset, root=f'{args.data_dir}')
            evaluator = Evaluator(name=args.dataset)
            splitted_idx = data.get_idx_split()
            train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
            graph, labels = data[0]
            n_classes = (labels.max() + 1).item()

        # add reverse edges
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

        # add self-loop
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        if args.dataset != 'yelp':
            graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

        in_feats = graph.ndata["feat"].shape[1]
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
        labels = labels.to(device)

        th.manual_seed(args.seed)
        alpha_ALC_rewards = alpha_chosen_index = alpha_CL = alpha_ST = alpha_CE = alpha_base = None
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        args.seed += 1

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy on {args.dataset}: {np.mean(test_accs)} ± {np.std(test_accs)}")
