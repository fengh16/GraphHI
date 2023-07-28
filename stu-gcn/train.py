import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gcn import GCN
import random
import os
import warnings
warnings.filterwarnings('ignore')


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.random.seed(seed)


def compute_micro_f1(logits, y, mask=None):
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def st_loss(logits_S, logits_T, temperature=1.0):
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1)
    temp_multiplier = temperature ** 2
    if isinstance(temperature, float) or isinstance(temperature, int):
        return loss.mean() * temp_multiplier
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


def run(args, g, n_classes, cuda, n_running):
    set_random_seed(args)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_edges = g.number_of_edges()

    CL_dim = None

    # load teacher knowledge
    if args.role in ['GraphHI']:
        inter_model_path = os.path.join(args.inter_model_dir, args.dataset + f'-knowledge.pth.tar')
        assert os.path.isfile(inter_model_path), "Please download teacher knowledge first"
        knowledge = torch.load(inter_model_path, map_location=g.device)
        tea_logits = knowledge['logits']
        tea_emb = knowledge['embedding']
        
        CL_dim = tea_emb.shape[-1]

        if 'perm' in knowledge.keys() and args.dataset in ['arxiv', 'reddit']:
            perm = knowledge['perm']
            inv_perm = perm.sort()[1]
            tea_logits = tea_logits[inv_perm]
            tea_emb = tea_emb[inv_perm]

        test_acc = compute_micro_f1(tea_logits, labels, test_mask)  # for val
        print(f'Teacher Test SCORE: {test_acc:.3%}')

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                CL_dim=CL_dim)

    if labels.dim() == 1:
        loss_fcn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown dataset with wrong labels: {}'.format(args.dataset))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    if cuda:
        model.cuda()

    if args.role == 'GraphHI':
        model_param_count = sum(param.numel() for param in model.parameters())
        if model.CL_dim == None:
            CL_transfer_param_count = 0
        else:
            CL_transfer_param_count = sum(param.numel() for param in model.CL_linear.parameters())
            model_param_count -= CL_transfer_param_count
        print("#Params in the student model:", model_param_count)

    dur = []
    log_every = 30
    best_eval_acc, final_test_acc, best_val_loss = 0, 0, float("inf")

    alpha_ST = args.GraphHI_alpha
    alpha_CE = 1
    alpha_CL = args.GraphHI_CL_weight
    
    alpha_ALC_rewards = [[1] for _ in range(3)]
    alpha_chosen_index = 0
    alpha_original_ = [alpha_CL, alpha_ST, alpha_CE]
    alpha_CL, alpha_ST, alpha_CE = [i / sum(alpha_original_) * 3 for i in alpha_original_]
    
    alpha_base = [alpha_CL, alpha_ST, alpha_CE]

    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        logits = model(features)
        label_loss = loss_fcn(logits[train_mask], labels[train_mask])
        if args.role == 'GraphHI':
            label_loss = loss_fcn(logits[train_mask], labels[train_mask])
            modify_epoch = args.GraphHI_dynamic_coefficient_after_epoches
            if modify_epoch is not None:
                if 0 < modify_epoch < 1:
                    modify_epoch = int(args.n_epochs * args.GraphHI_dynamic_coefficient_after_epoches)
                if epoch <= modify_epoch:
                    alpha_CL, alpha_ST, alpha_CE = alpha_base
            with torch.no_grad():
                temperature = (args.GraphHI_attention_temp_range_max - (args.GraphHI_attention_temp_range_max - args.GraphHI_attention_temp_range_min) * ((logits.max(dim=-1)[0] * args.GraphHI_attention_temp_w + args.GraphHI_attention_temp_b).sigmoid())).view(-1, 1).detach()
                from dgl.nn.pytorch import GraphConv
                GNN_layer = GraphConv(in_feats=1, out_feats=1, norm='both', weight=False, bias=False, activation=None)
                temperature = (GNN_layer(g, temperature) * args.GraphHI_attention_temp_smooth_addition_new_weight + temperature) / (args.GraphHI_attention_temp_smooth_addition_new_weight + 1)
                need_change_index = tea_logits.argmax(dim=-1).reshape(-1,1) != labels.reshape(-1,1)
                need_change_index[~train_mask] = False
                temperature[need_change_index] = args.GraphHI_attention_temp_range_max
            loss = alpha_CE * label_loss + alpha_ST * st_loss(logits, tea_logits, temperature=temperature)
            loss = loss + alpha_CL * cl_loss(model.CL_linear(model.emb), tea_emb, GraphHI_CL_sample_num=args.GraphHI_CL_sample_num, temperature=args.GraphHI_CL_temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_D = loss
        else:
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_D = loss

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = compute_micro_f1(logits, labels, train_mask) 
        val_loss = loss_fcn(logits[val_mask], labels[val_mask])
        eval_acc = compute_micro_f1(logits, labels, val_mask) 
        test_acc = compute_micro_f1(logits, labels, test_mask)
        
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_eval_acc = eval_acc
            final_test_acc = test_acc
        if epoch % log_every == 0:
            print(f"Run: {n_running}/{args.n_runs} | Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss_D.item():.4f} | "
            f"Val {eval_acc:.4f} | Test {test_acc:.4f} | Best Test {final_test_acc:.4f} | ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(args)
    print(f"Test accuracy on {args.dataset}: {final_test_acc:.2%}\n")
    return best_eval_acc, final_test_acc


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'flickr':
        from torch_geometric.datasets import Flickr
        import torch_geometric.transforms as T
        pyg_data = Flickr(root=f'{args.data_dir}/flickr', pre_transform=T.ToSparseTensor())[0]  # replace edge_index with adj
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))

        g.ndata['feat'] = pyg_data.x
        g.ndata['label'] = pyg_data.y
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        n_classes = pyg_data.y.max().item() + 1
    elif args.dataset == 'reddit':
        from torch_geometric.datasets import Reddit2
        import torch_geometric.transforms as T
        pyg_data = Reddit2(f'{args.data_dir}/Reddit2', pre_transform=T.ToSparseTensor())[0]
        pyg_data.x = (pyg_data.x - pyg_data.x.mean(dim=0)) / pyg_data.x.std(dim=0)
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))
        g.ndata['feat'] = pyg_data.x
        g.ndata['label'] = pyg_data.y
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        n_classes = pyg_data.y.max().item() + 1
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.dataset not in ['reddit', 'yelp', 'flickr', 'corafull']:
        g = data[0]
        n_classes = data.num_labels
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    # normalization
    degs = g.in_degrees().clamp(min=1).float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # run
    val_accs = []
    test_accs = []
    for i in range(args.n_runs):
        val_acc, test_acc = run(args, g, n_classes, cuda, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        args.seed += 1

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy on {args.dataset}: {np.mean(test_accs)} ± {np.std(test_accs)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=600, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--role", type=str, default="vani", choices=['vani', 'GraphHI'])
    parser.add_argument("--data_dir", type=str, default='../../dataset')
    parser.add_argument("--inter_model_dir", type=str, default='../inter_model')
    parser.set_defaults(self_loop=True)

    parser.add_argument("--n-runs", type=int, default=1, help="running times")

    parser.add_argument('--GraphHI_alpha', type=float, default=0.5)
    parser.add_argument('--GraphHI_CL_weight', type=float, default=1)
    parser.add_argument('--GraphHI_CL_temp', type=float, default=0.8)
    parser.add_argument('--GraphHI_CL_sample_num', type=int, default=10)
    parser.add_argument('--GraphHI_CL_delta', type=float, default=1.0)
    parser.add_argument('--GraphHI_dynamic_coefficient_after_epoches', type=float, default=0.5)
    parser.add_argument('--GraphHI_dynamic_coefficient_max', type=float, default=1.1)
    parser.add_argument('--GraphHI_dynamic_coefficient_epsilon', type=float, default=0.9)
    parser.add_argument('--GraphHI_dynamic_coefficient_gamma', type=float, default=1.001)
    parser.add_argument('--GraphHI_attention_temp_w', type=float, default=1)
    parser.add_argument('--GraphHI_attention_temp_b', type=float, default=0)
    parser.add_argument('--GraphHI_attention_temp_range_min', type=float, default=1)
    parser.add_argument('--GraphHI_attention_temp_range_max', type=float, default=20)
    parser.add_argument('--GraphHI_attention_temp_smooth_addition_new_weight', type=float, default=1, help="final T = (GraphHI_attention_temp_smooth_addition_new_weight*smoothed T + T)/(GraphHI_attention_temp_smooth_addition_new_weight + 1)")

    args = parser.parse_args()
    
    main(args)
