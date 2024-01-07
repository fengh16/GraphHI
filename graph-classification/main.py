from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
from gnn import GNN
from conv import GraphConvNoWeight
import random
import dgl


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


def st_loss(logits_S, logits_T, temperature=1.0):
    if len(logits_S.shape) == 2 and logits_S.shape[1] == 1:
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        loss = -(beta_logits_T * F.logsigmoid(beta_logits_S) + (1 - beta_logits_T) * F.logsigmoid(1 - beta_logits_S)).sum(dim=-1)
    else:
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


def load_knowledge(kd_path, device): # load teacher knowledge
    assert os.path.isfile(kd_path), "Please download teacher knowledge first"
    knowledge = torch.load(kd_path, map_location=device)
    tea_logits = knowledge['logits'].float()
    tea_h = knowledge['h-embedding']
    tea_g = knowledge['g-embedding']
    new_ptr = knowledge['ptr']
    return tea_logits, tea_h, tea_g, new_ptr

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
alpha_CL = alpha_ST = alpha_CE = alpha_base = None
modify_epoch = -1

def train(model, device, loader, optimizer, task_type, others):
    global alpha_CL, alpha_ST, alpha_CE, alpha_base, modify_epoch
    model.train()
    (tea_logits, tea_h, tea_g, new_ptr, loss_dis, epoch, args, train_ids) = others

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            # pred = model(batch)  # torch.Size([32, 1])
            pred, stu_bh, stu_bg = model(batch)  # torch.Size([32, 1]), [#nodes, out_dim], [#graphs, out_dim]
            new_ids = [(train_ids==vid).nonzero().item() for vid in batch.id]
            bids = torch.tensor(new_ids, device=new_ptr.device)
            new_pre = new_ptr[:-1]
            new_post = new_ptr[1:]
            bpre = new_pre[bids]
            bpost = new_post[bids]
            bnid = torch.cat([torch.arange(pre, post) for pre, post in list(zip(*[bpre, bpost]))], dim=0)
            tea_bh = tea_h[bnid].to(device)
            tea_bg = tea_g[bids].to(device)
            tea_by = tea_logits[bids].to(device)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:  # 'binary classification'
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            #++++++++++++++++++++++++
            if args.role == 'GraphHI':
                modify_epoch = args.GraphHI_dynamic_coefficient_after_epoches
                if modify_epoch is not None:
                    if 0 < modify_epoch < 1:
                        modify_epoch = int(args.epochs * args.GraphHI_dynamic_coefficient_after_epoches)
                    if epoch <= modify_epoch:
                        alpha_CL, alpha_ST, alpha_CE = alpha_base
                with torch.no_grad():
                    temperature = torch.ones((batch.x.shape[0],1), device=device) * args.GraphHI_attention_temp_range_max
                    temperature[:pred.shape[0]] = (args.GraphHI_attention_temp_range_max - (args.GraphHI_attention_temp_range_max - args.GraphHI_attention_temp_range_min) * ((pred.max(dim=-1)[0] * args.GraphHI_attention_temp_w + args.GraphHI_attention_temp_b).sigmoid())).view(-1, 1).detach()
                    GNN_layer = GraphConvNoWeight(in_channels=1, out_channels=1, weight=False, bias=False)
                    temperature = (GNN_layer(temperature, batch.edge_index) * args.GraphHI_attention_temp_smooth_addition_new_weight + temperature) / (args.GraphHI_attention_temp_smooth_addition_new_weight + 1)
                    temperature = temperature[:pred.shape[0]].reshape(-1, 1)
                    if batch.y.shape[-1] == 1:
                        need_change_index = tea_by.argmax(dim=-1).reshape(-1, 1) != batch.y[:pred.shape[0]].reshape(-1, 1)
                        temperature[need_change_index] = args.GraphHI_attention_temp_range_max
                loss = alpha_CE * loss + alpha_ST * st_loss(pred, tea_by, temperature=temperature)
                loss = loss + alpha_CL * (
                    cl_loss(model.CL_linear_h(stu_bh), tea_bh, GraphHI_CL_sample_num=args.GraphHI_CL_sample_num, temperature=args.GraphHI_CL_temp, delta=args.GraphHI_CL_delta)
                    + cl_loss(model.CL_linear_g(stu_bg), tea_bg, GraphHI_CL_sample_num=args.GraphHI_CL_sample_num, temperature=args.GraphHI_CL_temp, delta=args.GraphHI_CL_delta)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()

def eval(model, device, loader, evaluator, distill=False):
    model.eval()
    y_true = []
    y_pred = []
    #++++++++++++++++++++++++
    y_ids = []
    ptrs = []
    #++++++++++++++++++++++++

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            #++++++++++++++++++++++++Start
            if distill:
                y_ids.append(batch.id.cpu())
                if len(ptrs) == 0:
                    ptrs.append(batch.ptr.cpu())
                else:
                    ptr = batch.ptr.cpu() + ptrs[-1][-1]
                    ptrs.append(ptr)
            #++++++++++++++++++++++++End

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    #++++++++++++++++++++++++Start
    if distill:
        y_id = torch.cat(y_ids, dim=0)
        ptr = torch.cat(ptrs, dim=0).unique()
        inv_perm = y_id.sort()[1]
        ptr_diff = torch.tensor(np.diff(ptr.numpy()))
        inv_ptr = ptr_diff[inv_perm].cumsum(dim=0)
        inv_ptr = torch.cat([torch.tensor([0]), inv_ptr], dim=0)
    #++++++++++++++++++++++++

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if distill:
        return evaluator.eval(input_dict), inv_ptr

    return evaluator.eval(input_dict)


def main():
    global alpha_CL, alpha_ST, alpha_CE, alpha_base
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=48, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv", help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')
    parser.add_argument("--role", type=str, default="vani", choices=['stu', 'vani', 'GraphHI'])
    parser.add_argument("--data_dir", type=str, default='../../../dataset')
    parser.add_argument("--kd_dir", type=str, default='../../distilled')
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")

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
    set_random_seed(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset, root=f'{args.data_dir}/')

    # for i, data in enumerate(dataset):
    #     data.temperature = torch.ones((data.x.shape[0],1), dtype=torch.float) * args.GraphHI_attention_temp_range_max

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    CL_dim_h = None
    CL_dim_g = None

    #++++++++++++++++++++++++Load knowledge
    if args.role != 'vani':
        data_name=args.dataset.split('-')[1].upper()
        kd_path = os.path.join(args.kd_dir, data_name + f'-knowledge.pth.tar')
        if args.dataset == 'ogbg-molpcba':
            tea_logits, tea_h, tea_g, new_ptr = load_knowledge(kd_path, device='cpu')
        elif args.dataset == 'ogbg-molhiv':
            tea_logits, tea_h, tea_g, new_ptr = load_knowledge(kd_path, device=device)
        y_true = dataset.data.y[split_idx["train"]]
        input_dict = {"y_true": y_true, "y_pred": tea_logits}
        print(f'Teacher performance on Training set: {evaluator.eval(input_dict)}')
        CL_dim_h = tea_h.shape[-1]
        CL_dim_g = tea_g.shape[-1]
    else:
        tea_logits, tea_h, tea_g, new_ptr = None, None, None, None


    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, CL_dim_h=CL_dim_h, CL_dim_g=CL_dim_g).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, CL_dim_h=CL_dim_h, CL_dim_g=CL_dim_g).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, CL_dim_h=CL_dim_h, CL_dim_g=CL_dim_g).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True, CL_dim_h=CL_dim_h, CL_dim_g=CL_dim_g).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_dis = torch.nn.BCELoss()
    #++++++++++++++++++++++++

    valid_curve = []
    test_curve = []

    alpha_ST = args.GraphHI_alpha
    alpha_CE = 1
    alpha_CL = args.GraphHI_CL_weight
    
    alpha_ALC_rewards = [[] for _ in range(3)]
    alpha_chosen_index = 0
    alpha_original_ = [alpha_CL, alpha_ST, alpha_CE]
    alpha_CL, alpha_ST, alpha_CE = [i / sum(alpha_original_) * 3 for i in alpha_original_]
    
    alpha_base = [alpha_CL, alpha_ST, alpha_CE]

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        others = (tea_logits, tea_h, tea_g, new_ptr, loss_dis, epoch, args, split_idx["train"].to(device))
        train(model, device, train_loader, optimizer, dataset.task_type, others)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Validation': valid_perf, 'Test': test_perf})

        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if modify_epoch is None or epoch >= modify_epoch:
            if alpha_ALC_rewards[alpha_chosen_index] == []:
                for i in range(len(alpha_ALC_rewards)):
                    alpha_ALC_rewards[i].append(list(train_perf.values())[0])
            else:
                alpha_ALC_rewards[alpha_chosen_index].append(list(train_perf.values())[0])
            if random.random() < args.GraphHI_dynamic_coefficient_epsilon:
                # explore
                alpha_chosen_index = random.randint(0, 3 - 1)
            else:
                # exploit
                alpha_original_rewards = [sum(reward) / len(reward) for reward in alpha_ALC_rewards]
                alpha_chosen_index = alpha_original_rewards.index(max(alpha_original_rewards))
            alpha_original_ = [alpha_CL, alpha_ST, alpha_CE]
            alpha_original_[alpha_chosen_index] *= args.GraphHI_dynamic_coefficient_gamma
            if args.GraphHI_dynamic_coefficient_max > 1:
                if alpha_original_[alpha_chosen_index] > args.GraphHI_dynamic_coefficient_max * alpha_base[alpha_chosen_index]:
                    alpha_original_[alpha_chosen_index] = args.GraphHI_dynamic_coefficient_max * alpha_base[alpha_chosen_index]
            alpha_CL, alpha_ST, alpha_CE = [i / sum(alpha_original_) * 3 for i in alpha_original_]

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    print('Finished training!')
    print(args)
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))


if __name__ == "__main__":
    main()
