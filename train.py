import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification


def percentile_threshold(values: torch.Tensor, percentile: float) -> torch.Tensor:
    percentile = min(max(percentile, 0.0), 100.0)
    sorted_vals = torch.sort(values).values
    idx = int((percentile / 100.0) * (sorted_vals.numel() - 1))
    return sorted_vals[idx]


def train_grace(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()
    return loss.item()


def mine_unlabeled_positives(
        model: Model,
        x,
        edge_index,
        similarity_threshold,
        similarity_percentile,
        max_du_per_node,
        use_mutual_topk,
        beta):
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        sim = model.sim(z, z)

        num_nodes = sim.size(0)
        eye_mask = torch.eye(num_nodes, dtype=torch.bool, device=sim.device)

        offdiag = sim[~eye_mask]
        if similarity_threshold is None:
            active_threshold = percentile_threshold(offdiag, similarity_percentile)
        else:
            active_threshold = torch.tensor(float(similarity_threshold), device=sim.device)

        # D_U^+: mined from non-augmented pairs with high semantic similarity.
        du_pos_mask = (sim > active_threshold) & (~eye_mask)

        if max_du_per_node > 0:
            k = min(max_du_per_node, num_nodes - 1)
            topk_idx = torch.topk(sim.masked_fill(eye_mask, -1e9), k=k, dim=1).indices
            topk_mask = torch.zeros_like(du_pos_mask)
            topk_mask.scatter_(1, topk_idx, True)
            du_pos_mask = du_pos_mask & topk_mask

        if use_mutual_topk:
            du_pos_mask = du_pos_mask & du_pos_mask.t()

        sim_min = offdiag.min()
        sim_max = offdiag.max()
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-12)
        du_pos_weight = torch.exp(beta * sim_norm) * du_pos_mask.float()

        mined_pairs = int(du_pos_mask.sum().item())
        mean_weight = float(du_pos_weight[du_pos_mask].mean().item()) if mined_pairs > 0 else 0.0
        mean_pairs_per_node = float(du_pos_mask.float().sum(1).mean().item())

    return {
        'du_pos_mask': du_pos_mask,
        'du_pos_weight': du_pos_weight,
        'mined_pairs': mined_pairs,
        'sim_min': float(sim_min.item()),
        'sim_max': float(sim_max.item()),
        'active_threshold': float(active_threshold.item()),
        'mean_weight': mean_weight,
        'mean_pairs_per_node': mean_pairs_per_node
    }


def train_iflgr(model: Model, x, edge_index, du_pos_mask, du_pos_weight, unlabeled_weight):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(
        z1,
        z2,
        batch_size=0,
        corrected=True,
        du_pos_mask=du_pos_mask,
        du_pos_weight=du_pos_weight,
        unlabeled_weight=unlabeled_weight)

    loss.backward()
    optimizer.step()
    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--method', type=str, default='grace', choices=['grace', 'ifl-gr'])
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    warmup_epochs = config.get('warmup_epochs', 0)
    update_interval = config.get('update_interval', 10)
    similarity_threshold = config.get('similarity_threshold', 0.8)
    similarity_percentile = config.get('similarity_percentile', 99.5)
    max_du_per_node = config.get('max_du_per_node', 10)
    use_mutual_topk = config.get('use_mutual_topk', True)
    beta = config.get('beta', 2.0)
    unlabeled_weight = config.get('unlabeled_weight', 1.0)
    corrected_ramp_epochs = config.get('corrected_ramp_epochs', 50)

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())

    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    du_cache = None

    for epoch in range(1, num_epochs + 1):
        refresh_du = False
        mined_pairs = 0
        mean_weight = 0.0
        mean_pairs_per_node = 0.0
        active_threshold = 0.0
        current_unlabeled_weight = 0.0

        if args.method == 'grace':
            loss = train_grace(model, data.x, data.edge_index)
            phase = 'grace'
        else:
            if epoch <= warmup_epochs:
                loss = train_grace(model, data.x, data.edge_index)
                phase = 'warmup'
            else:
                if du_cache is None or (epoch - warmup_epochs - 1) % update_interval == 0:
                    du_cache = mine_unlabeled_positives(
                        model,
                        data.x,
                        data.edge_index,
                        similarity_threshold,
                        similarity_percentile,
                        max_du_per_node,
                        use_mutual_topk,
                        beta)
                    refresh_du = True

                mined_pairs = du_cache['mined_pairs']
                mean_weight = du_cache['mean_weight']
                mean_pairs_per_node = du_cache['mean_pairs_per_node']
                active_threshold = du_cache['active_threshold']

                progress = max(epoch - warmup_epochs, 0) / max(corrected_ramp_epochs, 1)
                current_unlabeled_weight = unlabeled_weight * min(progress, 1.0)

                loss = train_iflgr(
                    model,
                    data.x,
                    data.edge_index,
                    du_cache['du_pos_mask'],
                    du_cache['du_pos_weight'],
                    current_unlabeled_weight)
                phase = 'corrected'

        now = t()
        if args.method == 'grace' or phase == 'warmup':
            print(f'(T) | Epoch={epoch:03d}, phase={phase}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        else:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'phase={phase}, refresh_du={int(refresh_du)}, '
                  f'lambda_u={current_unlabeled_weight:.4f}, '
                  f'ts={active_threshold:.4f}, '
                  f'mined_pairs={mined_pairs}, '
                  f'avg_pairs_per_node={mean_pairs_per_node:.2f}, '
                  f'mean_w={mean_weight:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    test(model, data.x, data.edge_index, data.y, final=True)
