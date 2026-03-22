import argparse
import os
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
from torch_geometric.utils import dropout_adj, degree, to_undirected
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification

# Unified training entry for four methods on one dataset:
# - grace   : vanilla GRACE augment + InfoNCE
# - ifl-gr  : GRACE augment + semantically guided corrected InfoNCE
# - gca     : GCA structure-aware augment + InfoNCE
# - ifl-gc  : GCA augment + semantically guided corrected InfoNCE


def compute_pr(edge_index: torch.Tensor, damp: float = 0.85, k: int = 10):
    num_nodes = int(edge_index.max().item()) + 1
    src, dst = edge_index
    deg_out = degree(src, num_nodes=num_nodes).clamp_min(1.0)
    x = torch.ones((num_nodes,), dtype=torch.float32, device=edge_index.device)

    for _ in range(k):
        edge_msg = x[src] / deg_out[src]
        agg_msg = torch.zeros_like(x)
        agg_msg.index_add_(0, dst, edge_msg)
        x = (1.0 - damp) * x + damp * agg_msg

    return x


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.0):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.0

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.clamp_min(1e-12).log()
    s = (w.max() - w) / (w.max() - w.mean() + 1e-12)

    return s


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = deg_col.clamp_min(1e-12).log()
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean() + 1e-12)

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = pv_row.clamp_min(1e-12).log()
    s_col = pv_col.clamp_min(1e-12).log()

    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col

    weights = (s.max() - s) / (s.max() - s.mean() + 1e-12)
    return weights


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

        # D_U^+: mined from high-similarity pairs on clean embeddings.
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


def train_iflgc(
        model: Model,
        x,
        edge_index,
        drop_scheme,
        drop_weights,
        feature_weights,
        du_pos_mask,
        du_pos_weight,
        unlabeled_weight,
        refl_du_weight):
    model.train()
    optimizer.zero_grad()

    # Reuse GCA-style structure-aware views for IFL-GC.
    def gca_drop_edge(rate):
        if drop_scheme == 'uniform':
            return dropout_adj(edge_index, p=rate)[0]
        if drop_scheme in ['degree', 'pr']:
            return drop_edge_weighted(edge_index, drop_weights, p=rate, threshold=0.7)
        raise ValueError(f'undefined drop scheme: {drop_scheme}')

    edge_index_1 = gca_drop_edge(drop_edge_rate_1)
    edge_index_2 = gca_drop_edge(drop_edge_rate_2)

    if drop_scheme in ['degree', 'pr']:
        x_1 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_1)
        x_2 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_2)
    else:
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
        unlabeled_weight=unlabeled_weight,
        corrected_variant='ifl-gc',
        refl_du_weight=refl_du_weight)

    loss.backward()
    optimizer.step()
    return loss.item()


def train_gca(model: Model, x, edge_index, drop_scheme, drop_weights, feature_weights):
    model.train()
    optimizer.zero_grad()

    def gca_drop_edge(rate):
        if drop_scheme == 'uniform':
            return dropout_adj(edge_index, p=rate)[0]
        if drop_scheme in ['degree', 'pr']:
            return drop_edge_weighted(edge_index, drop_weights, p=rate, threshold=0.7)
        raise ValueError(f'undefined drop scheme: {drop_scheme}')

    edge_index_1 = gca_drop_edge(drop_edge_rate_1)
    edge_index_2 = gca_drop_edge(drop_edge_rate_2)

    if drop_scheme in ['degree', 'pr']:
        x_1 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_1)
        x_2 = drop_feature_weighted_2(x, feature_weights, drop_feature_rate_2)
    else:
        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    # Keep evaluation identical across all methods.
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Dataset cache root. Defaults to GRACE/datasets')
    parser.add_argument('--method', type=str, default='grace', choices=['grace', 'ifl-gr', 'gca', 'ifl-gc'])
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
    gca_drop_scheme = config.get('gca_drop_scheme', 'degree')
    gca_pr_k = config.get('gca_pr_k', 200)
    iflgc_refl_du_weight = config.get('iflgc_refl_du_weight', 0.3)

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())

    # Prefer project-local cache directory so server runs do not depend on ~/datasets.
    grace_dir = osp.dirname(osp.abspath(__file__))
    dataset_root = args.dataset_root if args.dataset_root else osp.join(grace_dir, 'datasets')
    os.makedirs(dataset_root, exist_ok=True)

    # Pass dataset_root directly. PyG datasets append dataset name internally.
    # This avoids duplicated folders like datasets/Cora/Cora.
    path = dataset_root
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
    gca_drop_weights = None
    gca_feature_weights = None

    # Precompute GCA augmentation weights once if needed by method.
    if args.method in ['gca', 'ifl-gc']:
        if gca_drop_scheme == 'degree':
            gca_drop_weights = degree_drop_weights(data.edge_index).to(device)
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            gca_feature_weights = feature_drop_weights(data.x, node_deg).to(device)
        elif gca_drop_scheme == 'pr':
            gca_drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=gca_pr_k).to(device)
            node_pr = compute_pr(data.edge_index, k=gca_pr_k)
            gca_feature_weights = feature_drop_weights(data.x, node_pr).to(device)
        elif gca_drop_scheme == 'uniform':
            gca_feature_weights = None
        else:
            raise ValueError(f'unsupported gca_drop_scheme: {gca_drop_scheme}')

    for epoch in range(1, num_epochs + 1):
        refresh_du = False
        mined_pairs = 0
        mean_weight = 0.0
        mean_pairs_per_node = 0.0
        active_threshold = 0.0
        current_unlabeled_weight = 0.0

        if args.method == 'grace':
            # Baseline branch: pure GRACE.
            loss = train_grace(model, data.x, data.edge_index)
            phase = 'grace'
        elif args.method == 'gca':
            # GCA branch: structure-aware augmentation only.
            loss = train_gca(
                model,
                data.x,
                data.edge_index,
                gca_drop_scheme,
                gca_drop_weights,
                gca_feature_weights)
            phase = 'gca'
        elif args.method == 'ifl-gc':
            # IFL-GC branch:
            # 1) warmup with GCA objective
            # 2) periodically mine D_U^+
            # 3) optimize corrected InfoNCE (cross-view + same-view semantic terms)
            if epoch <= warmup_epochs:
                loss = train_gca(
                    model,
                    data.x,
                    data.edge_index,
                    gca_drop_scheme,
                    gca_drop_weights,
                    gca_feature_weights)
                phase = 'warmup-gca'
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

                loss = train_iflgc(
                    model,
                    data.x,
                    data.edge_index,
                    gca_drop_scheme,
                    gca_drop_weights,
                    gca_feature_weights,
                    du_cache['du_pos_mask'],
                    du_cache['du_pos_weight'],
                    current_unlabeled_weight,
                    iflgc_refl_du_weight)
                phase = 'corrected-gca'
        else:
            # IFL-GR branch:
            # 1) warmup with GRACE objective
            # 2) periodically mine D_U^+
            # 3) optimize corrected InfoNCE (cross-view semantic term)
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
        if args.method in ['grace', 'gca'] or phase in ['warmup', 'warmup-gca']:
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
