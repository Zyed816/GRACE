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


def train(model: Model, x, edge_index, epoch, dynamic_resample_start_epoch):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    min_pos_sim = None
    promoted_pair_count = 0

    if epoch < dynamic_resample_start_epoch:
        loss = model.loss(z1, z2, batch_size=0)
    else:
        with torch.no_grad():
            sim = model.sim(z1, z2)
            min_pos_sim = sim.diag().min()

            # Keep original paired positives and promote hard negatives
            # whose similarity is higher than the weakest original positive.
            dynamic_pos_mask = sim > min_pos_sim
            dynamic_pos_mask.fill_diagonal_(True)

            promoted_pair_count = int(dynamic_pos_mask.sum().item() - sim.size(0))

        loss = model.loss(z1, z2, batch_size=0, between_pos_mask=dynamic_pos_mask)

    loss.backward()
    optimizer.step()

    if min_pos_sim is None:
        return loss.item(), None, promoted_pair_count
    return loss.item(), float(min_pos_sim.item()), promoted_pair_count


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
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
    dynamic_resample_start_epoch = config.get('dynamic_resample_start_epoch', num_epochs + 1)

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
    for epoch in range(1, num_epochs + 1):
        loss, min_pos_sim, promoted_pair_count = train(
            model,
            data.x,
            data.edge_index,
            epoch,
            dynamic_resample_start_epoch)

        now = t()
        if epoch < dynamic_resample_start_epoch:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        else:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'min_pos_sim={min_pos_sim:.4f}, promoted_pairs={promoted_pair_count}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    test(model, data.x, data.edge_index, data.y, final=True)
