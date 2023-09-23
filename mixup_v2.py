import os
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv

torch.autograd.set_detect_anomaly(True)

import torch_geometric.transforms as T
from torch_geometric.nn.aggr import SumAggregation
import numpy as np
import random
from torch import Tensor
from typing import Optional


class MixUp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: Tensor,
                lam: int,
                ptr: Optional[Tensor] = None) -> Tensor:
        if ptr is not None:
            ptr = ptr.tolist()
        else:
            ptr = [0, x.shape[0]]
        tracker = ptr[0]
        random_indices = torch.zeros(x.shape[0], dtype=torch.long)
        for idx, ptr in zip(range(x.shape[0]), ptr[1:]):
            node_indices = np.arange(tracker, ptr)
            np.random.shuffle(node_indices)
            random_indices[tracker: ptr] = torch.tensor(node_indices, dtype=torch.long)
            tracker = ptr
        out_x = (x * lam) + (1 - lam) * x[random_indices]
        return out_x


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(Net, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channel)
        self.mixup = MixUp()
        self.aggr = SumAggregation()

    def forward(self, x0, edge_index, lam, ptr):
        x1 = self.conv1(x0, edge_index)
        x1 = F.relu(x1)
        x1 = self.mixup(x1, lam, ptr)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = self.mixup(x2, lam, ptr)
        x2 = F.dropout(x2, p=0.4, training=self.training)

        x3 = self.conv2(x2, edge_index)
        x3 = F.relu(x3)
        x3 = self.mixup(x3, lam, ptr)
        x3 = F.dropout(x3, p=0.4, training=self.training)

        x3 = self.aggr(x3, ptr=ptr)
        # x3 = self.mixup(x3, lam, None)
        x = self.lin(x3)

        return x.log_softmax(dim=-1)

@torch.no_grad()
def test(data, lam):
    model.eval()

    class_dict = {}
    for idx, y in enumerate(data.y):
        y = y.cpu().item()
        if class_dict.get(y, None) is None:
            class_dict[y] = [idx]
        else:
            class_dict[y].append(idx)

    out = model(data.x.cuda(), data.edge_index.cuda(), lam, data.ptr.cuda())
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    return correct.sum().item() / out.shape[0]


def train(data):
    model.train()
    mixup = True

    if mixup:
        lam = np.random.beta(4.0, 4.0)
    else:
        lam = 1.0

    data = data.to(device)
    # print("Y:", data.y)

    optimizer.zero_grad()

    class_dict = {}
    for idx, y in enumerate(data.y):
        y_val = y.cpu().item()
        if class_dict.get(y_val, None) is None:
            class_dict[y_val] = [idx]
        else:
            class_dict[y_val].append(idx)

    out = model(data.x.cuda(), data.edge_index.cuda(), lam, data.ptr.cuda())
    # print(out)
    loss = F.nll_loss(out, data.y) * lam
    # print(loss)
    loss.backward()
    optimizer.step()

    return loss.item(), lam


if __name__ == "__main__":

    SEED = 1
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    d_name = 'PROTEINS'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', d_name)
    dataset = TUDataset(path, d_name, transform=T.NormalizeFeatures())

    train_size = 0.7
    test_size = 0.2
    val_size = 0.1
    batch_size = 100

    dataset.shuffle()

    total_graphs = len(dataset)

    train_loader = DataLoader(dataset=dataset[:int(total_graphs * train_size)],
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=dataset[int(total_graphs * train_size): int(total_graphs * train_size) +
                                                                             int(total_graphs * test_size)],
                             batch_size=batch_size,
                             shuffle=False)

    val_loader = DataLoader(dataset=dataset[int(total_graphs * train_size) + int(total_graphs * test_size):],
                            batch_size=batch_size,
                            shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(hidden_channels=256, in_channel=dataset.num_node_features, out_channel=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    accord_epoch = 0
    accord_train_acc = 0
    accord_train_loss = 0
    # print(dataset[0])
    for epoch in range(1, 300):
        losses = []
        train_accs = []
        val_accs = []
        test_accs = []
        for data in train_loader:
            loss, lam = train(data)
            accs = test(data, lam)
            losses.append(loss)
            train_accs.append(accs)
        for data in val_loader:
            accs = test(data, lam)
            val_accs.append(accs)
        for data in test_loader:
            accs = test(data, lam)
            test_accs.append(accs)
        print(f'Epoch: {epoch:02d}, Loss: {sum(losses) / len(train_loader):.4f}, '
              f'Train Acc: {sum(train_accs) / len(train_loader):.4f},'
              f'Val Acc: {sum(val_accs) / len(val_loader):.4f}',
              f'Test Acc: {sum(test_accs) / len(test_loader):.4f}')
