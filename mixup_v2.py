import os
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, GCN
from torch_geometric.transforms import OneHotDegree
from torch_geometric.nn.pool import global_add_pool

torch.autograd.set_detect_anomaly(True)

import argparse
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

    def forward(self, x0, edge_index, lam, ptr, batch):
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

        x3 = global_add_pool(x3, batch=batch)
        # x3 = self.mixup(x3, lam, None)
        x = self.lin(x3)

        return x.log_softmax(dim=-1)


@torch.no_grad()
def test(batch_data, lam_val, ranges):
    model.eval()
    out = model(batch_data.x.cuda(),
                batch_data.edge_index.cuda(),
                lam_val,
                batch_data.ptr.cuda(),
                batch_data.batch.cuda())
    labels = batch_data.y
    preds = out.argmax(dim=-1)
    correct_all = preds.eq(batch_data.y.to(device)).int().sum().cpu().item()
    batch_data = batch_data.to_data_list()
    graph_correct = {0: 0, 1: 0, 2: 0}
    total_graphs = {0: 0, 1: 0, 2: 0}
    for idx, graph in enumerate(batch_data):
        num_nodes = graph.num_nodes
        if num_nodes in ranges["head"]:
            graph_group = 2
        elif num_nodes in ranges["med"]:
            graph_group = 1
        elif num_nodes in ranges["tail"]:
            graph_group = 0
        else:
            assert False
        if preds[idx].cpu().item() == labels[idx].cpu().item():
            graph_correct[graph_group] += 1
        total_graphs[graph_group] += 1

    out = list()
    out.append((correct_all, len(labels)))
    for group in graph_correct.keys():
        out.append((graph_correct[group], total_graphs[group]))
    return out



def train(batch_data):
    model.train()

    lam_val = np.random.beta(4.0, 4.0)
    batch_data = batch_data.to(device)
    optimizer.zero_grad()

    out = model(batch_data.x.cuda(),
                batch_data.edge_index.cuda(),
                lam_val,
                batch_data.ptr.cuda(),
                batch_data.batch.cuda())
    batch_loss = F.nll_loss(out, batch_data.y)
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item(), lam_val


def get_args():
    parser = argparse.ArgumentParser(
        description='PyTorch MixUp for Graph Classification')
    parser.add_argument('--dataset', type=str, default="PTC_MR",
                        help='name of dataset (default: PTC)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test data ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='valid data ratio')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dim of Model (Default: 256)')
    return parser.parse_args()


def pass_data(loader, ranges, lam=None, to_train=False):
    losses = []
    n_samples, head_samples, med_samples, tail_samples = 0, 0, 0, 0
    accs, head_accs, med_accs, tail_accs = [], [], [], []
    for data in loader:
        if to_train:
            loss, lam = train(data)
            losses.append(loss)
        acc = test(data, 1, ranges)
        accs.append(acc[0][0])
        n_samples += acc[0][1]
        head_accs.append(acc[1][0])
        head_samples += acc[1][1]
        med_accs.append(acc[2][0])
        med_samples += acc[2][1]
        tail_accs.append(acc[3][0])
        tail_samples += acc[3][1]

    out_acc = [sum(accs) / n_samples,
               sum(head_accs) / head_samples,
               sum(med_accs) / med_samples,
               sum(tail_accs) / tail_samples]
    if not to_train:
        lam =  None
    return lam, sum(losses) / len(loader), out_acc


if __name__ == "__main__":
    args = get_args()
    print(args)
    seeds = range(0, 5)
    test_record = torch.zeros(len(seeds))
    valid_record = torch.zeros(len(seeds))
    tail_record = torch.zeros(len(seeds))
    medium_record = torch.zeros(len(seeds))
    head_record = torch.zeros(len(seeds))

    for SEED in seeds:
        print("Training with SEED - ", SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)  # Numpy module.
        random.seed(SEED)  # Python random module.

        # load data
        d_name = args.dataset
        if d_name == "IMDB-BINARY" or d_name == "FRANKENSTEIN":
          transforms = OneHotDegree(1000)
        else:
          transforms = None
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', d_name)
        dataset = TUDataset(path, d_name, transform=transforms)

        if args.dataset == "PROTEINS":
            K = [0, 371, 742, 1113]
        elif args.dataset == "PTC_MR":
            K = [0, 115, 230, 344]
        elif args.dataset == "IMDB-BINARY":
            K = [0, 333, 666, 1000]
        elif args.dataset == "DD":
            K = [0, 393, 785, 1178]
        elif args.dataset == "FRANKENSTEIN":
            K = [0, 1445, 2890, 4337]

        nodes = torch.zeros(len(dataset))
        for i in range(len(dataset)):
            nodes[i] = dataset[i].num_nodes

        values, _ = torch.sort(nodes, descending=True)

        ranges = dict()

        ranges["head"] = list(set(values[K[0]:K[1]]))
        ranges["med"] = list(set(values[K[1]:K[2]]))
        ranges["tail"] = list(set(values[K[2]:K[3]]))

        train_size = 1 - (args.valid_ratio + args.test_ratio)
        test_size = args.test_ratio
        val_size = args.valid_ratio
        batch_size = args.batch_size

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

        print("TRAIN: ", len(dataset[:int(total_graphs * train_size)]))
        print("TEST: ", len(dataset[int(total_graphs * train_size): int(total_graphs * train_size) +
                                                                                 int(total_graphs * test_size)]))
        print("VAL: ", len(dataset[int(total_graphs * train_size) + int(total_graphs * test_size):]))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(hidden_channels=args.hidden_dim,
                    in_channel=dataset.num_node_features,
                    out_channel=dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        max_acc = 0
        best_accs = [0, 0, 0, 0]
        for epoch in range(1, args.epochs):
            lam, train_loss, train_accs = pass_data(train_loader, ranges, to_train=True)
            _, val_loss, val_accs = pass_data(val_loader, ranges, lam=lam)
            if val_accs[0] > max_acc:
                max_acc = val_accs[0]
                _, best_loss, best_accs = pass_data(test_loader, ranges, lam=lam)

            print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}\n'
                  f'Train Acc: {train_accs[0]:.4f}, Head Acc: {train_accs[1]:.4f}, Med Acc: {train_accs[2]:.4f}, Tail Acc: {train_accs[3]:.4f}\n' 
                  f'Val   Acc: {val_accs[0]:.4f}, Head Acc: {val_accs[1]:.4f}, Med Acc: {val_accs[2]:.4f}, Tail Acc: {val_accs[3]:.4f}\n' 
                  f'Test  Acc: {best_accs[0]:.4f}, Head Acc: {best_accs[1]:.4f}, Med Acc: {best_accs[2]:.4f}, Tail Acc: {best_accs[3]:.4f}\n\n')

        valid_record[SEED] = max_acc
        test_record[SEED] = best_accs[0]
        head_record[SEED] = best_accs[1]
        medium_record[SEED] = best_accs[2]
        tail_record[SEED] = best_accs[3]

    print('Valid mean: %.4f, std: %.4f' %
          (valid_record.mean().item(), valid_record.std().item()))
    print('Test mean: %.4f, std: %.4f' %
          (test_record.mean().item(), test_record.std().item()))
    print('Head mean: %.4f, std: %.4f' %
          (head_record.mean().item(), head_record.std().item()))
    print('Medium mean: %.4f, std: %.4f' %
          (medium_record.mean().item(), medium_record.std().item()))
    print('Tail mean: %.4f, std: %.4f' %
          (tail_record.mean().item(), tail_record.std().item()))

    with open("metrics.txt", "a") as txt_file:
        txt_file.write(f"Dataset: {args.dataset}, \n"
                       f"Valid Mean: {round(valid_record.mean().item(), 4)}, \n"
                       f"Std Valid Mean: {round(valid_record.std().item(), 4)}, \n"
                       f"Test Mean: {round(test_record.mean().item(), 4)}, \n"
                       f"Std Test Mean: {round(test_record.std().item(), 4)}, \n"
                       f"Head Mean: {round(head_record.mean().item(), 4)}, \n"
                       f"Std Head Mean: {round(head_record.std().item(), 4)}, \n"
                       f"Medium Mean: {round(medium_record.mean().item(), 4)}, \n"
                       f"Std Medium Mean: {round(medium_record.std().item(), 4)}, \n"
                       f"Tail Mean: {round(tail_record.mean().item(), 4)}, \n"
                       f"Std Tail Mean: {round(tail_record.std().item(), 4)} \n\n"
                       )
