import dgl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from dgl.data.utils import load_graphs


class Dataset:
    def __init__(self, name='yelp', device='cpu', prefix='./datasets/'):  # /root/autodl-tmp/dataset/
        self.name = name
        graph = load_graphs(prefix + name)[0][0]
        graph = dgl.add_self_loop(graph)
        self.graph = dgl.to_bidirected(graph, copy_ndata=True)
        self.graph.ndata['feature'] = self.graph.ndata['feature'].float()
        self.feature: torch.Tensor = self.graph.ndata['feature']
        self.label: torch.Tensor = self.graph.ndata['label']
        self.device = device
        self.adj = self.graph.edges()
        print(self.graph.ndata['feature'].shape)

    def process_data(self, train_ratio):
        nodes = np.arange(self.feature.shape[0])
        mask = np.zeros_like(nodes, dtype=bool)
        index = list(range(self.label.shape[0]))
        test_size = int((1 - train_ratio) * len(index))
        train_idx, test_idx, y_train, y_test = train_test_split(index, self.label, stratify=self.label,
                                                                test_size=test_size, shuffle=True)
        mask[train_idx] = True
        mask = torch.from_numpy(mask).to(device=self.device)
        self.graph.ndata['mask'] = mask
        self.graph.ndata['label'] = self.label
        return self.graph, train_idx, test_idx, y_train, y_test


def save_results(results, file_name):
    file_dir = f'./results/{file_name}.txt'
    f = open(file_dir, 'a+')
    f.write(f"Time:{results['Time']}\tAUROC: {results['AUROC']}\tAUPRC: {results['AUPRC']}\tF1-Macro: {results['f1_macro']}\tgmean: {results['gmean']}\n")
    f.close()
    print(f'save to file name: {file_name}')
