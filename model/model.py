import torch
import numpy as np


class SpectralFeature:
    def __init__(self, edges, node_num, anomaly_nodes: torch.Tensor, normal_nodes: torch.Tensor, theta, device):
        self.edges = edges
        self.graph: list[torch.sparse.Tensor] = []
        self.node_num = node_num
        self.relation_num = len(edges)
        for edge_index in self.edges:
            rows = edge_index[0]
            cols = edge_index[1]
            data = torch.ones(len(rows), dtype=torch.int, device=device)
            adj_matrix_sparse = torch.sparse_coo_tensor(torch.stack([rows, cols]), data, (self.node_num, self.node_num))
            self.graph.append(adj_matrix_sparse)
        # self.lamb = lamb
        self.theta = theta
        self.anomaly_nodes = anomaly_nodes
        self.normal_nodes = normal_nodes
        self.train_nodes = torch.hstack((anomaly_nodes, normal_nodes))
        self.device = device
        self.homo_list = [self.homophilic(relation) for relation in range(self.relation_num)]

    def embedding(self, feature: torch.Tensor):
        fea_list = []
        n = self.node_num
        k = 1
        graph_change = self.change_firm_adj()
        for relation in range(self.relation_num):
            # 计算度矩阵（对角矩阵）
            graph_layer = graph_change[relation]
            degree_matrix_data = torch.sum(graph_layer, dim=1).to_dense()
            degree_matrix_sparse = torch.sparse_coo_tensor(torch.stack([torch.arange(n, device=self.device),
                                                                         torch.arange(n, device=self.device)]),
                                                           degree_matrix_data, (n, n))
            homo = self.homo_list[relation]
            # homo = np.tanh(homo)
            L = degree_matrix_sparse + self.theta * homo * graph_layer
            L_sym = (degree_matrix_data ** (-1 / 2)) * L * (degree_matrix_data ** (-1 / 2))
            h1 = L_sym @ feature
            h_final = torch.cat([feature, h1], -1)
            fea_list.append(h_final)
        return fea_list

    def change_firm_adj(self):
        graph_change = []
        for layer1 in range(self.relation_num):
            graph_result = self.graph[layer1].clone().float()
            for layer2 in range(self.relation_num):
                if layer1 == layer2:
                    continue
                graph_intersect = self.graph[layer1] * self.graph[layer2]
                graph_result += graph_intersect * self.compute_sim(layer1, layer2)
            graph_change.append(graph_result)
        return graph_change

    def compute_sim(self, layer1, layer2):
        graph_union = self.graph[layer1] + self.graph[layer2]
        graph_intersect = self.graph[layer1] * self.graph[layer2]
        sim = graph_intersect._nnz() / graph_union._nnz()
        return sim

    def homophilic(self, layer):
        edge_layer = np.array([self.edges[layer][0].cpu().numpy(), self.edges[layer][1].cpu().numpy()]).T
        edge_train = edge_layer[np.isin(edge_layer, self.train_nodes.cpu()).all(axis=1)]
        anomaly_homo = np.isin(edge_train, self.anomaly_nodes.cpu()).all(axis=1)
        anomaly_edge = edge_train[anomaly_homo]  # 408
        normal_homo = np.isin(edge_train, self.normal_nodes.cpu()).all(axis=1)
        normal_edge = edge_train[normal_homo]
        homophilic_score = (anomaly_edge.shape[0] + normal_edge.shape[0])/edge_train.shape[0]
        return homophilic_score
