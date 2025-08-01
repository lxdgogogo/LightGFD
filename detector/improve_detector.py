import dgl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
from imblearn.metrics import geometric_mean_score
import xgboost as xgb


class GraphDetector:
    def __init__(self, graph: dgl.DGLGraph, train_config: dict):
        n_estimators = 100 if train_config['n_estimators'] is None else train_config['n_estimators']
        learning_rate = 0.2 if train_config['learning_rate'] is None else train_config['learning_rate']
        max_depth = 12 if train_config['max_depth'] is None else train_config['max_depth']
        nodes = graph.nodes()
        labels = graph.ndata["label"]
        mask = graph.ndata["mask"]
        feature = graph.ndata["feature"]
        anomaly_nodes = nodes[((labels == 1) & (mask == 1))].numpy()
        normal_nodes = nodes[((labels == 0) & (mask == 1))].numpy()
        train_nodes = np.hstack((anomaly_nodes, normal_nodes))
        self.edges = graph.edges()
        self.homo = self.homophilic(train_nodes, anomaly_nodes, normal_nodes)
        print("homo", self.homo)
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                       eval_metric=roc_auc_score, max_depth=max_depth, scale_pos_weight=20)
        self.node_num = feature.shape[0]
        self.feature_final = self.message_passing(feature, self.edges)

    def message_passing(self, feature, edges):
        n = feature.shape[0]
        rows = edges[0]
        cols = edges[1]
        data = torch.ones(len(rows), dtype=torch.int)
        adj_matrix_sparse = torch.sparse_coo_tensor(torch.stack([rows, cols]), data, (self.node_num, self.node_num))

        degree_matrix_data = torch.sum(adj_matrix_sparse, dim=1).to_dense()
        i_sparse = torch.sparse_coo_tensor(torch.stack([torch.arange(n), torch.arange(n)]),
                                           torch.ones(n), (n, n))
        L = i_sparse + self.homo * adj_matrix_sparse
        L_sym = L * (degree_matrix_data ** (-1.0))
        h1 = L_sym @ feature
        h_final = torch.cat([feature, h1], -1)
        return h_final

    def train(self, train_idx, train_y, test_idx):
        x_trainScaled = self.feature_final[train_idx].cpu().numpy()
        y_trainScaled = train_y.cpu().numpy()
        self.model.fit(x_trainScaled, y_trainScaled)
        pred_y = self.model.predict_proba(self.feature_final[test_idx].cpu().numpy())
        pred_y = pred_y[:, 1]
        return pred_y

    def homophilic(self, train_nodes, anomaly_nodes, normal_nodes):
        edge_layer = np.array([self.edges[0].numpy(), self.edges[1].numpy()]).T
        edge_train = edge_layer[np.isin(edge_layer, train_nodes).all(axis=1)]
        anomaly_homo = np.isin(edge_train, anomaly_nodes).all(axis=1)
        anomaly_edge = edge_train[anomaly_homo]  # 408
        normal_homo = np.isin(edge_train, normal_nodes).all(axis=1)
        normal_edge = edge_train[normal_homo]
        homophilic_score = (anomaly_edge.shape[0] + normal_edge.shape[0]) / edge_train.shape[0]
        return homophilic_score


def eval(labels, probs: torch.Tensor):
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        AUROC = roc_auc_score(labels, probs)
        AUPRC = average_precision_score(labels, probs)
        labels = np.array(labels)
        k = int(labels.sum())
    RecK = sum(labels[probs.argsort()[-k:]]) / sum(labels)
    pred = probs.copy()
    pred[probs >= 0.5] = 1
    pred[probs < 0.5] = 0
    f1_micro = f1_score(labels, pred, average='micro')
    f1_macro = f1_score(labels, pred, average='macro')
    # label_pred = np.where(probs >= 0.5, 1, 0)
    # label_true = np.sum(label_pred == labels)
    recall = recall_score(labels, pred)
    g_mean = geometric_mean_score(labels, pred)
    return AUROC, AUPRC, RecK, f1_micro, f1_macro, recall, g_mean
