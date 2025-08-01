import argparse
import sys
from time import time

sys.path.append('')
from detector.improve_detector import GraphDetector, eval
from utils.utils import Dataset, save_results


def train(config: dict, graph, train_data: dict):
    dataset = config['dataset']
    file_name = f"{dataset}_{config['ratio']}"
    start = time()
    detector = GraphDetector(graph, config)
    end1 = time()
    print(end1 - start)
    pred_y = detector.train(train_data['train_idx'], train_data['y_train'], train_data['test_idx'])
    AUROC, AUPRC, RecK, f1_micro, f1_macro, recall, g_mean = eval(train_data['y_test'], pred_y)
    end = time()
    print(end - end1)
    model_result = {}
    model_result['AUROC'] = AUROC
    model_result['f1_macro'] = f1_macro
    model_result['gmean'] = g_mean
    model_result['Time'] = end - start
    model_result['AUPRC'] = AUPRC
    save_results(model_result, file_name)
    return model_result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightMGFD')
    parser.add_argument("--dataset", type=str, default="weibo",
                        help="Dataset for this model")
    parser.add_argument("--train_ratio", type=float, default=0.05, help="Training ratio")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--max_depth", type=int, default=12, help="Maximum tree depth of XGBoost")
    parser.add_argument("--T", type=int, default=200, help="Number of estimator trees of XGBoost")
    parser.add_argument("--device", type=str, default='cuda', help="Device (cpu/cuda)")
    args = parser.parse_args()
    train_config = {'n_estimators': args.T, 'learning_rate': args.lr, 'max_depth': args.max_depth,
                    'dataset': args.dataset, 'device': args.device, 'ratio': args.train_ratio}
    dataset = train_config['dataset']
    data = Dataset(dataset, train_config['device'])
    graph, train_idx, test_idx, y_train, y_test = data.process_data(train_config['ratio'])
    train_data = {'train_idx': train_idx, 'test_idx': test_idx, 'y_train': y_train, 'y_test': y_test}
    train(train_config, graph, train_data)

