import numpy as np
import pandas as pd
import random
import argparse
from sklearn.model_selection import KFold
import torch
import networkx as nx
from gnn.data_cgcnn_vector import CIFData, CIF_Lister
from torch_geometric.utils import to_networkx
from model_gnn import GNN_Graph
import os
import time
from gnn.Explainer import Explainer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
setup_seed(1)
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='EGIN')
parser.add_argument('--property', default='sample_data',
                        help='property to train')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--model_file', type=str, default='./trained_model/sample_data_bst_0.pth',
                    help='filename to read the model (if there is any)')
args = parser.parse_args()
data_path = './data/sample_data'

def main():
    CRYSTAL_DATA = CIFData(data_path)
    id_prop_file = os.path.join(data_path, 'id_prop.csv')
    dataset = pd.read_csv(id_prop_file, names=['material_ids', 'label'])
    class Normalizer(object):

        def __init__(self, tensor):
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)

        def norm(self, tensor):
            return (tensor - self.mean) / self.std

        def denorm(self, normed_tensor):
            return normed_tensor * self.std + self.mean

        def state_dict(self):
            return {'mean': self.mean,
                    'std': self.std}

        def load_state_dict(self, state_dict):
            self.mean = state_dict['mean']
            self.std = state_dict['std']
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=18012019)
    data_induce = np.arange(0, len(dataset))
    print('--------------------------------')
    fold = 0
    for train_val_idx, test_idx in kfold.split(data_induce):

        print(f'FOLD {fold}')
        print('--------------------------------')
        test_dataset = CIF_Lister(test_idx, CRYSTAL_DATA, df=dataset)
        graph = test_dataset[0]
        print(graph.label)
        model = GNN_Graph(num_layer=3,
                          num_classes=1,
                          emb_dim=64,
                          drop_ratio=0.0)
        if not args.model_file == "":
            checkpoint_ = torch.load(args.model_file)
            model.load_state_dict(checkpoint_['model_state_dict'])
        explainer = Explainer(model, epochs=200, lr=0.001, return_type='raw')
        node_feat_imp = np.zeros(92)
        node_feat_mask, edge_mask = explainer.explain_graph(graph)

        ax, G = explainer.visualize_subgraph(-1, graph.edge_index,
                                             edge_mask,
                                             y=graph.label)
        feat_importance = node_feat_mask.cpu().numpy()
        f1 = feat_importance[0:17].sum() / 18
        f2 = feat_importance[18:26].sum() / 9
        f3 = feat_importance[27:35].sum() / 9
        f4 = feat_importance[36:45].sum() / 10
        f5 = feat_importance[46:57].sum() / 12
        f6 = feat_importance[58:67].sum() / 10
        f7 = feat_importance[68:77].sum() / 10
        f8 = feat_importance[78:81].sum() / 4
        f9 = feat_importance[82:91].sum() / 10
        feat_importance_ = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        feat_labels = ['group number', 'period number', 'electronegativity', 'covalent radius', 'valence electrons',
                       'first ionization energy', 'electron affinity', 'block', 'atomic volume']
        df = pd.DataFrame({'feat_importance': feat_importance_},
                          index=feat_labels)
        df = df.sort_values("feat_importance", ascending=False)
        df = df.round(decimals=3)
        ax = df.plot(
            kind='barh',
            figsize=(5, 5),
            xlim=[0, float(max(feat_importance_)) + 0.2],
            legend=False,
        )
        plt.xlabel('Feature importance', fontsize=10)
        plt.ylabel('Feature label', fontsize=10)
        ax.tick_params(axis='x', tickdir='in', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        plt.tick_params(bottom=True, top=False, left=False, right=False)
        plt.gca().invert_yaxis()
        plt.gcf().subplots_adjust(left=0.25, top=0.91, bottom=0.09)
        ax.bar_label(container=ax.containers[0], label_type='edge')
        plt.show()
        node_feat_imp += node_feat_mask.cpu().numpy()
        bi_node_mask = edge_mask.reshape(6, -1)
        node_mask = torch.mean(bi_node_mask, dim=0)
        node_mask = node_mask.detach().cpu().numpy()

        def visualize(h, color, epoch=None, loss=None):
            plt.figure(figsize=(5,5))
            plt.xticks([])
            plt.yticks([])

            if torch.is_tensor(h):
                h = h.detach().cpu().numpy()
                plt.scatter(h[:, 0], h[:, 1], s=200, c=color, cmap="Set1")
                if epoch is not None and loss is not None:
                    plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=15)
            else:
                nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,linewidths=None,
                                 node_color=color, cmap="Set1", font_size=11, node_size=390)
            plt.show()
        def show_cmap(cmap, norm=None, extend=None):
            '''展示一个colormap.'''
            if norm is None:
                norm = mcolors.Normalize(vmin=0, vmax=cmap.N)
            im = cm.ScalarMappable(norm=norm, cmap=cmap)

            fig, ax = plt.subplots(figsize=(6,1))
            fig.subplots_adjust(bottom=0.5)
            fig.colorbar(im, cax=ax, orientation='horizontal', extend=extend)
            plt.axis('off')
            ax.axis('off')

        G = to_networkx(graph, to_undirected=True)
        bottom = cm.get_cmap('Blues_r', 256)
        top = cm.get_cmap('Oranges', 256)
        newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
        newcmp = ListedColormap(newcolors, name='OrangeBlue')
        show_cmap(newcmp)
        atom_color = [(newcmp(node_mask[idx])) for idx in range(len(node_mask))]
        visualize(G, color=atom_color)
        if fold == 0:
            break

if __name__ == "__main__":
    main()


