import csv
import json
from pymatgen.core.structure import Structure
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os
import numpy as np
import torch


class GaussianDistance(object):

    def __init__(self, dmin, dmax, step, var=None):

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):

        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)



class AtomInitializer(object):

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {key: value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)



class CIFData(Dataset):
    
    def __init__(self, root_dir, max_num_nbr=6, radius=8, step=0.2,
                 random_seed=42):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        atom_init_file = os.path.join(self.root_dir, 'cgcnn-embedding.json')
        assert os.path.exists(atom_init_file), 'cgcnn-embedding.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=0, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        print('cif_id', cif_id)
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(str(crystal[i].specie))
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        label = np.vstack([str(crystal[i].specie.number)
                           for i in range(len(crystal))])
        label = list(map(int, label))
        label = torch.LongTensor(label)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id, label

    def format_adj_matrix(self, adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes, all_dst_nodes), dim=0)


class CIF_Lister(Dataset):
    def __init__(self, crystals_ids, full_dataset, df=None):
        self.crystals_ids = crystals_ids
        self.full_dataset = full_dataset
        self.material_ids = df.iloc[crystals_ids].values[:, 0].squeeze()

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self, original_dataset):
        names = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i = self.crystals_ids[idx]
        material = self.full_dataset[i]

        n_features = material[0][0]
        e_features = material[0][1]
        e_features = e_features.view(-1, 41)
        a_matrix = material[0][2]
        y = material[1]
        cif_id = material[2]
        label = material[3]
        graph_crystal = Data(x=n_features, y=y, edge_index=a_matrix,
                             edge_attr=e_features, cif_id=cif_id, label=label)
        return graph_crystal