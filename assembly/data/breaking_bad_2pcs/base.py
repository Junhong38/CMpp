import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import Literal, List
import trimesh

class BreakingBad2PcsBase(Dataset):
    """
    Base Dataset for two-piece fractured surface samples.
    Each sample contains exactly two parts (piece_i and piece_j) and per-face "shared_faces" masks.
    """
    def __init__(
        self,
        split: Literal['train', 'val'] = 'train',
        data_root: str = 'data/data_shared_surface/data_pairwise_surface.hdf5',
        category: Literal['everyday', 'artifact', 'all'] = 'everyday',
        num_points_to_sample: int = 8192,
        mesh_sample_strategy: Literal['uniform', 'poisson'] = 'poisson',
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.category = category
        self.num_points_to_sample = num_points_to_sample
        self.mesh_sample_strategy = mesh_sample_strategy
        # Load list of fracture keys
        self.data_list = self.get_data_list()
        
    def get_data_list(self) -> List[str]:
        """
        Return list of sample keys, e.g. 'everyday/BeerBottle/.../fractured_63/0_1'
        """
        f = h5py.File(self.data_root, 'r')
        # gather object paths
        if self.category == 'all':
            raw = list(f['data_split']['everyday'][self.split]) + list(f['data_split']['artifact'][self.split])
        else:
            raw = list(f['data_split'][self.category][self.split])
        # decode to strings and flatten object/pair keys
        obj_list = [s.decode('utf-8') for s in raw]
        full_list: List[str] = []
        for obj in obj_list:
            grp = f[obj]
            for pair_key in grp.keys():
                full_list.append(f"{obj}/{pair_key}")
        f.close()
        return full_list

    def __len__(self) -> int:
        return len(self.data_list)

    def get_data(self, index: int) -> dict:
        """
        Load raw mesh and shared face masks for a given sample.
        """
        # full path includes object and pair key, e.g. 'everyday/.../fractured_63/0_1'
        pair_path = self.data_list[index]
        f = h5py.File(self.data_root, 'r')
        pair = f[pair_path]
        # load first piece
        piece_i = pair['piece_i']
        mesh_i = {
            'vertices': np.array(piece_i['vertices'][:]),
            'faces':    np.array(piece_i['faces'][:])
        }
        shared_i = np.array(piece_i['shared_faces'][:]) if 'shared_faces' in piece_i else np.array([])
        # read piece name
        name_i = piece_i['name'][()].decode('utf-8') if 'name' in piece_i else 'piece_i'
        # load piece_j
        piece_j = pair['piece_j']
        mesh_j = {
            'vertices': np.array(piece_j['vertices'][:]),
            'faces':    np.array(piece_j['faces'][:])
        }
        shared_j = np.array(piece_j['shared_faces'][:]) if 'shared_faces' in piece_j else np.array([])
        name_j = piece_j['name'][()].decode('utf-8') if 'name' in piece_j else 'piece_j'
        f.close()
        return {
            'index': index,
            'name': pair_path,
            'meshes': [mesh_i, mesh_j],
            'shared_faces': [shared_i, shared_j],
            'pieces': [name_i, name_j]
        }

    def sample_points(self, meshes: List[dict], shared_faces: List[np.ndarray]):
        """
        Abstract method to sample points on the two meshes.
        Should return (pointclouds, normals, fracture_masks).
        Override this in subclasses.
        """
        raise NotImplementedError

    def transform(self, data: dict) -> dict:
        """
        Abstract transform on sampled point clouds (e.g., normalization, augmentation).
        Override in subclasses.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict:
        data = self.get_data(index)
        return self.transform(data)

    def _pad_data(self, input_data):
        """Pad array to shape [2, ...]."""
        d = np.array(input_data)
        pad_shape = (2,) + tuple(d.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=d.dtype)
        pad_data[: d.shape[0]] = d
        return pad_data 