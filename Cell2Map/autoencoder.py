import numpy as np
from typing import Optional, Tuple
from anndata import AnnData
import torch
import torch.nn as nn
import scipy.sparse as sp
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size,OptTensor)
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import random
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from torch_geometric.data import Data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def merge_sc_and_sp_data(sc_data, sp_data):
   
    sc_x = sc_data.x  
    sp_x = sp_data.x  

    
    combined_x = torch.cat([sc_x, sp_x], dim=0)  
    sc_edge_index = sc_data.edge_index  
    sp_edge_index = sp_data.edge_index + sc_x.size(0)  
    combined_edge_index = torch.cat([sc_edge_index, sp_edge_index], dim=1)

    combined_data = Data(x=combined_x, edge_index=combined_edge_index)
    return combined_data
def embedding_feature(sc_adata,spatial_adata,k_cutoff=7, self_loop=True):
    # Normalization
    set_seed(100) 
    device = torch.device('cuda')
    
    sc_adata=Cal_Spatial_Net(sc_adata, k_cutoff=k_cutoff)
   
    spatial_adata=Cal_Spatial_Net_Radius(spatial_adata,radius_cutoff=5)

    sc_adata.X = sp.csr_matrix(sc_adata.X)
    spatial_adata.X = sp.csr_matrix(spatial_adata.X)

    if 'highly_variable' in sc_adata.var.columns:
        sc_adata_Vars = sc_adata[:, sc_adata.var['highly_variable']]
    else:
        sc_adata_Vars = sc_adata

    if 'highly_variable' in spatial_adata.var.columns:
        sp_adata_Vars = spatial_adata[:, spatial_adata.var['highly_variable']]
    else:
        sp_adata_Vars = spatial_adata

    sc_data = Transfer_pytorch_Data(sc_adata_Vars)
    sp_data = Transfer_pytorch_Data(sp_adata_Vars)

    data=merge_sc_and_sp_data(sc_data,sp_data)


    model = GATE(hidden_dims=[data.x.shape[1]] + [512,64]).to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

   
    n_epochs = 500
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()


    model.eval()
    z, out = model(data.x, data.edge_index)

    n_sc_cells = sc_adata.n_obs
    
    STAGATE_rep = z.to('cpu').detach().numpy()
    embeddings_sc = STAGATE_rep[:n_sc_cells]
    embeddings_st = STAGATE_rep[n_sc_cells:]

    sc_adata.obsm['embedding'] = embeddings_sc
    spatial_adata.obsm['embedding'] = embeddings_st

    ReX = out.to('cpu').detach().numpy()
    ReX[ReX < 0] = 0
    sc_STAGE = ReX[:n_sc_cells]
    st_STAGE = ReX[n_sc_cells:]
    sc_adata.obsm['STAGATE_ReX'] = sc_STAGE
    spatial_adata.obsm['STAGATE_ReX'] = st_STAGE
    return sc_adata,spatial_adata

def Transfer_pytorch_Data(adata):

    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def Cal_Spatial_Net(adata: AnnData,
                    k_cutoff: Optional[Union[None, int]] = None,
                    return_data: Optional[bool] = True,
                    verbose: Optional[bool] = True
                    ) -> AnnData:
    if 'spatial' not in adata.obsm.keys():
        if 'x' in adata.obs.keys() and 'y' in adata.obs.keys():
            spatial_coords = torch.tensor([adata.obs['x'], adata.obs['y']]).T
            edge_index = knn_graph(x=spatial_coords, flow='target_to_source',
                               k=k_cutoff, loop=True, num_workers=8)
        else:
            edge_index = knn_graph(x=torch.tensor(adata.obsm['X_pca'].copy()), flow='target_to_source',
                               k=k_cutoff, loop=True, num_workers=8)
    else:
        edge_index = knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source',
                               k=k_cutoff, loop=True, num_workers=8)
   
    edge_index = to_undirected(edge_index, num_nodes=adata.shape[0])

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
   
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
   
    adata.uns['Spatial_Net'] = graph_df

    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0] / adata.n_obs} neighbors per cell on average.')

    if return_data:
        return adata

def Cal_Spatial_Net_Radius(adata: AnnData,
                           radius_cutoff: float,
                           return_data: Optional[bool] = True,
                           verbose: Optional[bool] = True
                           ) -> AnnData:
    if 'spatial' not in adata.obsm.keys():
        if 'x' in adata.obs.keys() and 'y' in adata.obs.keys():
            spatial_coords = np.vstack([adata.obs['x'], adata.obs['y']]).T
    else:
        spatial_coords = adata.obsm['spatial']
    tree = cKDTree(spatial_coords)
    pairs = tree.query_pairs(r=radius_cutoff, output_type='set')

  
    graph_df = pd.DataFrame(list(pairs), columns=['Cell1', 'Cell2'])

  
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    
   
    adata.uns['Spatial_Net'] = graph_df

    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{graph_df.shape[0] / adata.n_obs} neighbors per cell on average.')
    
    if return_data:
        return adata

class GATE(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(GATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  
    def _build_loss(self, x, recons_x):
        size = x.shape[0]
        return torch.norm(x - recons_x, p='fro')**2 / size

class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


