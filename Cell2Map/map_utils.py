import scanpy as sc
import logging
import numpy as np
import pandas as pd
import torch
import random
from scipy.sparse import csc_matrix, csr_matrix
from sklearn import preprocessing
from torch.autograd import Variable
from torch.nn.functional import softmax, cosine_similarity
from . import autoencoder as au
from . import map_optimizer as mo
from . import utils as ut
import anndata
import Celloc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, 
                             normalized_mutual_info_score, 
                             fowlkes_mallows_score, 
                             homogeneity_score)

def process_adatas(sc_adata, spatial_adata,genes=None,gene_to_lowercase=True):
    sc.pp.filter_genes(sc_adata, min_cells=1)
    sc.pp.filter_genes(spatial_adata, min_cells=1)         
    
    
    if genes is None:
        genes=sc_adata.var_names
    if gene_to_lowercase:
        sc_adata.var_names= [g.lower() for g in sc_adata.var_names]
        spatial_adata.var_names = [g.lower() for g in spatial_adata.var_names]
        genes = list(g.lower() for g in genes)

    sc_adata.var_names_make_unique()       
    spatial_adata.var_names_make_unique()

    intersect_genes = list(set(sc_adata.var_names) & set(spatial_adata.var_names)&set(genes))
    
    sc_adata=sc_adata[:,intersect_genes]
    spatial_adata=spatial_adata[:,intersect_genes]

    sc.pp.normalize_total(sc_adata, target_sum=1e6)
    sc.pp.log1p(sc_adata)
   
    sc.pp.normalize_total(spatial_adata, target_sum=1e6)
    sc.pp.log1p(spatial_adata)
    sc.tl.pca(sc_adata, n_comps=50)
    rna_count_per_spot = np.array(spatial_adata.X.sum(axis=1)).squeeze()    
    spatial_adata.obs['rna_count_based_density']=rna_count_per_spot / np.sum(rna_count_per_spot)
    return sc_adata,spatial_adata




def map_cell_to_space(sc_adata,
                      spatial_adata,
                      learning_rate=0.005,
                      num_epochs=1500,
                      lambda_d=0.8,
                      lambda_g1=1.5,# gene-voxel cos sim
                      lambda_g2=0.8,# voxel-gene cos sim
                      b_init=None,
                      alpha=2,
                      lambda_mahalanobis=0.7,
                      lambda_distance=0.01,
                      verbose=True,
                      cell_count_prior='cell_counts',
                      device='cpu',
                    ):

    if(type(cell_count_prior)is str) and (cell_count_prior not in["cell_counts",None]):
        raise ValueError("cell_count_prior must be 'cell_counts' or None")
    if cell_count_prior is not None and(lambda_d==0 or lambda_d is None):
        lambda_d=1
    if lambda_d>0 and cell_count_prior is None:
        raise ValueError("When lambda_d is set, please define the cell_count_prior.")
    
    logging.info("Allocate tensors for mapping.")
    
    if cell_count_prior=="cell_counts":
        cell_count_prior=spatial_adata.obs["rna_count_based_density"]
    d=cell_count_prior

    S=sc_adata.obsm['embedding']
    G=spatial_adata.obsm['embedding']
    a=np.ones(S.shape[0])
    if b_init is not None:
        b=np.array(b_init)/S.shape[0]
    else:
        b=np.ones((G.shape[0],))/S.shape[0]

    sc_X = ut.to_dense_array(ut.extract_data_matrix(sc_adata, rep=None))
    sp_X = ut.to_dense_array(ut.extract_data_matrix(spatial_adata, rep=None))
    s_sc = sc_X+ 0.01
    s_sp = sp_X+ 0.01
    D=ut.kl_divergence_backend(s_sc, s_sp)+ut.pcc_distances(S,G)
        
    device = torch.device(device)  # for gpu
    if verbose:
        print_each = 100
    else:
        print_each = None
    hyperparameters = {
        "lambda_d": lambda_d,
        "lambda_g1": lambda_g1,
        "lambda_g2": lambda_g2,
        "alpha":alpha,
        "lambda_mahalanobis":lambda_mahalanobis,
        "lambda_distance":lambda_distance,
    }
    mapper=mo.Mapper(S=S,G=G,d=d,a=a,b=b,D=D,device=device,**hyperparameters)
    mapping_matrix = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )
    
    logging.info("Saving results..")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=sc_adata.obs.copy(),
        var=spatial_adata.obs.copy(),
    )
    return adata_map