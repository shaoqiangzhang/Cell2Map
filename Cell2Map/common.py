import numpy as np
import pandas as pd
import anndata as ad
import warnings
import datatable as dt
import scanpy as sc
import h5py

from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, 
                             normalized_mutual_info_score, 
                             fowlkes_mallows_score, 
                             homogeneity_score)
def read_file(file_path):
    # Read file
    try:
        file_delim = "," if file_path.endswith(".csv") else "\t"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            file_data = dt.fread(file_path, header=True)
            colnames = pd.read_csv(file_path, sep=file_delim, nrows=1, index_col=0).columns
            rownames = file_data[:, 0].to_pandas().values.flatten()
            file_data = file_data[:, 1:].to_pandas()
            file_data.index = rownames
            file_data.columns = colnames
   
    except Exception as e:
        raise IOError("Make sure you provided the correct path to input files. "
                      "The following input file formats are supported: .csv with comma ',' as "
                      "delimiter, .txt or .tsv with tab '\\t' as delimiter.")

    return file_data
def read_data(scRNA_path, cell_type_path, st_path, coordinates_path,celltype_col):
    if st_path.endswith("h5ad"):
        spatial_adata= sc.read_h5ad(st_path)
        if 'in_tissue' in spatial_adata.obs.columns:
            spatial_adata=spatial_adata[spatial_adata.obs['in_tissue']==1]        
        coordinates_data=pd.DataFrame(spatial_adata.obsm['spatial'],index=spatial_adata.obs_names,columns=['X','Y'])
        spatial_adata.obs_names=['SPOT_'+str(col) for col in spatial_adata.obs_names]
        spatial_adata.var_names=['GENE_'+str(col) for col in spatial_adata.var_names]
    elif st_path.endswith("h5"):
        with h5py.File(st_path, 'r') as file:
            print("Datasets in the file:")
            def printname(name):
                print(name)
            file.visit(printname)
            data = file['st_path'][:]  
            coordinates_data = pd.read_csv(coordinates_path, index_col=0)  
            
            spatial_adata = ad.AnnData(data)
            spatial_adata.obsm['spatial'] = np.array(coordinates_data)
    else:   
        st_data = read_file(st_path)
        st_data = st_data[~st_data.index.duplicated(keep=False)]
        coordinates_data = read_file(coordinates_path)
        
        st_data.columns = ['SPOT_'+str(col) for col in st_data.columns]
        st_data.index = ['GENE_'+str(idx) for idx in st_data.index]
        spatial_adata=ad.AnnData(st_data.T)
        spatial_adata.obsm['spatial']=np.array(coordinates_data)
    if scRNA_path.endswith("h5ad"):
        sc_adata = sc.read_h5ad(scRNA_path)
        sc_adata.obs_names=['CELL_'+str(col) for col in sc_adata.obs_names]
        sc_adata.var_names=['GENE_'+str(col) for col in sc_adata.var_names]
        sc_adata.obs['CellType']=['TYPE_'+str(cell) for cell in list(sc_adata.obs[celltype_col])]
    else:
        scRNA_data = read_file(scRNA_path)
        scRNA_data.columns = ['CELL_'+str(col) for col in scRNA_data.columns]
        scRNA_data.index = ['GENE_'+str(idx) for idx in scRNA_data.index]
        scRNA_data = scRNA_data[~scRNA_data.index.duplicated(keep=False)]

        cell_type_data = read_file(cell_type_path)
        cell_type_data.index = ['CELL_'+str(idx) for idx in cell_type_data.index]
       
    
        sc_adata=ad.AnnData(scRNA_data.T)
        sc_adata.obs=cell_type_data
        sc_adata.obs['CellType']=['TYPE_'+str(cell) for cell in list(sc_adata.obs[celltype_col])]

    return sc_adata, spatial_adata, coordinates_data

def normalize_data(data):
    data = np.nan_to_num(data).astype(float)
    data *= 10**6 / np.sum(data, axis=0, dtype=float)
    np.log2(data + 1, out=data)
    np.nan_to_num(data, copy=False)
    return data
#估计每个spot中的细胞数量
def estimate_cell_number_RNA_reads(st_data, mean_cell_numbers):
    # Read data
    expressions = st_data.values.astype(float)

    # Data normalization
    expressions_tpm_log = normalize_data(expressions)

    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)

    min_cell_numbers = 1 if min_RNA_reads > 0 else 0

    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)

    return cell_number_to_node_assignment



def read_true_labels_from_file(file_path):
   
    # 假设文件中有一列名为 'cell_type'
    df = pd.read_csv(file_path)  # 或者使用 pd.read_table(file_path) 适用于 TSV 文件
    true_labels = df['CellType'].values  # 假设标签列名为 'cell_type'
    return true_labels