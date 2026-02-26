import numpy as np
import scipy
import torch

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]
def euclidean_distances(X, Y, squared=False):
    a2 = np.einsum('ij,ij->i', X, X)
    b2 = np.einsum('ij,ij->i', Y, Y)

    c = -2 * np.dot(X, Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = np.maximum(c, 0)

    if not squared:
        c = np.sqrt(c)

    if X is Y:
        c = c * (1 - np.eye(X.shape[0], dtype=c.dtype))

    return c




def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
   
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result



def pcc_distances(v1, v2):
    if v1.shape[1] != v2.shape[1]:
        raise ValueError("The two matrices v1 and v2 must have equal dimensions; two slice data must have the same genes")

    n = v1.shape[1]
    sums = np.multiply.outer(v1.sum(1), v2.sum(1))
    stds = np.multiply.outer(v1.std(1), v2.std(1))
    correlation = (v1.dot(v2.T) - sums / n) / stds / n
    distances=1-correlation
    return distances

def kl_divergence_backend(X, Y):
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/np.sum(X,axis=1, keepdims=True)
    Y = Y/np.sum(Y,axis=1, keepdims=True)
    log_X = np.log(X)
    log_Y = np.log(Y)
    X_log_X = np.einsum('ij,ij->i',X,log_X)
    X_log_X = np.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - np.dot(X,log_Y.T)
    return D

def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    p=p/len(p)
    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b

    constC1 = np.dot(
        np.dot(f1(C1), np.reshape(p, (-1, 1))),
        np.ones((1, len(q)), dtype=q.dtype)
    )

    constC2 = np.dot(
        np.ones((len(p), 1), dtype=p.dtype),
        np.dot(np.reshape(q, (1, -1)), f2(C2).T)
    )

    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def annotate_gene_sparsity(adata):
  
    mask = adata.X != 0         
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs      
    gene_sparsity = np.asarray(gene_sparsity)  

    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))      
    adata.var["sparsity"] = gene_sparsity