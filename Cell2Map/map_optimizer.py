import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from torch.nn.functional import softmax, cosine_similarity
import torch.nn.functional as F

from . import utils as ut
from . import loss as ls
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (adjusted_rand_score, 
                             normalized_mutual_info_score,)

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Mapper:
    def __init__(self,
                 S,
                 G,
                 d,
                 a,
                 b,
                 D,
                 device='cpu',
                 lambda_d=0,
                 lambda_g1=1.0,  
                 lambda_g2=0,    
                 alpha=1,
                lambda_mahalanobis=0.7,
                lambda_distance=0.01,
                 adata_map=None,
                 ):
        
        self.S=torch.tensor(S,device=device, dtype=torch.float32)
        self.G=torch.tensor(G,device=device, dtype=torch.float32)
   
        self.target_cell_count=d is not None
        if self.target_cell_count:
            self.d=torch.tensor(d,device=device, dtype=torch.float32)

        self.lambda_d=lambda_d
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.alpha = alpha
        self.lambda_mahalanobis=lambda_mahalanobis
        self.lambda_distance=lambda_distance

        self.D = torch.tensor(D, device=device, dtype=torch.float32)
        self.a = torch.tensor(a, device=device, dtype=torch.float32)
        self.b = torch.tensor(b, device=device, dtype=torch.float32)
    
        M0 = a[:, None] * b[None, :] if adata_map is None else (1/np.sum(adata_map)) * adata_map
        self.M = torch.tensor(M0, device=device, requires_grad=True, dtype=torch.float32)

        self.d_loss = nn.L1Loss()
       

    def _loss_fn(self,verbose=True):
        M_probs = F.softmax(F.relu(self.M)+ 1e-10, dim=1) 
        M_probs = torch.clamp(M_probs, min=1e-10,max=1 - 1e-10)  

        if self.target_cell_count:
            d_pred = torch.log(M_probs.sum(axis=0) / self.M.shape[0])
            count_term=self.lambda_d*self.d_loss(d_pred, self.d)
        else:
            count_term=None

        G_pred = torch.matmul(M_probs.t(), self.S)
        euclidean_distance_loss = F.pairwise_distance(G_pred, self.G,p=2).mean()
        cov_matrix = torch.cov(torch.cat((G_pred, self.G), dim=0).t())
        inv_cov_matrix = torch.inverse(cov_matrix)

        diff = G_pred - self.G
        mahalanobis_distance_loss = torch.sqrt(torch.sum((diff @ inv_cov_matrix) * diff, dim=1)).mean()
       
        combined_distance_loss = (1-self.lambda_mahalanobis) * euclidean_distance_loss + self.lambda_mahalanobis * mahalanobis_distance_loss
        
        distance_term = self.lambda_distance * combined_distance_loss
        
       
        gv_term = self.lambda_g1 * cosine_similarity(G_pred, self.G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, self.G, dim=1).mean()

        expression_term1 = self.alpha * ls.exp_loss(self.D, M_probs/len(self.a))
      

        total_loss =-(gv_term + vg_term)+expression_term1-distance_term
        if count_term is not None:
            total_loss += count_term
    
        if verbose:
            print(
                f"Total_loss: {total_loss.item():.3f}, "
                f"gv_term: {gv_term.item():.3f}, "
                f"vg_term: {vg_term.item():.3f}, "
                f"density_term: {count_term.item():.3f}, "
                f"expression_term: {expression_term1.item():.3f}, "
                f"distance_term:{distance_term.item():.3f},"

                  )
        
        return total_loss, M_probs
    
    def train(self, num_epochs, learning_rate=0.1, print_each=100):
        seed_value = 1000
        print("seed:" + str(seed_value))
        seed_everything(seed_value)
        optimizer = torch.optim.AdamW([self.M], lr=learning_rate,weight_decay=1e-6) 

        scheduler = torch.optim.lr_scheduler. ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        """
        {
            "total_loss": [],
            "main_loss": [],
            "vg_reg": [],
            "kl_reg": [],
            "entropy_reg": []
        }
        """

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]
            result=run_loss[-1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)  

        with torch.no_grad():
            output =result.cpu().numpy()
        return output  