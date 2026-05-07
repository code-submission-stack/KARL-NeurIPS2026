import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from kan_layers import EfficientKANLayer # Our newly created B-Spline KAN layer

class MeanAggregator(nn.Module):
    """
    B-Spline KAN Aggregator for Graph Message Passing.
    Aggregates a node's embeddings using the mean of neighbors' embeddings, 
    then transforms the aggregated topological features using piecewise 
    B-spline univariate functions (EfficientKAN) instead of linear MLPs.
    """
    def __init__(self, features, cuda=False, gcn=False): 
        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
        # Calculate embedding dimension dynamically based on input features
        embed_dim = features(torch.tensor([0])).shape[1] 
        self.out_dim = embed_dim
        
        # ------------------------------------------------------------------
        # THE ARCHITECTURAL LEAP: Eliminating MLPs
        # Replacing standard nn.Parameter weight matrix with Efficient B-Spline KAN
        # grid_size=5 and spline_order=3 ensure stable local support on hub nodes
        # ------------------------------------------------------------------
        self.kan_weight = EfficientKANLayer(
            in_features=embed_dim, 
            out_features=self.out_dim, 
            grid_size=5, 
            spline_order=3
        )

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of target nodes in a batch
        to_neighs --- list of sets, each set is the neighbors for the target node
        num_sample --- number of neighbors to sample
        """
        # 1. Neighbor Sampling Logic (Standard GraphSAGE)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            # Include self in the aggregation if GCN mode is active
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
            
        # 2. Create the Adjacency Mask Matrix
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        
        if self.cuda:
            mask = mask.cuda()
            
        # Normalize the mask for Mean Aggregation
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        
        # Fetch initial embeddings for the unique nodes
        embed_matrix = self.features(torch.tensor(unique_nodes_list))
        if self.cuda:
            embed_matrix = embed_matrix.cuda()
            
        # Multiply mask with embeddings to get the aggregated neighbor features
        neigh_h = mask.mm(embed_matrix)
        
        # ------------------------------------------------------------------
        # B-SPLINE KAN FORWARD PASS 
        # (Replacing the old: return neigh_h.mm(self.weight))
        # ------------------------------------------------------------------
        
        # A) Apply tanh normalization to keep input features strictly within 
        # the KAN grid domain [-1, 1], preventing gradient explosion on scale-free graphs.
        normalized_neigh_h = torch.tanh(neigh_h) 
        
        # B) Pass through the learnable univariate B-Spline functions
        output = self.kan_weight(normalized_neigh_h)
        
        return output