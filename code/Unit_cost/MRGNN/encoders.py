import torch
import torch.nn as nn
from kan_layers import EfficientKANLayer

class Encoder(nn.Module):
    """
    B-Spline KAN Encoder for Graph Neural Networks
    """
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=10, base_model=None, gcn=False, cuda=False, feature_transform=False): 
        super(Encoder, self).__init__()
        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.base_model = base_model
        self.gcn = gcn
        self.cuda = cuda
        
        # ELIMINATE MLPs: Replace standard nn.Parameter projection with EfficientKANLayer
        # If GCN is false, it concatenates self + neighbor features, so input dim is feat_dim * 2
        input_dim = self.feat_dim if self.gcn else 2 * self.feat_dim
        self.kan_weight = EfficientKANLayer(in_features=input_dim, out_features=self.embed_dim, grid_size=5, spline_order=3)

    def forward(self, nodes):
        # 1. Get aggregated neighbor features from our new B-Spline Aggregator
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        
        # 2. Combine with self features
        if not self.gcn:
            self_feats = self.features(torch.tensor(nodes))
            if self.cuda:
                self_feats = self_feats.cuda()
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
            
        # 3. B-SPLINE KAN PROJECTION (Replacing standard nn.Linear + ReLU)
        # We MUST apply tanh to keep features in the [-1, 1] grid support range
        normalized_combined = torch.tanh(combined)
        output = self.kan_weight(normalized_combined)
        
        return output