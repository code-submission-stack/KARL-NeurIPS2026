from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
import numpy as np
from kan_layers import EfficientKANLayer
# from MRGNN.encoders import Encoder
# from MRGNN.aggregators import MeanAggregator
# from MRGNN.utils import LogisticRegression
# from MRGNN.mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention
# from MRGNN.mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention, KANformerMultiplexFusion
import sys

def _to_float_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(np.array(x), dtype=torch.float).to(device)

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):

        x = torch.tanh(x) 
        cheby_polys = [torch.ones_like(x), x]
        for i in range(2, self.degree + 1):
            cheby_polys.append(2 * x * cheby_polys[-1] - cheby_polys[-2])
        poly_stack = torch.stack(cheby_polys, dim=-1)
        y = torch.einsum('bid,iod->bo', poly_stack, self.cheby_coeffs)
        
        return y + self.bias


class KARL_net(nn.Module):
    def __init__(self, layerNodeAttention_weight,
                 embedding_size=64, w_initialization_std=1, reg_hidden=32, max_bp_iter=3,
                 embeddingMethod=1, aux_dim=4, device=None, node_attr=False):
        super(KARL_net, self).__init__()

        self.layerNodeAttention_weight = layerNodeAttention_weight
        self.rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size), 2)
        self.embedding_size = embedding_size
        self.w_initialization_std = w_initialization_std
        self.reg_hidden = reg_hidden
        self.max_bp_iter = max_bp_iter
        self.embeddingMethod = embeddingMethod
        self.aux_dim = aux_dim
        self.device = device
        self.node_attr = node_attr
        self.act = nn.ReLU()
        self.w_n2l = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, size=(2, self.embedding_size)))
        self.p_node_conv_kan = EfficientKANLayer(in_features=self.embedding_size, out_features=self.embedding_size, grid_size=5, spline_order=3)
        self.p_node_conv2_kan = EfficientKANLayer(in_features=self.embedding_size, out_features=self.embedding_size, grid_size=5, spline_order=3)
        self.p_node_conv3_kan = EfficientKANLayer(in_features=2 * self.embedding_size, out_features=self.embedding_size, grid_size=5, spline_order=3)
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))

        if self.reg_hidden > 0:
            self.kan_layer1 = ChebyKANLayer(self.embedding_size, self.reg_hidden, degree=4)
            self.kan_layer2 = ChebyKANLayer(self.reg_hidden + self.aux_dim, 1, degree=4)
        else:
            self.kan_layer1 = None 
            self.kan_layer2 = ChebyKANLayer(self.embedding_size + self.aux_dim, 1, degree=4)

        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.embedding_size, 1)))
        self.w_layer1 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(embedding_size, 128)))
        self.w_layer2 = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(128, 1)))
        self.flag = 0

    def train_forward(self, node_input, subgsum_param, n2nsum_param, action_select, aux_input, adj, v_adj):
        nodes_cnt = n2nsum_param[0]['m']
        node_input = torch.zeros((2, nodes_cnt, 2)).to(self.device)                       
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((2, y_nodes_size, 2)).to(self.device)
        adj   = _to_float_tensor(adj,   self.device)
        v_adj = _to_float_tensor(v_adj, self.device)
        node_embedding = []
        lay_num = 2
        for l in range(lay_num):
            for i in range(y_nodes_size):
                node_in_graph = torch.where(v_adj[l][i] == 1)
                if node_in_graph[0].numel() == 0:
                    continue
                degree = torch.sum(adj[l][node_in_graph], axis=1, keepdims=True)
                degree_max,_ = torch.max(degree,dim=0)
                degree_new = degree/degree_max
                node_feature = torch.cat((degree_new,degree_new),axis = 1)
                node_input[l][node_in_graph] = node_feature
        for l in range(lay_num):
            input_message = torch.matmul(node_input[l], self.w_n2l)
            input_potential_layer = self.act(input_message)
            y_input_message = torch.matmul(y_node_input[l], self.w_n2l)
            y_input_potential_layer = self.act(y_input_message)
            lv = 0
            cur_message_layer = input_potential_layer
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
            y_cur_message_layer = y_input_potential_layer
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            while lv < self.max_bp_iter:
                lv =lv + 1
                n2npool = torch_sparse.spmm(n2nsum_param[l]['index'], n2nsum_param[l]['value'],
                        n2nsum_param[l]['m'], n2nsum_param[l]['n'], cur_message_layer)
                node_linear = self.p_node_conv_kan(torch.tanh(n2npool))
                y_n2npool = torch_sparse.spmm(subgsum_param[l]['index'], subgsum_param[l]['value'],
                        subgsum_param[l]['m'], subgsum_param[l]['n'], cur_message_layer)
                y_node_linear = self.p_node_conv_kan(torch.tanh(y_n2npool))
                cur_message_layer_linear = self.p_node_conv2_kan(torch.tanh(cur_message_layer))
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                kan_out = self.p_node_conv3_kan(torch.tanh(merged_linear))
                residual = self.residual_mlp(cur_message_layer) * self.residual_alpha
                cur_message_layer = kan_out + residual
                cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
                y_cur_message_layer_linear = self.p_node_conv2_kan(torch.tanh(y_cur_message_layer))
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_kan_out = self.p_node_conv3_kan(torch.tanh(y_merged_linear))
                y_residual = self.residual_mlp(y_cur_message_layer) * self.residual_alpha
                y_cur_message_layer = y_kan_out + y_residual
                y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)


            node_output = torch.cat((cur_message_layer,y_cur_message_layer),axis = 0)
            #node_output = torch.nn.functional.normalize(node_output, p=2, dim=1)
            node_embedding.append(node_output)    
                    
        node_embedding_0 = node_embedding[0]
        node_embedding_1 = node_embedding[1]
        if self.embeddingMethod == 1:  # MRGNN
            nodes = np.array(list(range(nodes_cnt + y_nodes_size)))
            embeds = [node_embedding_0,node_embedding_1]
            message_layer = torch.zeros(lay_num, nodes.size, self.embedding_size, device=self.device)
            for l in range(lay_num):
                result_temp = self.layerNodeAttention_weight(embeds,nodes,l)
                message_layer[l] = result_temp
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)
        q = 0
        q_list = []
        w_layer = []
        for l in range(lay_num):
            y_potential = y_cur_message_layer[l]
            action_embed = torch_sparse.spmm(action_select[l]['index'], action_select[l]['value'], \
                                             action_select[l]['m'], action_select[l]['n'],
                                             cur_message_layer[l])
            temp = torch.matmul(torch.unsqueeze(action_embed, dim=2), torch.unsqueeze(y_potential, dim=1))
            Shape = action_embed.size()
            embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product, [Shape[0], 1]), \
                                                                       [Shape[0], Shape[1], 1])), Shape)
            if self.reg_hidden > 0:
                hidden = self.kan_layer1(embed_s_a) 
                last_output = torch.concat([hidden, aux_input[:,l,:]], 1)
                q_pred = self.kan_layer2(last_output)
            else:
                last_output = torch.concat([embed_s_a, aux_input[:,l,:]], 1)
                q_pred = self.kan_layer2(last_output)
            w_layer.append((self.act(y_potential @ self.w_layer1))@self.w_layer2)
            q_list.append(q_pred)
        w_layer = torch.concat(w_layer,dim = 1)
        w_layer_softmax = F.softmax(w_layer,dim = 1)
        q = w_layer_softmax[:,0].unsqueeze(1) * q_list[0] + w_layer_softmax[:,1].unsqueeze(1) * q_list[1]
        return q, cur_message_layer

    def test_forward(self, node_input, subgsum_param, n2nsum_param, rep_global, aux_input, adj, v_adj):
        nodes_cnt = n2nsum_param[0]['m']
        node_input = torch.zeros((2, nodes_cnt, 2), dtype=torch.float).to(self.device)                            
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((2, y_nodes_size, 2), dtype=torch.float).to(self.device)
        adj = torch.tensor(np.array(adj),dtype=torch.float).to(self.device)
        v_adj = torch.tensor(np.array(v_adj),dtype=torch.float).to(self.device)
        node_embedding = []
        lay_num = 2
        for l in range(lay_num):
            for i in range(y_nodes_size):
                node_in_graph = torch.where(v_adj[l][i] == 1)
                if node_in_graph[0].numel() == 0:
                    continue
                degree = torch.sum(adj[l][node_in_graph], axis=1, keepdims=True)
                degree_max,_ = torch.max(degree,dim=0)
                degree_new = degree/degree_max
                node_feature = torch.cat((degree_new,degree_new),axis = 1)
                node_input[l][node_in_graph] = node_feature

        for l in range(lay_num):
            input_message = torch.matmul(node_input[l], self.w_n2l)
            input_potential_layer = self.act(input_message)
            y_input_message = torch.matmul(y_node_input[l], self.w_n2l)
            y_input_potential_layer = self.act(y_input_message)
            lv = 0
            cur_message_layer = input_potential_layer
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
            y_cur_message_layer = y_input_potential_layer
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
            while lv < self.max_bp_iter:
                lv =lv + 1
                n2npool = torch_sparse.spmm(n2nsum_param[l]['index'], n2nsum_param[l]['value'],\
                        n2nsum_param[l]['m'], n2nsum_param[l]['n'], cur_message_layer)
    
                node_linear = self.p_node_conv_kan(torch.tanh(n2npool))
                y_n2npool = torch_sparse.spmm(subgsum_param[l]['index'], subgsum_param[l]['value'],\
                        subgsum_param[l]['m'], subgsum_param[l]['n'], cur_message_layer)
                y_node_linear = self.p_node_conv_kan(torch.tanh(y_n2npool))             
                cur_message_layer_linear = self.p_node_conv2_kan(torch.tanh(cur_message_layer))
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1)
                kan_out = self.p_node_conv3_kan(torch.tanh(merged_linear))
                residual = self.residual_mlp(cur_message_layer) * self.residual_alpha
                cur_message_layer = kan_out + residual
                cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)
                y_cur_message_layer_linear = self.p_node_conv2_kan(torch.tanh(y_cur_message_layer))
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1)
                y_kan_out = self.p_node_conv3_kan(torch.tanh(y_merged_linear))
                y_residual = self.residual_mlp(y_cur_message_layer) * self.residual_alpha
                y_cur_message_layer = y_kan_out + y_residual
                y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)

            node_output = torch.cat((cur_message_layer,y_cur_message_layer),axis = 0)
            node_embedding.append(node_output)    
                    
        node_embedding_0 = node_embedding[0]
        node_embedding_1 = node_embedding[1]
        if self.embeddingMethod == 1:  # MRGNN
            nodes = np.array(list(range(nodes_cnt + y_nodes_size)))
            embeds = [node_embedding_0,node_embedding_1]
            message_layer = torch.zeros(lay_num, nodes.size, self.embedding_size, device=self.device)
            for l in range(lay_num):
                result_temp = self.layerNodeAttention_weight(embeds,nodes,l)
                message_layer[l] = result_temp
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)
        q = 0
        q_list = []
        w_layer = []
        for l in range(lay_num):
            y_potential = y_cur_message_layer[l]
            rep_y = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'].to(self.device), \
                                      rep_global[l]['m'], rep_global[l]['n'], y_potential.to(self.device))
            temp1 = torch.matmul(torch.unsqueeze(cur_message_layer[l], dim=2),
                                 torch.unsqueeze(rep_y, dim=1))
            Shape1 = cur_message_layer[l].size()
            embed_s_a_all = torch.reshape(torch.matmul(temp1,
                                                       torch.reshape(torch.tile(self.cross_product, [Shape1[0], 1]),
                                                                     [Shape1[0], Shape1[1], 1])), Shape1)
            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                hidden = self.kan_layer1(embed_s_a_all)
                rep_aux = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'], rep_global[l]['m'], rep_global[l]['n'], aux_input[:,l,:])
                last_output = torch.concat([hidden, rep_aux], 1)
                q_on_all = self.kan_layer2(last_output)
            else:
                rep_aux = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'], rep_global[l]['m'], rep_global[l]['n'], aux_input[:,l,:])
                last_output = torch.concat([embed_s_a_all, rep_aux], 1)
                q_on_all = self.kan_layer2(last_output)


            w_layer.append((self.act(rep_y @ self.w_layer1))@self.w_layer2)
            q_list.append(q_on_all)
        w_layer = torch.concat(w_layer,dim = 1)
        w_layer_softmax = F.softmax(w_layer,dim = 1)
        q = w_layer_softmax[:,0].unsqueeze(1) * q_list[0] + w_layer_softmax[:,1].unsqueeze(1) * q_list[1]
        return q