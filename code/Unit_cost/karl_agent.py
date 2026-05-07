import torch
from torch import nn
import torch.optim as optim
import torch_sparse
import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import batch_graph
import graph
import replay_mem
import replay_mem_prioritized
import karl_env
import utils
import scipy.linalg as linalg
import scipy
import os
import pandas as pd
import os.path
from torch.autograd import Variable
from karl_net import KARL_net 
import os,sys
import math
os.chdir(sys.path[0])
from MRGNN.encoders import Encoder
# from MRGNN.mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention, KANformerMultiplexFusion
from MRGNN.mutil_layer_weight import KANformerMultiplexFusion
# from MRGNN.aggregators import MeanAggregator


GAMMA = 1  # decay rate of past observations
UPDATE_TIME = 1000
EMBEDDING_SIZE = 64
MAX_ITERATION = 100001
LEARNING_RATE = 0.0001   #
MEMORY_SIZE = 100000
Alpha = 0.001 ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
epsilon = 0.0000001  # small amount to avoid zero priority
alpha = 0.6  # [0~1] convert the importance of TD error to priority
beta = 0.4  # importance-sampling, from initial value increasing to 1
beta_increment_per_sampling = 0.001
TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
N_STEP = 5
NUM_MIN = 30
NUM_MAX = 50
REG_HIDDEN = 32
M = 4  # how many edges selected each time for BA model

BATCH_SIZE = 64
initialization_stddev = 0.01 
n_valid = 200 
n_train = 1000 
aux_dim = 4
num_env = 32
inf = 2147483647/2
max_bp_iter = 3
aggregatorID = 0 #0:sum; 1:mean; 2:GCN
embeddingMethod = 1   #0:structure2vec; 1:graphsage

class KARL:
    def __init__(self):
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'GMM'
        self.TrainSet = graph.GSet()
        self.TestSet = graph.GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.Utils()
        self.IsHuberloss = False
        if(self.IsHuberloss):
            self.loss = nn.HuberLoss(delta=1.0)
        else:
            self.loss = nn.MSELoss()

        self.IsDoubleDQN = False
        self.IsPrioritizedSampling = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)

        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = replay_mem_prioritized.Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = replay_mem.NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            self.env_list.append(karl_env.KARLEnv(NUM_MAX))
            self.g_list.append(graph.Graph())

        self.test_env = karl_env.KARLEnv(NUM_MAX)
        print("CUDA:", torch.cuda.is_available())
        torch.set_num_threads(16)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        layerNodeAttention_weight1 = KANformerMultiplexFusion(
            features_num=EMBEDDING_SIZE, 
            dropout=0.5, 
            alpha=0.5,
            metapath_number=2, 
            device=self.device,
            num_heads=4 # Tune the number of Transformer heads here
        ).to(self.device)

        self.karl_net = KARL_net(layerNodeAttention_weight1, device=self.device)
        self.karl_net_T = KARL_net(layerNodeAttention_weight1, device=self.device)
        self.karl_net.to(self.device)
        self.karl_net_T.to(self.device)
        self.karl_net_T.eval()
        base_params = []
        kan_params = []
        for name, param in self.karl_net.named_parameters():
            if 'kan_weight' in name or 'spline_weight' in name or 'ChebyKAN' in name:
                kan_params.append(param)
            else:
                base_params.append(param)
        self.optimizer = optim.Adam([
            {'params': base_params, 'lr': LEARNING_RATE},
            {'params': kan_params, 'lr': LEARNING_RATE * 10}
        ], weight_decay=1e-5) # Weight decay helps prevent B-spline overfitting


        pytorch_total_params = sum(p.numel() for p in self.karl_net.parameters())
        print("Total number of karl_net parameters: {}".format(pytorch_total_params))
        self.flag = 1
    def gen_graph(self,num_min,num_max):
        max_n = num_max
        min_n = num_min
        cur_n = np.random.randint(max_n - min_n + 1) + min_n
        g = graph.Graph(cur_n)
        return g

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        for i in tqdm(range(n_train)):
            g = self.gen_graph(num_min, num_max)
            if g.max_rank == 1:
                continue
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, g)
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, g)
    def PrepareValidData(self):
        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            self.InsertGraph(g, is_test=True)
    def Run_simulator(self, num_seq, eps, TrainSet, n_step):
        num_env = len(self.env_list)
        n = 0
        while n < num_seq:
            for i in range(num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.add_from_env(self.env_list[i], n_step)
                        #print ('add experience transition!')
                    g_sample = TrainSet.Sample()
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph
            if n >= num_seq:
                break
            Random = False
            if random.uniform(0, 1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list],[env.remove_edge for env in self.env_list])
            else:
                Random = True
            for i in range(num_env):
                if Random:
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = self.argMax(pred[i])
                self.env_list[i].step(a_t)


    def PlayGame(self,n_traj, eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)

    def SetupSparseT(self, sparse_dicts):
        for sparse_dict in sparse_dicts:
            sparse_dict['index'] = Variable(sparse_dict['index']).to(self.device)
            sparse_dict['value'] = Variable(sparse_dict['value']).to(self.device)
        return sparse_dicts

    def SetupTrain(self, idxes, g_list, covered, actions, target, remove_edges):
        self.m_y = target
        self.inputs['target'] = Variable(torch.tensor(self.m_y).type(torch.FloatTensor)).to(self.device)
        batch_graph1 = batch_graph.BatchGraph(aggregatorID)
        batch_graph1.SetupTrain(idxes, g_list, covered, actions, remove_edges)
        batch_graph1.idx_map_list = [it[0] for it in batch_graph1.idx_map_list]
        self.inputs['action_select'] = self.SetupSparseT(batch_graph1.act_select)
        self.inputs['rep_global'] = self.SetupSparseT(batch_graph1.rep_global)
        self.inputs['n2nsum_param'] = self.SetupSparseT(batch_graph1.n2nsum_param)
        self.inputs['laplacian_param'] = self.SetupSparseT(batch_graph1.laplacian_param)
        self.inputs['subgsum_param'] = self.SetupSparseT(batch_graph1.subgsum_param)
        self.inputs['node_input'] = None
        self.inputs['aux_input'] = Variable(torch.tensor(batch_graph1.aux_feat).type(torch.FloatTensor)).to(self.device)
        self.inputs['adj'] = batch_graph1.adj
        self.inputs['v_adj'] = batch_graph1.virtual_adj

    def temp_batch_graph(self,batch_graph):
        batch_graph.act_select = batch_graph.act_select[0]
        batch_graph.rep_global = batch_graph.rep_global[0]
        batch_graph.n2nsum_param = batch_graph.n2nsum_param[0]
        batch_graph.laplacian_param = batch_graph.laplacian_param[0]
        batch_graph.subgsum_param = batch_graph.subgsum_param[0]
        #batch_graph.subgraph_id_span = batch_graph.subgraph_id_span[0]
        batch_graph.avail_act_cnt = batch_graph.avail_act_cnt[0]
        batch_graph.graph = batch_graph.graph[0]
        return batch_graph

    def SetupPredAll(self, idxes, g_list, covered, remove_edges):
        batch_graph1 = batch_graph.BatchGraph(aggregatorID)
        batch_graph1.SetupPredAll(idxes, g_list, covered, remove_edges)
        batch_graph1.idx_map_list = [it[0] for it in batch_graph1.idx_map_list]
        self.inputs['rep_global'] = self.SetupSparseT(batch_graph1.rep_global)

        self.inputs['n2nsum_param'] = self.SetupSparseT(batch_graph1.n2nsum_param)

        self.inputs['subgsum_param'] = self.SetupSparseT(batch_graph1.subgsum_param)

        self.inputs['node_input'] = None
        self.inputs['aux_input'] = Variable(torch.tensor(batch_graph1.aux_feat).type(torch.FloatTensor)).to(self.device)
        self.inputs['adj'] = batch_graph1.adj
        self.inputs['v_adj'] = batch_graph1.virtual_adj
        return batch_graph1.idx_map_list

    def Predict(self,g_list,covered,remove_edges,isSnapSnot):
        n_graphs = len(g_list)
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)
            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered, remove_edges)
            if isSnapSnot:
                result = self.karl_net_T.test_forward(node_input=self.inputs['node_input'],\
                    subgsum_param=self.inputs['subgsum_param'], n2nsum_param=self.inputs['n2nsum_param'],\
                    rep_global=self.inputs['rep_global'], aux_input=self.inputs['aux_input'],adj=self.inputs['adj'],v_adj=self.inputs['v_adj'])
            else:
                result = self.karl_net.test_forward(node_input=self.inputs['node_input'],\
                    subgsum_param=self.inputs['subgsum_param'], n2nsum_param=self.inputs['n2nsum_param'],\
                    rep_global=self.inputs['rep_global'], aux_input=self.inputs['aux_input'],adj=self.inputs['adj'],v_adj=self.inputs['v_adj'])
            raw_output = result[:,0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = raw_output[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithCurrentQNet(self,g_list,covered,remove_edges):
        result = self.Predict(g_list,covered,remove_edges,False)
        return result

    def PredictWithSnapshot(self,g_list,covered,remove_edges):
        result = self.Predict(g_list,covered,remove_edges,True)
        return result
    #pass
    def TakeSnapShot(self):
        self.karl_net_T.load_state_dict(self.karl_net.state_dict())

    def Fit(self):
        sample = self.nStepReplayMem.sampling(BATCH_SIZE)
        ness = False
        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes, sample.list_s_primes_edges)

        list_target = np.zeros([BATCH_SIZE, 1])

        for i in range(BATCH_SIZE):
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:
                    q_rhs=GAMMA * list_pred[i]
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i])
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs
            # list_target.append(q_rhs)
        if self.IsPrioritizedSampling:
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target, sample.list_st_edges)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        '''
        loss = 0.0
        n_graphs = len(g_list)
        i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            my_dict = {}
            my_dict[self.action_select] = self.inputs['action_select']
            my_dict[self.rep_global] = self.inputs['rep_global']
            my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            my_dict[self.ISWeights] = np.mat(ISWeights).T
            my_dict[self.target] = self.inputs['target']

            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict=my_dict)
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        return loss / len(g_list)
        '''
        return None


    def fit(self,g_list,covered,actions,list_target,remove_edges):
        loss_values = 0.0
        n_graphs = len(g_list)
        for i in range(0,n_graphs,BATCH_SIZE):
            self.optimizer.zero_grad()

            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
            batch_idxes = np.int32(batch_idxes)
            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target, remove_edges)
            q_pred, cur_message_layer = self.karl_net.train_forward(node_input=self.inputs['node_input'],\
                subgsum_param=self.inputs['subgsum_param'], n2nsum_param=self.inputs['n2nsum_param'],\
                action_select=self.inputs['action_select'], aux_input=self.inputs['aux_input'],adj=self.inputs['adj'],v_adj=self.inputs['v_adj'])
            loss = self.calc_loss(q_pred, cur_message_layer)
            loss.backward()
            self.optimizer.step()
            loss_values += loss.item()*bsize
        return loss_values / len(g_list)

    def calc_loss(self, q_pred, cur_message_layer) :
        loss = torch.zeros(1,device=self.device)
        loss1 = torch.zeros(1,device=self.device)
        loss2 = torch.zeros(1,device=self.device)
        for i in range(2):
            temp = cur_message_layer[i]
            loss_recons = 2 * torch.trace(torch.matmul(torch.transpose(cur_message_layer[i],0,1),\
                torch_sparse.spmm(self.inputs['laplacian_param'][i]['index'], self.inputs['laplacian_param'][i]['value'],\
                self.inputs['laplacian_param'][i]['m'], self.inputs['laplacian_param'][i]['n'],\
                 cur_message_layer[i])))
            edge_num = torch.sum(self.inputs['n2nsum_param'][i]['value'])
            loss_recons = torch.divide(loss_recons, edge_num)                  
            loss2 = torch.add(loss2,loss_recons)
        loss1 = torch.add(loss1,self.loss(self.inputs['target'], q_pred))
        loss = torch.add(loss1, loss2, alpha = Alpha)
        return loss

    def Train(self, skip_saved_iter=False):
        self.PrepareValidData()   
        self.gen_new_graphs(NUM_MIN, NUM_MAX)   
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()
        eps_start = 1.0
        eps_end = 0.05
        eps_step = 10000.0
        loss = 0

        save_dir = './models/KARL_%s_%s_%s'%(self.g_type,NUM_MIN, NUM_MAX)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX)
        start_iter=0
        if(skip_saved_iter):
            if(os.path.isfile(VCFile)):
                f_read = open(VCFile)
                line_ctr = f_read.read().count("\n")
                f_read.close()
                start_iter = max(300 * (line_ctr-1), 0)
                start_model = '%s/karl_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, start_iter)
                print(f'Found VCFile {VCFile}, choose start model: {start_model}')
                if(os.path.isfile(VCFile)):
                    self.LoadModel(start_model)
                    print(f'skipping iterations that are already done, starting at iter {start_iter}..')                    
                    f_out = open(VCFile, 'a')
                else:
                    print('failed to load starting model, start iteration from 0..')
                    start_iter=0
                    f_out = open(VCFile, 'w')                
        else:
            f_out = open(VCFile, 'w')
        
        best_frac = inf
        for iter in range(MAX_ITERATION):
            start = time.perf_counter()
            if( (iter and iter % 5000 == 0) or (iter==start_iter)):
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            if iter % 10 == 0:
                self.PlayGame(10, eps)
            if iter % 10000 == 0:
                if(iter == 0 or iter == start_iter):
                    N_start = start
                else:
                    N_start = N_end
                frac = 0.0
                test_start = time.time()
                for idx in range(n_valid):
                    frac += self.Test(idx)
                if frac < best_frac :
                    best_frac = frac
                    self.SaveModel('%s/best_model.ckpt' % (save_dir))
                test_end = time.time()
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file
                f_out.flush()
                print('iter %d, eps %.4f, average size of vc:%.6f'%(iter, eps, frac/n_valid))
                print ('testing 200 graphs time: %.2fs'%(test_end-test_start))
                N_end = time.perf_counter()
                print ('500 iterations total time: %.2fs\n'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                if(skip_saved_iter and iter==start_iter):
                    pass
                else:
                    if iter % 10000 == 0:
                        self.SaveModel(model_path)
            if( (iter % UPDATE_TIME == 0) or (iter==start_iter)):
                self.TakeSnapShot()
            self.Fit()
        f_out.close()


    def findModel(self):
        VCFile = './models/ModelVC_%d_%d.csv'%(NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 500 * min_vc
        best_model = './models/nrange_%d_%d_iter_%d.ckpt' % (NUM_MIN, NUM_MAX, best_model_iter)
        return best_model


    def Evaluate(self, data_test, data_test_name,data_type, model_file=None):
        if model_file == None: 
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        n_test = 20
        result_list_score = []
        result_list_time = []
        cost_value_list = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            adj1 = np.load(f"../../data/synthetic/{data_type}/syn_%s/adj1_%s.npy"%(data_test_name,i))
            adj2 = np.load(f"../../data/synthetic/{data_type}/syn_%s/adj2_%s.npy"%(data_test_name,i))
            G1 = nx.from_numpy_array(adj1)
            G2 = nx.from_numpy_array(adj2)
            g = graph.Graph_test(G1,G2)
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol , cost_value = self.GetSol(i)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
            cost_value_list.append(cost_value)
        self.ClearTestGraphs()
        cost_mean = np.mean(cost_value_list)
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return score_mean, score_std, time_mean, time_std, cost_mean

    def read_multiplex(self,path, N):
        layers_matrix = []
        graphs = []
        _ii = []
        _jj = []
        _ww = []
        g = nx.Graph()
        for i in range(0, N):
            g.add_node(i)
        with open(path, "r") as lines:
            cur_id = 1
            for l in lines:
                elems = l.strip(" \n").split(" ")
                layer_id = int(elems[0])
                if cur_id != layer_id:
                    adj_matr = nx.adjacency_matrix(g)
                    layers_matrix.append(adj_matr)
                    graphs.append(g)
                    g = nx.Graph()

                    for i in range(0, N):
                        g.add_node(i)

                    cur_id = layer_id
                node_id_1 = int(elems[1]) - 1
                node_id_2 = int(elems[2]) - 1
                if node_id_1 == node_id_2:
                    continue
                g.add_edge(node_id_1, node_id_2)

        adj_matr = nx.adjacency_matrix(g)
        layers_matrix.append(adj_matr)
        graphs.append(g)
        return layers_matrix, graphs
    
    def adj_list_to_adj(self,adj_list):
        num_nodes = len(adj_list)
        adj = np.zeros((num_nodes, num_nodes))
        for i, neighbors in adj_list:
            for neighbor in neighbors:
                adj[i][neighbor] = 1  
        return adj

    def EvaluateRealData(self, model_file, data_test, save_dir, stepRatio,num_nodes,layers):  
        solution_time = 0.0
        test_name = data_test.split('/')[-1]
        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        result_file1 = '%s/%s_%s_%s%s.%s' %(save_dir_local, "Soluion",test_name.split('.')[0], layers[0], layers[1], 'txt')
        result_file2 = '%s/%s_%s_%s%s.%s' %(save_dir_local, "NormalizedLMCC", test_name.split('.')[0], layers[0], layers[1], 'txt')
        layers_matrix, graphs = self.read_multiplex(
        "../../data/real/%s"%(test_name),num_nodes)
        g = graph.Graph_test(graphs[layers[0]-1],graphs[layers[1]-1])
        Mcc_average = [0] * g.num_nodes
        result_list_score = []
        with open(result_file1, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            if stepRatio > 0:
                step = np.max([int(stepRatio*g.num_nodes),1]) #step size
            else:
                step = 1
            #step = g.num_nodes
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            average_n = 1
            for num in tqdm(range(average_n)):
                solution, score, MaxCCList = self.GetSolution(0,test_name,step)
                Mcc_average=[Mcc_average[i]+MaxCCList[i] for i in range(min(len(Mcc_average),len(MaxCCList)))]
                result_list_score.append(score)
            t2 = time.time()
            solution_time = (t2 - t1)
            score_mean = np.mean(result_list_score)
            print(score_mean)
            score_std = np.std(result_list_score)
            for i in range(len(solution)):
                f_out.write('%d\n' % solution[i])
        with open(result_file2, 'w') as f_out:
            for j in range(g.num_nodes):
                if j < len(Mcc_average):
                    f_out.write('%.8f\n' % (float(Mcc_average[j]/average_n)))
                else:
                    Mcc = 1 / g.max_rank
                    f_out.write('%.8f\n' % Mcc)
        nodes = list(range(g.num_nodes))
        remain_nodes = list(set(nodes)^set(solution))
        #score_total = score + (len(remain_nodes)-1) / (g.max_rank * g.num_nodes)
        with open(result_file2, 'a') as f_out:
            f_out.write('%.8f\n' % score_mean)
            f_out.write('%.8f\n' % score_std)
        self.ClearTestGraphs()
        return solution, solution_time, score
    
    # def EvaluateRealDataLookahead(self, checkpoints, data_test, save_dir, stepRatio, num_nodes, layers):
    #     """
    #     Ordinal Rank Consensus (Borda Count) for Res-GKAN:
    #     Aggregates policies by averaging ordinal ranks rather than raw Q-values,
    #     completely immunizing the ensemble against Chebyshev scale drift.
    #     """
    #     import time
    #     import copy
    #     import os
    #     import mvc_env
    #     from scipy.stats import rankdata # NEW IMPORT
        
    #     start_time = time.time()
    #     ensemble_nets = []
    #     print(f"Loading {len(checkpoints)} checkpoints for Rank-Based Consensus...")
    #     for ckpt in checkpoints:
    #         self.LoadModel(ckpt)
    #         net_copy = copy.deepcopy(self.MultiDismantler_net)
    #         net_copy.eval()
    #         ensemble_nets.append(net_copy)

    #     test_name = data_test.split('/')[-1]
    #     save_dir_local = f'{save_dir}/StepRatio_{stepRatio:.4f}_hybrid'
    #     if not os.path.exists(save_dir_local):
    #         os.makedirs(save_dir_local, exist_ok=True)
            
    #     result_file1 = '%s/%s_%s_%s%s.%s' % (save_dir_local, "Soluion", test_name.split('.')[0], layers[0], layers[1], 'txt')
    #     result_file2 = '%s/%s_%s_%s%s.%s' % (save_dir_local, "NormalizedLMCC", test_name.split('.')[0], layers[0], layers[1], 'txt')
        
    #     layers_matrix, graphs = self.read_multiplex("../../data/real/%s" % (test_name), num_nodes)
    #     g = graph.Graph_test(graphs[layers[0]-1], graphs[layers[1]-1])
        
    #     env = mvc_env.MvcEnv(self.test_env.norm)
    #     env.s0(g)
        
    #     sol = []
    #     f_sol = open(result_file1, 'w')
        
    #     with torch.no_grad():
    #         while not env.isTerminal():
    #             g_list = [env.graph]
    #             total_ranks = None
                
    #             for net in ensemble_nets:
    #                 self.MultiDismantler_net = net
    #                 list_pred = self.PredictWithCurrentQNet(g_list, [env.action_list], [env.remove_edge])
    #                 q_pred = list_pred[0] 
                    
    #                 # Convert uncalibrated Q-values to strict ordinal ranks
    #                 # Highest Q-value gets the highest rank number
    #                 ranks = rankdata(q_pred)
                    
    #                 if total_ranks is None:
    #                     total_ranks = ranks.copy()
    #                 else:
    #                     total_ranks += ranks
                
    #             # Pick the node with the highest aggregate Borda rank
    #             best_actual_node = self.argMax(total_ranks)
                
    #             env.stepWithoutReward(best_actual_node)
    #             sol.append(best_actual_node)
    #             f_sol.write('%d\n' % best_actual_node)
                
    #     f_sol.close()
        
    #     with open(result_file2, 'w') as f_out:
    #         for j in range(env.graph.num_nodes):
    #             if j < len(env.MaxCCList):
    #                 f_out.write('%.8f\n' % env.MaxCCList[j])
    #             else:
    #                 f_out.write('%.8f\n' % (1 / env.graph.max_rank))
            
    #         f_out.write('%.8f\n' % env.score)
    #         f_out.write('%.8f\n' % 0.0) 
        
    #     return sol, time.time() - start_time, env.score


    # def EvaluateRealDataPhysicalRollout(self, model_file, data_test, save_dir, stepRatio, num_nodes, layers, top_m=8):
    #     """
    #     Corrected KAN-Guided Physical Rollout (Single Model)
    #     KAN proposes Top-M candidates; physics simulates each removal and selects the true best.
    #     """
    #     import time
    #     import copy
    #     import os
    #     import numpy as np
    #     from scipy.integrate import simpson

    #     start_time = time.time()
        
    #     self.LoadModel(model_file)
    #     self.MultiDismantler_net.eval()
        
    #     test_name = data_test.split('/')[-1]
    #     save_dir_local = f'{save_dir}/StepRatio_{stepRatio:.4f}_physical_rollout'
    #     os.makedirs(save_dir_local, exist_ok=True)
        
    #     result_file1 = f'{save_dir_local}/Solution_{test_name.split(".")[0]}_{layers[0]}{layers[1]}.txt'
    #     result_file2 = f'{save_dir_local}/NormalizedLMCC_{test_name.split(".")[0]}_{layers[0]}{layers[1]}.txt'
        
    #     layers_matrix, graphs = self.read_multiplex(f"../../data/real/{test_name}", num_nodes)
    #     g = graph.Graph_test(graphs[layers[0]-1], graphs[layers[1]-1])
        
    #     env = mvc_env.MvcEnv(self.test_env.norm)
    #     env.s0(g)
        
    #     sol = []
    #     with open(result_file1, 'w') as f_sol:
    #         while not env.isTerminal():
    #             g_list = [env.graph]
                
    #             # KAN proposes Q-values for valid actions only
    #             list_pred = self.PredictWithCurrentQNet(g_list, [env.action_list], [env.remove_edge])
    #             q_pred = list_pred[0].flatten()
                
    #             # Top-M candidates (safe indexing)
    #             current_top_m = min(top_m, len(env.action_list))
    #             if current_top_m == 0:
    #                 break
    #             top_indices = np.argsort(-q_pred)[:current_top_m]
    #             top_candidates = [env.action_list[i] for i in top_indices]
                
    #             # Physical verification (lightweight)
    #             best_action = top_candidates[0]
    #             best_lmcc = float('inf')
                
    #             for candidate in top_candidates:
    #                 env_clone = copy.deepcopy(env)
    #                 env_clone.stepWithoutReward(candidate)
    #                 simulated_lmcc = env_clone.MaxCCList[-1] if env_clone.MaxCCList else 1.0
                    
    #                 if simulated_lmcc < best_lmcc:
    #                     best_lmcc = simulated_lmcc
    #                     best_action = candidate
                
    #             # Execute best action
    #             env.stepWithoutReward(best_action)
    #             sol.append(best_action)
    #             f_sol.write(f'{best_action}\n')
        
    #     # Consistent AUDC calculation
    #     padded = [env.MaxCCList[j] if j < len(env.MaxCCList) else 1.0 / env.graph.max_rank 
    #               for j in range(env.graph.num_nodes)]
    #     final_audc = simpson(np.array(padded), np.linspace(0, 1, len(padded)))
        
    #     with open(result_file2, 'w') as f_out:
    #         for val in padded:
    #             f_out.write(f'{val:.8f}\n')
    #         f_out.write(f'{final_audc:.8f}\n')
    #         f_out.write('0.00000000\n')
        
    #     return sol, time.time() - start_time, final_audc
    
    def GetSolution(self, gid, test_name, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        start = time.time()
        iter = 0
        sum_sort_time = 0

        while (not self.test_env.isTerminal()):              
            print ('Iteration:%d'%iter)
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list], [self.test_env.remove_edge])
            start_time = time.time()
            batchSol = np.argsort(-list_pred[0])[:step]

            end_time = time.time()
            sum_sort_time += (end_time-start_time)
            # for new_action in target_index_val:
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        return sol, self.test_env.score, self.test_env.MaxCCList

    def Test(self,gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid)) 
        g_list.append(self.test_env.graph)
        cost = 0.0
        sol = []
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list], [self.test_env.remove_edge])
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes))
        remian_nodes = list(set(nodes)^set(sol))
        return self.test_env.score + len(remian_nodes) / (self.test_env.graph.max_rank * self.test_env.graph.num_nodes)


    def GetSol(self, gid, step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g = self.test_env.graph
        g_list.append(g)
        cost = 0.0
        sol = []
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list], [self.test_env.remove_edge])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        cost_value = len(sol)/(g.num_nodes)
        nodes = list(range(g.num_nodes))
        remain_nodes = list(set(nodes)^set(sol))
        return self.test_env.score, sol, cost_value


    def SaveModel(self,model_path):
        torch.save(self.karl_net.state_dict(), model_path)
        print('model has been saved success!')

    def LoadModel(self,model_path):
        try:
            self.karl_net.load_state_dict(torch.load(model_path))
        except:
            self.karl_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        print('restore model from file successfully')

    def argMax(self, scores):
        n = len(scores)
        pos = -1
        best = -10000000
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        n = len(scores)
        pos = -1
        best = -10000000
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best


    def HXA(self, g, method):
        sol = []
        G = g.copy()
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':
                dc = nx.degree_centrality(G)
            elif method == 'HBA':
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA':
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':
                dc = nx.pagerank(G)
            keys = list(dc.keys())
            values = list(dc.values())
            maxTag = np.argmax(values)
            node = keys[maxTag]
            sol.append(int(node))
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        Robustness = self.utils.getRobustness(g, solutions)
        return Robustness, sol
    
def basic_ci(graph, node, degrees):
    ci = 0
    neighbors = list(graph.neighbors(node))
    node_degree = degrees[node]
    if node_degree == 0:
        ci = -1
    else :
        for neighbor in neighbors:
            ci += degrees[neighbor] - 1

        ci *= (node_degree - 1)
    return ci

def get_ci_dict(graph):
    degrees = dict(graph.degree()) 
    ci_values = {node: basic_ci(graph, node, degrees) for node in graph.nodes()}
    return ci_values