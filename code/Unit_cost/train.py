from karl_agent import KARL
import os,sys
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
os.chdir(sys.path[0])

def main():
    dqn = KARL()
    dqn.Train()


if __name__=="__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    main()
