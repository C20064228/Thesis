import random
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_seed(seed):
    plt.rcParams['font.size'] = 14
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True