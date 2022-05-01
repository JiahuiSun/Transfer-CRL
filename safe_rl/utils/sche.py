import numpy as np


class ThresholdScheduler:
    def __init__(self, epoch_per_threshold=1):
        self.epoch_per_threshold = epoch_per_threshold
        self.threshold_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        # np.random.shuffle(self.threshold_list)
        self.t_idx = 0
        self.threshold = self.threshold_list[self.t_idx]
        self.test_threshold_list = [5, 20, 25, 30, 55, 0, 27.5, 32.5, 37.5, 100]
        self.sampled_threshold_list = np.random.uniform(0, 100, 10)
    
    def update(self, epoch):
        """循环训练weights, 为什么要循环训练？因为让所有weight共同长进，而不是一枝独秀
        """
        if epoch % self.epoch_per_threshold == 0:
            self.threshold = self.threshold_list[self.t_idx%len(self.threshold_list)]
            self.t_idx += 1
        return self.threshold


class WeightScheduler:
    def __init__(self, epoch_per_weight=1):
        self.epoch_per_weight = epoch_per_weight
        self.wr_list = [[1, 0], [0.5, 0.5], [0, 1]]
        self.wc_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.threshold_list = [10, 25, 50]
        self.idx = 0
        self.threshold = self.threshold_list[self.idx]
        self.wr = self.wr_list[self.idx]
        self.wc = self.wc_list[self.idx]
        self.r_dim, self.c_dim = len(self.wr), len(self.wc)
        
    def update(self, epoch):
        if epoch % self.epoch_per_weight == 0:
            self.threshold = self.threshold_list[self.idx%len(self.threshold_list)]
            self.idx += 1
        return self.wr, self.wc, self.threshold
