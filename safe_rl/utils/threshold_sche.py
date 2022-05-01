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
