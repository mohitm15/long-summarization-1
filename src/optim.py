import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Optimizer(object):
    def __init__(self, method, lr, acc=None, max_grad_norm=None):
        self.method = method
        self.lr = lr
        self.acc = acc
        self.max_grad_norm = max_grad_norm
        self.optim = None

    def set_parameters(self, params):
        self.params = [param for param in list(params) if param.requires_grad]
        if self.method == 'sgd':
            self.optim = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optim = optim.Adagrad(self.params, lr=self.lr, initial_accumulator_value=self.acc)
        elif self.method == 'adam':
            self.optim = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError(f"Unsupported optimizer method: {self.method}")

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optim.step()
