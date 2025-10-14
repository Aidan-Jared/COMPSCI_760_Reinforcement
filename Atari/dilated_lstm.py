import torch
import torch.nn as nn

class DilatedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, radius=10):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.index = torch.arange(0, radius * hidden_size, radius)
        self.dilation = 0

    def forward(self, state, hidden):
        d_idx = self.dilation_idx
        hx, cx = hidden
        hx[:,d_idx], cx[:,d_idx] = self.rnn(state, (hx[:,d_idx], cx[:,d_idx]))
        detatached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detatached_hx = detatached_hx.view(detatached_hx.shape[0], self.hidden_size, self.radius - 1)
        detatached_hx = detatached_hx.sum(-1)
        y = (hx[:,d_idx] + detatached_hx) / self.radius
        return y , (hx,cx)
    
    def masked_idx(self, dilated_idx):
        all_indices = torch.arange(self.radius * self.hidden_size)
        mask = torch.ones(self.radius * self.hidden_size, dtype=torch.bool)
        mask[dilated_idx] = False
        masked_idx = all_indices[mask]
        return masked_idx
    
    @property
    def dilation_idx(self):
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.radius
        return dilation_idx