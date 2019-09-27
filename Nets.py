import torch

# create Net architecture
class LogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LogNet, self).__init__()
        H1 = 100
        H2 = 50
        self._relu = torch.nn.ReLU()
        self._input_linear = torch.nn.Linear(D_in, H1)
        self._middle_linear = torch.nn.Linear(H1, H)
        self._output_linear = torch.nn.Linear(H, D_out)
        self._epsilon = 1

    def forward(self, x):
        h = self._input_linear(x).clamp(min=self._epsilon)
        h = torch.log(h)
        h = self._middle_linear(h)
        h = torch.exp(h)
        y_pred = self._output_linear(h)
        return y_pred

# create Net architecture
class BPNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(BPNet, self).__init__()
        self._tanh = torch.nn.Tanh()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = self._tanh(h)
        y_pred = self._output_linear(h)
        return y_pred

class ReLuNet(torch.nn.Module):
    def __init__(self, D_in, H_list, D_out):
        super(ReLuNet, self).__init__()
        self.relu = torch.nn.ReLU()
        H_list = [D_in] + H_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(H_list[i], H_list[i+1]) for i in range(len(H_list)-1)])
        self.output_layer = torch.nn.Linear(H_list[-1], D_out)
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
        x = self.output_layer(x)
        return x
