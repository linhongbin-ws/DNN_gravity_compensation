import torch

def Lagrange_Net(model, feature, delta_q, device='cpu'):
    target_hat = torch.zeros(feature.shape, dtype=torch.float32, device=device)
    base_hat = model(feature)
    base_hat.require_grad = False
    for j in range(feature.shape[1]):
        feature_d = feature.clone()
        feature_d[:, j] = feature_d[:, j].clone() + torch.ones(feature_d[:, j].shape, device=device).float() * delta_q
        target_hat[:, j] = model(feature_d).squeeze()-base_hat.squeeze()
    target_hat = target_hat / delta_q
    return target_hat

# create Net architecture
class LogNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LogNet, self).__init__()
        H1 = 100
        H2 = 50
        self._relu = torch.nn.ReLU()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._middle_linear = torch.nn.Linear(H, H)
        self._output_linear = torch.nn.Linear(H, D_out)
        self._epsilon = 1

    def forward(self, x):
        h = x
        h = self._input_linear(h).clamp(self._epsilon)
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

class SinNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SinNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._middle_linear = torch.nn.Linear(H, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = torch.sin(h)
        y_pred = self._output_linear(h)
        return y_pred


class SigmoidNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SigmoidNet, self).__init__()
        self._input_linear = torch.nn.Linear(D_in, H)
        self._output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self._input_linear(x)
        h = torch.sigmoid(h)
        y_pred = self._output_linear(h)
        return y_pred