import torch

def exp_loss(M, G):
    return torch.sum(M * G)


def space_loss2(constC, hC1, hC2, G):
    tens = tensor_product(constC, hC1, hC2, G)
    return torch.sum(torch.abs(tens * G))

def tensor_product(constC, hC1, hC2, G):
    A = - torch.matmul(
        torch.matmul(hC1, G), hC2.T
    )
    tens = constC + A
 
    return tens
        


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

def calculate_kl_divergence(d, d_pred):
    d_tensor = torch.tensor(d, dtype=torch.float32)
    d_pred_tensor = torch.tensor(d_pred, dtype=torch.float32)

    p = d_tensor / d_tensor.sum()
    q = d_pred_tensor / d_pred_tensor.sum()

    kl_divergence = torch.sum(p * torch.log(p / q + 1e-6))

    return kl_divergence