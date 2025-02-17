
import torch
import torch.nn as nn

def static_loss(model, f, f_bc, x, y):
    y_out = model.forward(f, f_bc, x)
    loss = ((y_out - y)**2).mean()
    return loss

def static_forward(model, f, f_bc, x):
    """
    f: M*dim_f
    x: N*dim_x
    y_br: M*dim_h
    y_tr: N*dim_h
    y_out: y_br(ij) tensorprod y_tr(kj)
    """
    y_br1 = model._branch1(f)
    y_br2 = model._branch2(f_bc)
    y_br = y_br1*y_br2

    y_tr = model._trunk(x)
    y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
    # print(y_out.shape)
    return y_out


def static_init(model, branch1_dim, branch2_dim, trunk_dim):
    # self.branch_dim = branch_dim
    # self.trunk_dim = trunk_dim
    model.z_dim = trunk_dim[-1]

    ## build branch net
    modules = []
    for i, h_dim in enumerate(branch1_dim):
        if i == 0:
            in_channels = h_dim
        else:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.Tanh()
                )
            )
            in_channels = h_dim
    model._branch1 = nn.Sequential(*modules)

    modules = []
    for i, h_dim in enumerate(branch2_dim):
        if i == 0:
            in_channels = h_dim
        else:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.Tanh()
                )
            )
            in_channels = h_dim
    model._branch2 = nn.Sequential(*modules)

    ## build trunk net
    modules = []
    for i, h_dim in enumerate(trunk_dim):
        if i == 0:
            in_channels = h_dim
        else:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.Tanh()
                )
            )
            in_channels = h_dim
    model._trunk = nn.Sequential(*modules)
    # self._out_layer = nn.Linear(self.z_dim, 1, bias = True)


class opnn(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim):
        super(opnn, self).__init__()
        static_init(self,branch1_dim, branch2_dim, trunk_dim)

    def forward(self, f, f_bc, x):
       return static_forward(self,f, f_bc, x)
    
    def loss(self, f, f_bc, x, y):
        return static_loss(self, f, f_bc, x, y)