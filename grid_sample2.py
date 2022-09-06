from importlib.metadata import requires
import sys
# from typing_extensions import Required
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        ix_nw = IW-1 - (IW-1-ix_nw.abs()).abs()
        iy_nw = IH-1 - (IH-1-iy_nw.abs()).abs()

        ix_ne = IW-1 - (IW-1-ix_ne.abs()).abs()
        iy_ne = IH-1 - (IH-1-iy_ne.abs()).abs()

        ix_sw = IW-1 - (IW-1-ix_sw.abs()).abs()
        iy_sw = IH-1 - (IH-1-iy_sw.abs()).abs()

        ix_se = IW-1 - (IW-1-ix_se.abs()).abs()
        iy_se = IH-1 - (IH-1-iy_se.abs()).abs()

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    # image = torch.rand(1, 1, 10, 10, requires_grad=True)
    
    # def _fn(x, w):
    #     theta = w @ x.view(-1,1)
    #     theta = theta.view(-1, 2, 3)
    #     grid = F.affine_grid(theta, x.size(), align_corners=True)
    #     x = F.grid_sample(x, grid, align_corners=True)
    #     return x

    # W = 10 * torch.rand(6, 100, requires_grad=True)
    # # for i in range(10 * 10):
    # fn = lambda w: _fn(image, w).sum() #.reshape(1, -1)[:,i]
    # print(torch.autograd.gradcheck(fn, W))
    # sys.exit(0)
    
    for seed in range(20):
        i = 256
        gscale = 1.0
        torch.manual_seed(seed)
        image = torch.randn(1, 3, i, i)
        model = nn.Sequential(*[nn.Linear(int(gscale * i) * int(gscale * i) * 3, 128), nn.ReLU(), 
                                nn.Linear(128, 128), nn.ReLU(), 
                                # nn.Linear(128, 128), nn.ReLU(), 
                                # nn.Linear(128, 128), nn.ReLU(), 
                                nn.Linear(128, 1000)])
        
        model = model.double()
        image = image.double()

        for mode in ['bilinear']:#, 'bicubic']:
            # scale = torch.zeros(1, requires_grad=True).double()
            
            # def fn(scale):
            #     shift = torch.from_numpy(np.array([[0, 1]])).double()[...,None]
            #     # torch.ones(1, 2, 1)
            #     shift /= (shift**2).sum(dim=1,keepdims=True).sqrt()
            #     shift = scale * shift
            #     theta = torch.cat([torch.eye(2)[None], shift], dim=-1)
            #     grid = F.affine_grid(theta, (1, 1, int(gscale * i), int(gscale * i)))
            #     # print(grid)
            #     # print(grid.shape)
            #     align_corners = False if mode == 'bicubic' else True
            #     x = grid_sample(image, grid)#, mode=mode, align_corners=align_corners)
            #     # print(image - x)
            #     return model(x.view(1,-1)).sum()

            theta = torch.randn(1, 2, 3, requires_grad=True).double()

            def fn(theta):
                grid = F.affine_grid(theta, (1, 1, int(gscale * i), int(gscale * i)))
                # print(grid)
                # print(grid.shape)
                align_corners = False if mode == 'bicubic' else True
                x = grid_sample(image, grid)#, mode=mode, align_corners=align_corners)
                # print(image - x)
                return model(x.view(1,-1)).sum()

            center = 0
            epsilon = 1e-6
            # fd = (fn(center + epsilon) - fn(center - epsilon))/(2*epsilon)
            # print(f"fd: {fd}")
            try:
                # print(torch.autograd.gradcheck(fn, scale))
                # print(torch.autograd.grad(fn(scale), scale)[0])
                print(torch.autograd.gradcheck(fn, theta))
                print(torch.autograd.grad(fn(theta), theta)[0])         
            except Exception as e:
                print(e)

        print("*******************************************")

    # jac2 = torch.autograd.functional.jacobian(func, grid)

    # print((output1-output2).abs().max())
    # print((jac1-jac2).abs().max())
