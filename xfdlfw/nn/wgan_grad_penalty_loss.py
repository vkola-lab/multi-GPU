"""
Created on Thu Aug 26 15:42:38 2021

@author: cxue2
"""

import torch


class WGANGradPenaltyLoss():

    def __init__(self, discriminator, lambda_=10):
        
        self.discriminator = discriminator
        self.lambda_ = lambda_

    def __call__(self, x, y):

        # parse inputs
        x_fake = x[y == 0]
        x_real = x[y == 1]
        device = x.device

        # batch size
        assert x_fake.shape[0] == x_real.shape[0]
        B = x_real.shape[0]

        # eps ~ U[0, 1]
        eps = torch.rand([B] + [1] * (len(x_real.shape) - 1)) # [B, 1, 1, ...]
        eps = eps.expand_as(x_real)
        eps = eps.to(device)

        # calculate interpolations
        x_intp = eps * x_real.detach() + (1 - eps) * x_fake.detach()
        x_intp.requires_grad = True

        # feed to the discriminator
        output = self.discriminator(x_intp)

        # calculate gradients of the discriminator outputs (probabilities) w.r.t. x_intp
        grads = torch.autograd.grad(
            outputs = output,
            inputs = x_intp,
            grad_outputs = torch.ones_like(output),
            create_graph = True,
            retain_graph = True,
        )[0]

        # calculate and return loss
        grads = grads.view(B, -1)
        loss = self.lambda_ * ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return loss


if __name__ == '__main__':

    net = torch.nn.Sequential(
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    )
    loss = WGANGradPenaltyLoss(net)

    x = torch.randn(8, 5)
    y = torch.ones(8)
    y[:4] = 0

    print(loss(x, y))

    # print(x, y, loss(x, y))



