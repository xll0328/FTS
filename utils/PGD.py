import torch
import torch.nn as nn

class PGD(nn.Module):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, device='cpu', pred_len=48):
        super(PGD, self).__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.pred_len = pred_len
        self.device = device

    def forward(self, inputs, labels):
        inputs = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.L1Loss()
        adv_inputs = inputs.clone().detach()

        if self.random_start:
            adv_inputs = adv_inputs + torch.normal(mean=0.0, std=self.eps, size=adv_inputs.size()).to(self.device) * 0.2
            adv_inputs = adv_inputs.detach()

        for _ in range(self.steps):
            adv_inputs.requires_grad = True
            outputs, _, _ = self.model(adv_inputs)
            labels = labels[:, -self.pred_len:, -1:]
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_inputs, retain_graph=False, create_graph=False)[0]
            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            adv_inputs =(inputs + delta).detach()

        return adv_inputs