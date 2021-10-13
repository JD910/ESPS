import copy
import math
import torch
import torch.nn.functional

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()[0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()

class SiLU(torch.nn.Module):

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

class Conv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        s = self.stride
        d = self.dilation
        k = self.weight.shape[-2:]
        h, w = x.size()[-2:]
        pad_h = max((math.ceil(h / s[0]) - 1) * s[0] + (k[0] - 1) * d[0] + 1 - h, 0)
        pad_w = max((math.ceil(w / s[1]) - 1) * s[1] + (k[1] - 1) * d[1] + 1 - w, 0)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0)

        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)

class Conv(torch.nn.Module):
    def __init__(self, args, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        if args.tf:
            self.conv = Conv2d(in_ch, out_ch, k, s, 1, g, bias=False)
        else:
            self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, k // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.silu = activation

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))

class SE(torch.nn.Module):

    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      torch.nn.SiLU(),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))

class Residual(torch.nn.Module):

    def __init__(self, args, in_ch, out_ch, s, r, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        if fused:
            if args.tf and r == 1:
                features = [Conv(args, in_ch, r * in_ch, torch.nn.SiLU(), 3, s)]
            else:
                features = [Conv(args, in_ch, r * in_ch, torch.nn.SiLU(), 3, s),
                            Conv(args, r * in_ch, out_ch, identity)]
        else:
            if r == 1:
                features = [Conv(args, r * in_ch, r * in_ch, torch.nn.SiLU(), 3, s, r * in_ch),
                            SE(r * in_ch, r),
                            Conv(args, r * in_ch, out_ch, identity)]
            else:
                features = [Conv(args, in_ch, r * in_ch, torch.nn.SiLU()),
                            Conv(args, r * in_ch, r * in_ch, torch.nn.SiLU(), 3, s, r * in_ch),
                            SE(r * in_ch, r),
                            Conv(args, r * in_ch, out_ch, identity)]
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)

class EfficientNet2(torch.nn.Module):
    def __init__(self, args, num_class=2) -> None:
        super().__init__()
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 272, 1792]
        feature = [Conv(args, 3, filters[0], torch.nn.SiLU(), 3, 2)]
        if args.tf:
            filters[5] = 256
            filters[6] = 1280
        for i in range(2):
            if i == 0:
                feature.append(Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                feature.append(Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(Residual(args, filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                feature.append(Residual(args, filters[1], filters[1], 1, 4, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(Residual(args, filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                feature.append(Residual(args, filters[2], filters[2], 1, 4, gate_fn[0]))

        for i in range(6):
            if i == 0:
                feature.append(Residual(args, filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                feature.append(Residual(args, filters[3], filters[3], 1, 4, gate_fn[1]))

        for i in range(9):
            if i == 0:
                feature.append(Residual(args, filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                feature.append(Residual(args, filters[4], filters[4], 1, 6, gate_fn[1]))

        for i in range(15):
            if i == 0:
                feature.append(Residual(args, filters[4], filters[5], 2, 6, gate_fn[1]))
            else:
                feature.append(Residual(args, filters[5], filters[5], 1, 6, gate_fn[1]))
        feature.append(Conv(args, filters[5], filters[6], torch.nn.SiLU()))

        self.feature = torch.nn.Sequential(*feature)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.5, True),
                                      torch.nn.Linear(filters[6], num_class))
        
        #self.softmax = torch.nn.Softmax()
                                      
        torch.nn.Linear(filters[6], num_class)

        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x)
        return self.fc(x.mean((2, 3)))
        #return x

    def export(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'silu'):
                if isinstance(m.silu, torch.nn.SiLU):
                    m.silu = SiLU()
            if type(m) is SE:
                if isinstance(m.se[1], torch.nn.SiLU):
                    m.se[1] = SiLU()
        return self

class EMA(torch.nn.Module):

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.module.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(self.decay * e + (1. - self.decay) * m)

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        prob = self.softmax(x)
        loss = -prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        return ((1. - self.epsilon) * loss + self.epsilon * (-prob.mean(dim=-1))).mean()

class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        for param_group in self.optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        self.values = [param_group['initial_lr'] for param_group in self.optimizer.param_groups]
        self.update(self.values)

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3
        self.warmup_lr_init = 1e-6

        self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.values]
        self.update(self.warmup_lr_init)

    def __str__(self) -> str:
        return 'step'

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.values]
        if values is not None:
            self.update(values)

    def update(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value

class RMSprop(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0,
                 momentum=0., centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss
