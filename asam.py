import torch
from collections import defaultdict


class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()


import torch
from collections import defaultdict


class NRASAM(ASAM):
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        super().__init__(optimizer, model, rho, eta)
        self.accum_grad = []  # 历史梯度累积 Σ∇L
        self.prev_adv_grad = []  # 上一步对抗点梯度 ∇L(w^{adv}_{t-1})
        self.current_epoch = 0
        # 初始化历史梯度存储
        for p in model.parameters():
            if p.requires_grad:
                self.accum_grad.append(torch.zeros_like(p.data))
                self.prev_adv_grad.append(torch.zeros_like(p.data))
            else:
                self.accum_grad.append(None)
                self.prev_adv_grad.append(None)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        # 1. 计算并保存当前对抗点梯度 (θ_k^adv)
        adv_grads = []
        for p in self.model.parameters():
            adv_grads.append(p.grad.clone() if p.grad is not None else None)

        # 2. 更新历史梯度累积 (Σ_{i=0}^k ∇L(θ_i^adv))
        for i, grad in enumerate(adv_grads):
            if grad is not None:
                self.accum_grad[i] = self.accum_grad[i] + grad  # 公式(7)累加项

        # 3. 恢复原始参数 (θ_k)
        for n, p in self.model.named_parameters():
            if p.grad is None: continue
            p.sub_(self.state[p]["eps"])

        if self.current_epoch < 40:
            multiplier = 0.0000001
        elif self.current_epoch < 100:
            multiplier = 0.0000001
        elif self.current_epoch < 160:
            multiplier = 0.0000001
        else:  # 160及以上epoch
            multiplier = 0.0000001

        # 4. 执行NRSAM更新 (θ_{k+1} = θ_k - λ·prev_adv_grad + μ·accum_grad)
        for i, p in enumerate(self.model.parameters()):
            if adv_grads[i] is None: continue

            current_lr = self.optimizer.param_groups[0]['lr']
            lam = current_lr
            mu = multiplier * lam
            # 文献公式(7): -λ∇L(θ_{k-1}^adv) + μΣ∇L(θ_i^adv)
            update = -lam * self.prev_adv_grad[i] +mu * self.accum_grad[i]
            p.add_(update)  # 注：add_等效于 θ = θ + update

        # 5. 保存当前对抗梯度 (供下一时间步使用)
        self.prev_adv_grad = adv_grads



class NRSAM(SAM):
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        super().__init__(optimizer, model, rho, eta)
        self.accum_grad = []  # 历史梯度累积 Σ∇L
        self.prev_adv_grad = []  # 上一步对抗点梯度 ∇L(w^{adv}_{t-1})

        # 初始化历史梯度存储
        for p in model.parameters():
            if p.requires_grad:
                self.accum_grad.append(torch.zeros_like(p.data))
                self.prev_adv_grad.append(torch.zeros_like(p.data))
            else:
                self.accum_grad.append(None)
                self.prev_adv_grad.append(None)
    @torch.no_grad()
    def ascent_step(self):
        grads = []

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
    @torch.no_grad()
    def descent_step(self):
        # 1. 计算并保存当前对抗点梯度 (θ_k^adv)
        adv_grads = []
        for p in self.model.parameters():
            adv_grads.append(p.grad.clone() if p.grad is not None else None)

        # 2. 更新历史梯度累积 (Σ_{i=0}^k ∇L(θ_i^adv))
        for i, grad in enumerate(adv_grads):
            if grad is not None:
                self.accum_grad[i] = self.accum_grad[i] + grad  # 公式(7)累加项

        # 3. 恢复原始参数 (θ_k)
        for n, p in self.model.named_parameters():
            if p.grad is None: continue
            p.sub_(self.state[p]["eps"])

        # 4. 执行NRSAM更新 (θ_{k+1} = θ_k - λ·prev_adv_grad + μ·accum_grad)
        for i, p in enumerate(self.model.parameters()):
            if adv_grads[i] is None: continue

            current_lr = self.optimizer.param_groups[0]['lr']
            lam = current_lr
            mu = 0.000000001 * lam
            # 文献公式(7): -λ∇L(θ_{k-1}^adv) + μΣ∇L(θ_i^adv)
            update = -lam * self.prev_adv_grad[i] +mu * self.accum_grad[i]
            p.add_(update)  # 注：add_等效于 θ = θ + update

        # 5. 保存当前对抗梯度 (供下一时间步使用)
        self.prev_adv_grad = adv_grads

