def lr_poly(base_lr, i_iter, alpha=10, beta=0.75, num_steps=250000):
    if i_iter < 0:
        return base_lr
    return base_lr / ((1 + alpha * float(i_iter) / num_steps) ** (beta))



class LRScheduler:
    def __init__(self, learning_rate, warmup_learning_rate=0.0, warmup_steps=2000, num_steps=200000, alpha=10,
                 beta=0.75,
                 double_bias_lr=False, base_weight_factor=False):
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.double_bias_lr = double_bias_lr
        self.base_weight_factor = base_weight_factor

    def __call__(self, optimizer, i_iter):
        if i_iter < self.warmup_steps:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            lr = self.learning_rate

        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
        elif len(optimizer.param_groups) == 2:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter,
                                                                                         alpha=self.alpha,
                                                                                         beta=self.beta,
                                                                                         num_steps=self.num_steps)
        elif len(optimizer.param_groups) == 4:
            optimizer.param_groups[0]['lr'] = lr_poly(lr, lr_i_iter, alpha=self.alpha, beta=self.beta,
                                                      num_steps=self.num_steps)
            optimizer.param_groups[1]['lr'] = (1 + float(self.double_bias_lr)) * lr_poly(lr, lr_i_iter,
                                                                                         alpha=self.alpha,
                                                                                         beta=self.beta,
                                                                                         num_steps=self.num_steps)
            optimizer.param_groups[2]['lr'] = self.base_weight_factor * lr_poly(lr, lr_i_iter, alpha=self.alpha,
                                                                                beta=self.beta,
                                                                                num_steps=self.num_steps)
            optimizer.param_groups[3]['lr'] = (1 + float(self.double_bias_lr)) * self.base_weight_factor * lr_poly(lr,
                                                                                                                   lr_i_iter,
                                                                                                                   alpha=self.alpha,
                                                                                                                   beta=self.beta,
                                                                                                                   num_steps=self.num_steps)
        else:
            raise RuntimeError('Wrong optimizer param groups')

    def current_lr(self, i_iter):
        if i_iter < self.warmup_steps:
            return self.warmup_learning_rate
        else:
            lr_i_iter = max(i_iter - self.warmup_steps, 0)
            return lr_poly(self.learning_rate, lr_i_iter, alpha=self.alpha, beta=self.beta, num_steps=self.num_steps)