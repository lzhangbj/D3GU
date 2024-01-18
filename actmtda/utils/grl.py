from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, delta):
        ctx.delta = delta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.delta
        return output, None


class ForwardLayerF(Function):

    @staticmethod
    def forward(ctx, x, delta):
        ctx.delta = delta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.delta
        return output, None

