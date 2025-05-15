from torch.autograd import Function

class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        # print('reverse forward')
        ctx.alpha = alpha
        
        return x.view_as(x).clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('reverse backward')
        output = grad_output.neg().clone()
        
        return output, None