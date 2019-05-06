import torch
import torch.autograd as ag
from torch.autograd.function import Function
from torch._thnn import type2backend


class AdaptiveMaxPool2d(Function):
    def __init__(self, widthOut, heightOut):
        super(AdaptiveMaxPool2d, self).__init__()
        self.widthOut = widthOut
        self.heightOut = heightOut

    def forward(self, input):
        output = input.new()
        indices = input.new().long()
        self.save_for_backward(input)
        self.indices = indices
        self._backend = type2backend[input.type()]
        self._backend.SpatialAdaptiveMaxPooling_updateOutput(
            self._backend.library_state, input, output, indices,
            self.widthOut, self.heightOut)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        indices = self.indices
        grad_input = grad_output.new()
        self._backend.SpatialAdaptiveMaxPooling_updateGradInput(
            self._backend.library_state, input, grad_output, grad_input,
            indices)
        return grad_input, None


def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0], size[1])(input)

def pool_layer(input, R1s, size=(7, 7), scale1=1.0):
    output = []
    R1s = R1s.data.float()
    num_R1s = R1s.size(0)
    R1s[:, 1:].mul_(scale1)
    R1s = R1s.long()
    for i in range(num_R1s):
        R1 = R1s[i]
        # im = input.narrow(0, im_idx, 1)

        output.append(adaptive_max_pool(im, size))

    return torch.cat(output, 0)
epoch=new_epoch
if __name__ == '__main__':
    input = ag.Variable(torch.rand(2, 1, 10, 10), requires_grad=False)

    out = pool_layer(input, R1s, size=(16, 16))
