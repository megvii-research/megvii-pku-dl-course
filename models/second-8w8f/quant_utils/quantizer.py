import importlib
import torch
from torch import nn
from .observer import build_observer


def build_quantizer(config, c_axis, weight=None, act_func=None):
    if weight is not None:
        assert isinstance(weight, torch.Tensor), "weight must be a Tensor"
        _cfg = config["W"]
        quantizer = QWeight(_cfg, c_axis, weight)
    elif act_func is not None:
        _cfg = config["A"]
        quantizer = QAct(_cfg, c_axis, act_func)
    else:
        raise NotImplementedError("Only support weight_quantizer or act_quantizer")
    return quantizer


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_int = x.round()
        return x_int

    @staticmethod
    def backward(ctx, grad):
        return grad


class Quantizer(nn.Module):
    def __init__(self, config, c_axis, weight=None, act_func=None):
        super(Quantizer, self).__init__()
        self.type = "Basic"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.granularity = config["GRANULARITY"]
        self.is_symmetry = config["SYMMETRY"]
        self.c_axis = c_axis
        self.bit = config["BIT"]
        self.scale = torch.tensor([1.0])
        self.zero_point = torch.tensor([0.0])
        self.observer = build_observer(config, c_axis)
        self.act_func = act_func
        self.weight = weight
        if self.weight is not None:
            self.observer.update(self.weight.clone())

    def init_quant_params(self):
        pass

    def _reshape_quant_params(self, x):
        dst_shape = [1] * len(x.shape)
        dst_shape[self.c_axis] = -1
        if isinstance(self.scale, nn.Parameter):
            self.scale.data = self.scale.data.reshape(dst_shape)
        else:
            self.scale = self.scale.reshape(dst_shape)
        self.zero_point = self.zero_point.reshape(dst_shape)

    def set_bit(self, bit):
        assert bit >= 0, "only support bit is a non-negative number"
        self.bit = bit

    def quant(self, x):
        pass

    def dequant(self, x):
        pass

    def forward(self, x):
        if self.bit == 0:
            return x
        else:
            self._reshape_quant_params(x)
            x_q = self.quant(x)
            x_dq = self.dequant(x_q)
        return x_dq

    def __repr__(self):
        return "{}, bit={}, granularity={}".format(
            self.type, self.bit, self.granularity
        )

    def update(self, x):
        self.observer.update(x)


class QWeight(Quantizer):
    def __init__(self, config, c_axis, weight):
        super(QWeight, self).__init__(config, c_axis=c_axis, weight=weight)
        self.type = "uniform"

    def init_quant_params(self):
        if self.bit != 0:
            self.scale, self.zero_point = self.observer.calc_quant_params(bit=self.bit)

    def quant(self, x_f):
        if self.is_symmetry:
            x_int = STE.apply(x_f / self.scale) + self.zero_point
            x_q = torch.clamp(x_int, -(2 ** (self.bit - 1)), 2 ** (self.bit - 1) - 1)
        else:
            x_uint = STE.apply(x_f / self.scale) + self.zero_point
            x_q = torch.clamp(x_uint, 0, 2 ** self.bit - 1)
        return x_q

    def dequant(self, x_q):
        x_dq = (x_q - self.zero_point) * self.scale
        return x_dq


class QAct(QWeight):
    def __init__(self, config, c_axis, act_func=None):
        super(__class__.__base__, self).__init__(config, c_axis=c_axis)
        self.type = "uniform"
