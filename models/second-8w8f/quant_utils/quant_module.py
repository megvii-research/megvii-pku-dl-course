import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.constants import FILTER_HWIO
from spconv.pytorch.core import SparseConvTensor
from .quantizer import build_quantizer


class QuantConv2d(nn.Module):
    """
    Quantized Conv2d that can perform quantized convolution or normal convolution.
    To activate quantization, please use enable_quant function.
    """

    def __init__(
        self, org_module: nn.Conv2d, config, w_c_axis=0, a_c_axis=1
    ):
        super(QuantConv2d, self).__init__()
        self.cfg = config
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.w_c_axis = w_c_axis
        self.a_c_axis = a_c_axis
        self.weight_quantizer = None
        self.output_quantizer = None
        self.use_aq = False
        self.norm_function = nn.Identity()
        self.act_function = nn.Identity()

    def build_quantizer(self, cfg):
        self.weight_quantizer = build_quantizer(
            cfg, c_axis=self.w_c_axis, weight=self.weight
        )
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def enable_weight_quant(self):
        self.weight_quantizer.init_quant_params()
        self.weight.data = self.weight_quantizer(self.weight).data

    def enable_act_quant(self):
        self.output_quantizer.init_quant_params()
        self.use_aq = True

    def forward(self, x_in: torch.Tensor):
        out = F.conv2d(x_in, self.weight, self.bias, **self.fwd_kwargs)
        out = self.norm_function(out)
        out = self.act_function(out)
        if self.use_aq:
            out = self.output_quantizer(out)
        return out


class QuantDeConv2d(nn.Module):
    """
    Quantized DeConv2d that can perform quantized Deconvolution or normal Deconvolution.
    To activate quantization, please use enable_quant function.
    """

    def __init__(
        self,
        org_module: nn.ConvTranspose2d,
        config,
        w_c_axis=1,
        a_c_axis=1,
    ):
        super(QuantDeConv2d, self).__init__()
        self.cfg = config
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=0,
            output_padding=org_module.output_padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
        )
        pad_h, pad_w = org_module.padding
        self.padding = (pad_w, pad_w, pad_h, pad_h)
        self.padding_value = 0
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.w_c_axis = w_c_axis
        self.a_c_axis = a_c_axis
        self.weight_quantizer = None
        self.output_quantizer = None
        self.use_aq = False
        self.norm_function = nn.Identity()
        self.act_function = nn.Identity()

    def build_quantizer(self, cfg):
        self.weight_quantizer = build_quantizer(
            cfg, c_axis=self.w_c_axis, weight=self.weight
        )
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def enable_weight_quant(self):
        self.weight_quantizer.init_quant_params()
        self.weight.data = self.weight_quantizer(self.weight).data

    def enable_act_quant(self):
        self.output_quantizer.init_quant_params()
        self.use_aq = True

    def forward(self, x_in: torch.Tensor):
        x_in = nn.functional.pad(
            x_in, self.padding, mode="constant", value=self.padding_value
        )
        out = F.conv_transpose2d(x_in, self.weight, self.bias, **self.fwd_kwargs)
        out = self.norm_function(out)
        out = self.act_function(out)
        if self.use_aq:
            out = self.output_quantizer(out)
        return out


class QuantSparseConv3d(spconv.SparseConv3d):
    def __init__(
        self,
        org_module: spconv.SparseConv3d,
        config,
    ):
        super(QuantSparseConv3d, self).__init__(org_module.in_channels,
                                            org_module.out_channels,
                                            org_module.kernel_size,
                                            org_module.stride,
                                            org_module.padding,
                                            org_module.dilation,
                                            org_module.groups,
                                            org_module.bias,
                                            indice_key=org_module.indice_key,
                                            algo=org_module.algo,
                                            fp32_accum=org_module.fp32_accum,
                                            name=org_module.name)
        self.cfg = config
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.weight_quantizer = None
        self.output_quantizer = None
        self.use_aq = False
        self.norm_function = nn.Identity()
        self.act_function = nn.Identity()
        self.norm_act = None

        if self.algo == ConvAlgo.Native:
            if FILTER_HWIO:
                # RSCK
                self.w_c_axis = -1
            else:
                # RSKC
                self.w_c_axis = -2
        else:
            # KRSC
            self.w_c_axis = 0
        self.a_c_axis = 1

    def build_quantizer(self, cfg):
        self.weight_quantizer = build_quantizer(
            cfg, c_axis=self.w_c_axis, weight=self.weight
        )
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def enable_weight_quant(self):
        self.weight_quantizer.init_quant_params()
        self.weight.data = self.weight_quantizer(self.weight).data

    def enable_act_quant(self):
        self.output_quantizer.init_quant_params()
        self.use_aq = True

    def forward(self, x_in: SparseConvTensor):
        out_tensor = super().forward(x_in)
        if self.norm_act is not None:
            out_tensor = self.norm_act(out_tensor)
        if self.use_aq:
            out_tensor = out_tensor.replace_feature(self.output_quantizer(out_tensor.features))
        return out_tensor


class QuantSubMConv3d(spconv.SubMConv3d):
    def __init__(
        self,
        org_module: spconv.SubMConv3d,
        config,
    ):
        super(QuantSubMConv3d, self).__init__(org_module.in_channels,
                                            org_module.out_channels,
                                            org_module.kernel_size,
                                            org_module.stride,
                                            org_module.padding,
                                            org_module.dilation,
                                            org_module.groups,
                                            org_module.bias,
                                            indice_key=org_module.indice_key,
                                            algo=org_module.algo,
                                            fp32_accum=org_module.fp32_accum,
                                            name=org_module.name)
        self.cfg = config
        self.weight = org_module.weight
        self.bias = org_module.bias
        self.weight_quantizer = None
        self.output_quantizer = None
        self.use_aq = False
        self.norm_function = nn.Identity()
        self.act_function = nn.Identity()
        self.norm_act = None

        if self.algo == ConvAlgo.Native:
            if FILTER_HWIO:
                # RSCK
                self.w_c_axis = -1
            else:
                # RSKC
                self.w_c_axis = -2
        else:
            # KRSC
            self.w_c_axis = 0
        self.a_c_axis = 1

    def build_quantizer(self, cfg):
        self.weight_quantizer = build_quantizer(
            cfg, c_axis=self.w_c_axis, weight=self.weight
        )
        self.output_quantizer = build_quantizer(
            cfg, c_axis=self.a_c_axis, act_func=self.act_function
        )

    def enable_weight_quant(self):
        self.weight_quantizer.init_quant_params()
        self.weight.data = self.weight_quantizer(self.weight).data

    def enable_act_quant(self):
        self.output_quantizer.init_quant_params()
        self.use_aq = True

    def forward(self, x_in: SparseConvTensor):
        out_tensor = super().forward(x_in)
        if self.norm_act is not None:
            out_tensor = self.norm_act(out_tensor)
        if self.use_aq:
            out_tensor = out_tensor.replace_feature(self.output_quantizer(out_tensor.features))
        return out_tensor
