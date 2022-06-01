import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch.core import SparseConvTensor
from pcdet.models import load_data_to_gpu
from .quant_module import QuantConv2d, QuantDeConv2d, QuantSparseConv3d, QuantSubMConv3d

OPERATORS_REFACTOR_MAPPING = {
    nn.Conv2d: QuantConv2d,
    nn.ConvTranspose2d: QuantDeConv2d,
    spconv.SparseConv3d: QuantSparseConv3d,
    spconv.SubMConv3d: QuantSubMConv3d,
}


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.cfg = config
        self.quant_module_refactor(self.model)
        self.warpper_sparseconv_bn_activation()
        self.build_quantizer(config)

    def forward(self, x):
        out = self.model(x)
        return out
    
    def quant_module_refactor(self, module: nn.Module):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        """
        prev_quantmodule = None
        for n, m in module.named_children():
            if isinstance(m, tuple(OPERATORS_REFACTOR_MAPPING.keys())):
                QuantModule = OPERATORS_REFACTOR_MAPPING[m.__class__]
                setattr(module, n, QuantModule(m, self.cfg))
                prev_quantmodule = getattr(module, n)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if prev_quantmodule is not None and isinstance(
                    prev_quantmodule.norm_function, nn.Identity
                ):
                    prev_quantmodule.norm_function = m
                    setattr(module, n, nn.Identity())
                else:  # bn前面是个elemenwise-opr
                    continue
            elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardtanh, nn.LeakyReLU, nn.GELU)):
                if prev_quantmodule is not None and isinstance(
                    prev_quantmodule.act_function, nn.Identity
                ):
                    prev_quantmodule.act_function = m
                    setattr(module, n, nn.Identity())
                else:  # relu前面是个elemenwise-opr
                    continue
            elif isinstance(m, nn.Identity):
                continue
            else:
                prev_quantmodule = self.quant_module_refactor(m)
        return prev_quantmodule
    
    def build_quantizer(self, config):
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(OPERATORS_REFACTOR_MAPPING.values())):
                m.build_quantizer(config)
    
    def enable_weight_quant(self):
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(OPERATORS_REFACTOR_MAPPING.values())):
                m.enable_weight_quant()
    
    def enable_act_quant(self):
        for n, m in self.model.named_modules():
            if isinstance(m, tuple(OPERATORS_REFACTOR_MAPPING.values())):
                m.enable_act_quant()

    def warpper_sparseconv_bn_activation(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (QuantSparseConv3d, QuantSubMConv3d)):
                m.norm_act = spconv.SparseSequential(
                    m.norm_function,
                    m.act_function,
                )

    def enable_quant(self, dataloader, size):
        self.calibration(dataloader, size)
        self.enable_weight_quant()
        self.enable_act_quant()

    def _register_hook(self):
        def _forward_hook(module, x_in, x_out):
            if hasattr(module, "output_quantizer") and module.output_quantizer:
                if isinstance(x_out, SparseConvTensor):
                    module.output_quantizer.observer.update(x_out.features.detach())
                else:
                    module.output_quantizer.observer.update(x_out.detach())

        self.forward_hook_handles = []
        for m in self.model.modules():
            if isinstance(m, tuple(OPERATORS_REFACTOR_MAPPING.values())):
                self.forward_hook_handles.append(
                    m.register_forward_hook(hook=_forward_hook)
                )

    def calibration(self, dataloader, size):
        if size != 0:
            self.model.eval()
            if size < 0:  # use all datas
                size = len(dataloader) * dataloader.batch_size
            tmp_size = size
            self._register_hook()
            iter_dataloader = iter(dataloader)
            while tmp_size > 0:
                data = next(iter_dataloader)
                load_data_to_gpu(data)
                with torch.no_grad():
                    self.model(data)
                tmp_size -= dataloader.batch_size
            for h in self.forward_hook_handles:
                h.remove()

    