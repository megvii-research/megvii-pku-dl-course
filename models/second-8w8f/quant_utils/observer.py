import torch
from torch import nn
import numpy as np


def build_observer(config, c_axis):
    return {
        # "MSE": MSEObserver,
        "MINMAX": MinMaxObserver,
    }[config["OBSERVER_METHOD"]](config, c_axis)


class Observer(nn.Module):
    def __init__(self, config, c_axis):
        super(Observer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer("min_val", None)
        self.register_buffer("max_val", None)
        self.feature_length = 0
        self.granularity = config["GRANULARITY"]
        self.is_symmetry = config["SYMMETRY"]
        self.c_axis = c_axis

    def update(self, data):
        pass

    def clear_data_cache(self):
        pass

    def calc_quant_params(self, bit=8):
        if self.is_symmetry:
            max_abs_value = torch.maximum(
                torch.abs(self.min_val.to(self.device)),
                torch.abs(self.max_val.to(self.device)),
            )
            scale = max_abs_value / float((2 ** bit - 1) / 2)  # 127.5
            zero_point = torch.zeros(scale.shape).to(scale.device)
        else:
            scale = (self.max_val.to(self.device) - self.min_val.to(self.device)) / (
                2 ** bit - 1
            )
            zero_point = -torch.round(self.min_val.to(self.device) / scale)
        return scale, zero_point

    def __repr__(self):
        if isinstance(self.min_val, (float, torch.Tensor)):
            return "min_value={}, max_value={}".format(
                self.max_val.reshape(-1), self.min_val.reshape(-1)
            )
        else:
            return "min_value & max_value is None"


class MinMaxObserver(Observer):
    def __init__(self, config, c_axis=1):
        super(MinMaxObserver, self).__init__(config, c_axis)

    def update(self, data):
        data_c_first = data.transpose(self.c_axis, 0).reshape(
            data.shape[self.c_axis], -1
        )
        if self.granularity == "layerwise":
            if self.max_val is None:
                self.max_val = data.max()
                self.min_val = data.min()
            else:
                self.max_val = torch.max(self.max_val, data.max())
                self.min_val = torch.min(self.min_val, data.min())
        elif self.granularity == "channelwise":
            if self.max_val is None:
                self.max_val = data_c_first.max(axis=1).values
                self.min_val = data_c_first.min(axis=1).values
            else:
                self.max_val = torch.max(self.max_val, data_c_first.max(axis=1).values)
                self.min_val = torch.min(self.min_val, data_c_first.min(axis=1).values)
        else:
            raise NotImplementedError("no support {}".format(self.granularity))
