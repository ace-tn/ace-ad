import torch
import tqdm
from acead.ctmrg import CTMRG
from acead.measurement.measure import measure
from acetn.model.model_factory import model_factory
from acetn.ipeps.ipeps_config import ModelConfig
from acetn.ipeps.ipeps_config import TNConfig
from .tensor_network import TensorNetwork

class Ipeps(TensorNetwork,torch.nn.Module):
    def __init__(self, ipeps_config, ipeps=None):
        self.config      = ipeps_config
        self.dims        = ipeps_config.get('dims')
        self.ctmrg_steps = ipeps_config.get('ctmrg_steps')

        dtype = ipeps_config.get('dtype')
        device = ipeps_config.get('device')
        tn_config = TNConfig(nx=2, ny=2, dims=ipeps_config.get('dims'))
        TensorNetwork.__init__(self, ipeps, tn_config, dtype, device)

        torch.nn.Module.__init__(self)
        self.assign_parameters()

    def assign_parameters(self):
        for site in self.site_list:
            param = torch.nn.Parameter(self[site]['A'])
            self.register_parameter(f'param_{site}', param)
            self[site]['A'] = param

    def set_model(self, model_cls, params, name=None):
        name = "custom" if name is None else name.lower()
        model_factory.register(name, model_cls)
        self.config['model'] = {'name': name, 'params': params}

    def set_model_params(self, **kwargs):
        for key,val in kwargs.items():
            self.config['model']['params'][key] = val
            tqdm.write(f"Model parameter set: {key}={val}")

    def build_model(self):
        model_config = ModelConfig(dtype=self.dtype,
                                   device=self.device,
                                   dim=self.dims["phys"],
                                   **self.config.get('model'))
        return model_factory.create(model_config)

    def forward(self):
        CTMRG(self).forward(self)
        model = self.build_model()
        ene, mag_x, mag_y, mag_z = measure(self, model)
        return ene, mag_x, mag_y, mag_z
