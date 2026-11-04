"""Lazy public exports for training components."""

from importlib import import_module

_MODULES = {'CausalDiffusion': 'diffusion', 'CausVid': 'causvid', 'DMD': 'dmd', 'GAN': 'gan', 'SiD': 'sid', 'ODERegression': 'ode_regression', 'NaiveConsistency': 'naive_consistency'}
__all__ = list(_MODULES)

def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module("." + _MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
