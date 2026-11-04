"""Lazy public exports for training components."""

from importlib import import_module

_MODULES = {'DiffusionTrainer': 'diffusion', 'GANTrainer': 'gan', 'ODETrainer': 'ode', 'ScoreDistillationTrainer': 'distillation', 'ConsistencyDistillationTrainer': 'naive_cd'}
__all__ = list(_MODULES)

def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module("." + _MODULES[name], __name__)
    value = getattr(module, "Trainer")
    globals()[name] = value
    return value
