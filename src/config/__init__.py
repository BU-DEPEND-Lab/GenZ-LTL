from .experiment_config import *
from .ppo_config import *
from .model_config import *

model_configs = {
    'PointLtl2-v0': zones,
    'PointLtlSafety2-v0': zones_safety,
    'PointLtlSafety3-v0': zones_safety,
    'PointLtlSafety4-v0': zones_safety,
    'PointLtlSafety5-v0': zones_safety,
    'LetterEnv-v0': letter,
    'LetterSafetyEnv-v0': letter_safety,
    'FlatWorld-v0': flatworld,
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'RCOConfig', 'ModelConfig', 'ModelSafetyConfig', 'SetNetConfig', 'model_configs']
