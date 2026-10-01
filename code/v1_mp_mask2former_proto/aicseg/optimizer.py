"""Keep the control's backbone grouping; exempt new query embeddings only."""

from torch import nn
from mmseg.engine.optimizers import LearningRateDecayOptimizerConstructor
from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class AICMask2FormerOptimizerConstructor(LearningRateDecayOptimizerConstructor):
    """Preserve every backbone LR/WD value from the v04 constructor.

    The shared constructor already exempts biases and 1-D norm parameters.
    It otherwise decays 2-D query/level embeddings. Split those head embedding
    weights into a no-decay group without changing their learning rates.
    """

    def add_params(self, params, module, **kwargs):
        original = []
        super().add_params(original, module, **kwargs)
        embedding_ids = {
            id(child.weight) for name, child in module.named_modules()
            if name.startswith('decode_head.') and isinstance(child, nn.Embedding)
        }
        for group in original:
            regular, embedding = [], []
            regular_names, embedding_names = [], []
            for name, param in zip(group['param_names'], group['params']):
                if id(param) in embedding_ids:
                    embedding.append(param)
                    embedding_names.append(name)
                else:
                    regular.append(param)
                    regular_names.append(name)
            if regular:
                params.append(dict(group, params=regular,
                                   param_names=regular_names))
            if embedding:
                params.append(dict(group, params=embedding,
                                   param_names=embedding_names,
                                   weight_decay=0.,
                                   group_name=group['group_name'] + '_embedding'))
