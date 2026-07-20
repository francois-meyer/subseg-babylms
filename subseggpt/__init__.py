from .configuration_subseggpt import SubSegGPTConfig
from .modeling_subseggpt import SubSegGPTModel, SubSegGPTForCausalLM, SubSegGPTPreTrainedModel, SubSegGPTOutput
from .tokenization_subseggpt import SubSegGPTTokenizer
from .data_collator import DataCollatorForSubSegGPT

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor

AutoConfig.register("subseggpt", SubSegGPTConfig)
AutoModel.register(SubSegGPTConfig, SubSegGPTModel)
AutoModelForCausalLM.register(SubSegGPTConfig, SubSegGPTForCausalLM)
AutoTokenizer.register(SubSegGPTConfig, slow_tokenizer_class=SubSegGPTTokenizer)
AutoProcessor.register(SubSegGPTConfig, SubSegGPTTokenizer)

__version__ = "0.1.0"

__all__ = [
    "SubSegGPTConfig",
    "SubSegGPTModel",
    "SubSegGPTForCausalLM",
    "SubSegGPTPreTrainedModel",
    "SubSegGPTOutput",
    "SubSegGPTTokenizer",
    "DataCollatorForSubSegGPT",
]
