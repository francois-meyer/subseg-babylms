from .configuration_subsegdeberta import SubSegDeBERTaConfig
from .tokenization_subsegdeberta import SubSegDeBERTaTokenizer, SPECIAL_TOKENS
from .data_collator import DataCollatorForSubSegDeBERTa
from .modeling_subsegdeberta import (
    SubSegDeBERTaModel,
    SubSegDeBERTaForMaskedLM,
    SubSegDeBERTaPreTrainedModel,
    SubSegDeBERTaOutput,
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoProcessor,
)

AutoConfig.register("subsegdeberta", SubSegDeBERTaConfig)
AutoModel.register(SubSegDeBERTaConfig, SubSegDeBERTaModel)
AutoModelForMaskedLM.register(SubSegDeBERTaConfig, SubSegDeBERTaForMaskedLM)
AutoTokenizer.register(SubSegDeBERTaConfig, slow_tokenizer_class=SubSegDeBERTaTokenizer)
AutoProcessor.register(SubSegDeBERTaConfig, SubSegDeBERTaTokenizer)

__version__ = "0.1.0"

__all__ = [
    "SubSegDeBERTaConfig",
    "SubSegDeBERTaTokenizer",
    "SPECIAL_TOKENS",
    "DataCollatorForSubSegDeBERTa",
    "SubSegDeBERTaModel",
    "SubSegDeBERTaForMaskedLM",
    "SubSegDeBERTaPreTrainedModel",
    "SubSegDeBERTaOutput",
]
