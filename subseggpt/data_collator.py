from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class DataCollatorForSubSegGPT:
    """
    Data collator for causal language modelling with SubSegGPT.
    Pads input_ids/attention_mask to common length and precomputes lex_ids.
    """

    tokenizer: Any
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"]} for f in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["lex_ids"] = self.tokenizer.compute_lex_ids(batch["input_ids"])
        return batch