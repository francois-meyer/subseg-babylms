"""SubSegGPT: Subword Segmental Decoder-only Language Model."""

import math

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from transformers import PreTrainedModel, GPT2Model, GPT2Config
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPastAndCrossAttentions

from .configuration_subseggpt import SubSegGPTConfig
from .modules.char_decoder import LSTMCharDecoder
from .modules.lex_decoder import LexiconDecoder, MixtureGate
from .modules.segment_scorer import SegmentScorer
from .modules.forward_algorithm import (
    forward_algorithm,
    viterbi_decode,
)


@dataclass
class SubSegGPTOutput(ModelOutput):
    """Output type for SubSegGPT."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    log_alpha: Optional[torch.FloatTensor] = None
    seg_logp: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SubSegGPTPreTrainedModel(PreTrainedModel):
    """Base class for SubSegGPT."""

    config_class = SubSegGPTConfig
    base_model_prefix = "subseggpt"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)


class SubSegGPTModel(SubSegGPTPreTrainedModel):
    """
    SubSegGPT backbone: character-level transformer which produces 
    character-level output embeddings (no segmentation or LM head).

    Used by downstream tasks (e.g. finetuning classifiers) that need
    contextualised output embeddings rather than logits.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            activation_function=config.activation_function,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.attention_dropout,
            layer_norm_epsilon=config.layer_norm_eps,
        )
        self.backbone = GPT2Model(gpt2_config)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """
        Forward pass returning GPT-2 output, which includes final-layer output embeddings
        last_hidden_state of shape (batch_size, sequence_length, hidden_size).
        """
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )


class SubSegGPTForCausalLM(SubSegGPTPreTrainedModel):
    """
    SubSegGPT for causal language modeling.

    Uses a GPT2 backbone for context representations and character/lexicon
    decoders for segment probability computation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Transformer backbone
        self.subseggpt = SubSegGPTModel(config)

        # Segment scorer based on char decoder and lexicon decoder
        self.segment_scorer = SegmentScorer(
            vocab_size=config.vocab_size,
            lex_vocab_size=config.lex_vocab_size,
            context_size=config.hidden_size,
            char_decoder_hidden_size=config.char_decoder_hidden_size,
            char_decoder_num_layers=config.char_decoder_num_layers,
            char_embed_dim=config.char_embed_dim,
            max_seg_len=config.max_seg_len,
            dropout=config.dropout,
            seg_end_token_id=config.seg_end_token_id,
            n_special=5,  # SubSegGPT tokenizer special tokens: pad, bos, eos, seg, unk
            n_alpha=config.n_alpha,
        )

        # Init only segment scorer (leave the rest to GPT-2 and SubSegGPTModel init).
        self.segment_scorer.apply(self._init_weights)

        # Tokenizer directly available for computing lex_ids when not provided by 
        # the data collator (e.g. in BabyLM eval pipeline).
        self._tokenizer = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        from .tokenization_subseggpt import SubSegGPTTokenizer
        # Forward download kwargs so tokenizer loads from the same local dir or Hub revision as model.
        tok_kwargs = {k: kwargs[k] for k in
                      ("revision", "cache_dir", "token", "local_files_only", "force_download")
                      if k in kwargs}
        try:
            model._tokenizer = SubSegGPTTokenizer.from_pretrained(
                str(pretrained_model_name_or_path), **tok_kwargs
            )
        except Exception:
            model._tokenizer = None
        return model

    def get_input_embeddings(self):
        return self.subseggpt.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.subseggpt.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        lex_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> SubSegGPTOutput:
        """
        Forward pass for SubSegGPT decoder LM, returning SubSegGPTOutput with loss, logits, log_alpha, etc.
        """
        # Compute lex_ids if not provided and tokenizer directly available.
        # Common cases will be training (data collator provides lex_ids as part of batch)
        # and BabyLM eval (no lex_ids provided so tokenizer loaded in from_pretrained).
        if lex_ids is None:
            if self._tokenizer is None:
                raise ValueError(
                    "lex_ids not provided and no tokenizer attached"
                )
            lex_ids = self._tokenizer.compute_lex_ids(input_ids)

        # Get context representations from backbone
        backbone_outputs = self.subseggpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        hidden_states = backbone_outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Compute content lengths from attention_mask (subtract 1 for BOS)
        if attention_mask is not None:
            content_lens = attention_mask.sum(dim=1).long() - 1  # (batch_size,)
        else:
            content_lens = None

        # Compute segment log-probabilities
        scorer_output = self.segment_scorer(
            hidden_states=hidden_states,
            input_ids=input_ids,
            lex_ids=lex_ids,
            content_lens=content_lens,
        )
        seg_logp = scorer_output['seg_logp'] # (batch_size, content_len, max_seg_len)

        # Compute marginal log-probabilities with dynamic programming
        log_alpha = forward_algorithm(
            seg_logp=seg_logp,
            max_seg_len=self.config.max_seg_len,
        ) # (batch_size, content_len + 1)    

        # Compute loss = negative marginal log-likelihood = -log P(content)
        loss = None
        batch_size = input_ids.size(0)
        if labels is not None:
            if content_lens is not None:
                # Use per-sequence content length to index log_alpha
                sentence_log_probs = log_alpha[
                    torch.arange(batch_size, device=input_ids.device), content_lens
                ]
            else:
                # No padding
                sentence_log_probs = log_alpha[:, -1]
            loss = -sentence_log_probs.mean()

        # Compute logits based on marginalised log-probs
        logits = self.compute_logits(
            input_ids=input_ids, log_alpha=log_alpha, content_lens=content_lens
        )

        return SubSegGPTOutput(
            loss=loss,
            logits=logits,
            log_alpha=log_alpha,
            seg_logp=seg_logp,
            hidden_states=backbone_outputs.hidden_states if output_hidden_states else None,
            attentions=backbone_outputs.attentions if output_attentions else None,
        )

    def compute_logits(self, input_ids, log_alpha, content_lens=None):
        """
        This is for compatibility with the BabyLM zer-shot eval pipeline, which expects
        per-token logits to compute and compare log-probabilities of minimal pairs.
        
        SubSegGPT computes the log-probability of a whole sentence. To convert this to
        per-character logits, we spread sentence log-probability uniformly across positions:
            per_char_log_prob = log_alpha[content_len] / content_len
        and compute the corresponding logit values. We spread the remaining probability
        distribution uniformly across the other vocabulary items (these are not used by the
        eval pipeline anyway - it only extracts the logits of the actual target sentence).

        The BabyLM eval pipeline can recover the target sentence log-probability from our per-
        character logits, which would allow minimal pair comparison to works as intended by 
        comparing the log-probabilities of two sentences. 
        
        NOTE: The BabyLM eval pipeline masks out matching sentence context and only compares 
        log-probabilities of the spans that are corrupted in the minimal pair. Since we
        spread log-probabily evenly across characters, we are not comparing exactly what the
        BabyLM eval pipeline should compare - log-probs of specific phrases. Instead, we are
        comparing overall sentence log-probs, but with a weird normalisation (divided by
        sentence character length and then multiplied by unmasked phrase character length).
        A better solution would be to change the  BabyLM eval pipeline code to compare full
        sentence log-probabilities, instead of masking out matching surrounding context.
        
        To do this, change babylm-eval/strict/evaluation_pipeline/sentence_zero_shot/dataset.py:89
        from        phrase_mask = [0 for _ in range(len(tokens))]
        to          phrase_mask = [1 for _ in range(len(tokens))]
        
        Args:
            input_ids: (batch_size, seq_len) Character indices starting with BOS.
            log_alpha: (batch_size, content_len + 1) Forward algorithm output.
            content_lens: Optional (batch_size,) Per-sequence length (excludes BOS and padding)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = log_alpha.dtype
        V = self.config.vocab_size

        content_len = seq_len - 1 # exclude BOS token

        if content_lens is not None:
            # Divide by per-sequence content length if available
            sentence_log_probs = log_alpha[
                torch.arange(batch_size, device=device), content_lens
            ]  # (batch,)
            logp_per_pos = (sentence_log_probs / content_lens.float()).unsqueeze(-1)  # (batch, 1)
        else:
            # Divide by content_len if no padding
            sentence_log_probs = log_alpha[:, -1]
            logp_per_pos = (sentence_log_probs / content_len).unsqueeze(-1)  # (batch, 1)

        # Target characters that logits[t] should predict
        target_chars = input_ids[:, 1:]  # (batch, content_len)

        # Build valid log-probability distribution at each position
        #   P(target) = exp(logp_per_pos)
        #   P(other)  = (1 - P(target)) / (V - 1)
        p_target = logp_per_pos.exp()  # (batch, 1)
        # log1p(x) = log(1+x) and is better for numerical accuracy
        log_p_other = torch.log1p(-p_target) - math.log(V - 1)  # (batch, 1)

        # Initialize all positions with uniform log-prob (for padding + last position)
        logits = torch.full(
            (batch_size, seq_len, V),
            -math.log(V),
            device=device,
            dtype=dtype,
        )

        # Fill content positions 0..content_len-1 with log_p_other as default
        logits[:, :content_len, :] = log_p_other.unsqueeze(-1).expand(-1, content_len, V)

        # Override the target character entry with the per-position log-prob
        logits[:, :content_len, :].scatter_(
            dim=2,
            index=target_chars.unsqueeze(-1),
            src=logp_per_pos.expand(-1, content_len).unsqueeze(-1),
        )

        # For padded positions, reset to uniform (phrase_mask zeros them out anyway)
        if content_lens is not None:
            for b in range(batch_size):
                cl = content_lens[b].item()
                if cl < content_len:
                    logits[b, cl:, :] = -math.log(V)

        return logits


    @torch.no_grad()
    def get_best_segmentation(self, input_ids, attention_mask=None, lex_ids=None):
        """
        Get the best segmentations of a batch using Viterbi decoding.

        Args:
            input_ids: (batch_size, seq_len) Character indices.
            attention_mask: Optional attention mask.
            lex_ids: Optional lexicon indices.

        Returns:
            best_log_prob: (batch_size,) Log-probability of best segmentation.
            segmentations: List of best segmentations for each batch item.
                Each segmentation is a list of (start, end) tuples in 0-indexed content positions.
                E.g. [(0, 2), (2, 5)] means segments c_1..c_2 and c_3..c_5.
        """
        # Compute lex_ids if not provided and tokenizer directly available.
        if lex_ids is None:
            if self._tokenizer is None:
                raise ValueError(
                    "lex_ids not provided and no tokenizer attached"
                )
            lex_ids = self._tokenizer.compute_lex_ids(input_ids)

        # Get context representations from backbone
        backbone_outputs = self.subseggpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_outputs.last_hidden_state

        # Compute content lengths from attention_mask (subtract 1 for BOS)
        if attention_mask is not None:
            content_lens = attention_mask.sum(dim=1).long() - 1
        else:
            content_lens = None

        # Compute segment log-probabilities
        scorer_output = self.segment_scorer(
            hidden_states=hidden_states,
            input_ids=input_ids,
            lex_ids=lex_ids,
            content_lens=content_lens,
        )

        # Extract highest-probability segmentations with Viterbi algorithm
        return viterbi_decode(
            seg_logp=scorer_output['seg_logp'],
            max_seg_len=self.config.max_seg_len,
            content_lens=content_lens,
        )
