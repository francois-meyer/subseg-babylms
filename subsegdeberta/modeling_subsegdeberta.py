"""SubSegDeBERTa: Subword Segmental Masked Language Model."""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from transformers import PreTrainedModel, DebertaV2Model, DebertaV2Config
from transformers.modeling_outputs import ModelOutput

from .configuration_subsegdeberta import SubSegDeBERTaConfig
from .modules.word_context_encoder import LSTMWordContextEncoder
from .modules.segment_scorer import SegmentScorer
from .modules.forward_algorithm import (
    forward_algorithm,
    viterbi_decode,
)


@dataclass
class SubSegDeBERTaOutput(ModelOutput):
    """Output type for SubSegDeBERTa."""

    loss: Optional[torch.FloatTensor] = None
    word_log_probs: Optional[torch.FloatTensor] = None
    seg_logp: Optional[torch.FloatTensor] = None
    log_alpha: Optional[torch.FloatTensor] = None
    gate: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SubSegDeBERTaPreTrainedModel(PreTrainedModel):
    """Base class for SubSegDeBERTa."""

    config_class = SubSegDeBERTaConfig
    base_model_prefix = "subsegdeberta"
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


class SubSegDeBERTaModel(SubSegDeBERTaPreTrainedModel):
    """
    SubSegDeBERTa encoder: character-level bidirectional transformer which
    produces character-level output embeddings (no segmentation or LM head).

    Used by downstream tasks (e.g. finetuning classifiers) that need
    contextualised output embeddings rather than logits/loss.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        deberta_config = DebertaV2Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            relative_attention=config.relative_attention,
            position_buckets=config.position_buckets,
            max_relative_positions=config.max_relative_positions,
            pos_att_type=config.pos_att_type,
            norm_rel_ebd=config.norm_rel_ebd,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            pad_token_id=config.pad_token_id,
        )
        self.backbone = DebertaV2Model(deberta_config)

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
    ):
        """
        Forward pass returning DeBERTa-v2 output, which includes final-layer output embeddings
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


class SubSegDeBERTaForMaskedLM(SubSegDeBERTaPreTrainedModel):
    """
    SubSegDeBERTa for masked language modeling.

    Uses a DeBERTa-v2 encoder/backbone for context representations and 
    a subword segmental decoder to generate masked words.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Bidirectional transformer backbone
        self.subsegdeberta = SubSegDeBERTaModel(config)

        # Word context encoder for each position in masked word
        self.word_context_encoder = LSTMWordContextEncoder(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            context_size=config.hidden_size,
            char_embed_dim=config.char_embed_dim,
            lstm_hidden_size=config.word_context_hidden_size,
            num_layers=config.word_context_num_layers,
            dropout=config.dropout,
            start_token_id=config.cls_token_id,
        )

        # Segment scorer based on word context encoder, char decoder and lexicon decoder
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
            pad_token_id=config.pad_token_id,
            start_token_id=config.cls_token_id,
            n_special=7,  # SubSegDeBERTa tokenizer specials tokens: pad, cls, sep, mask, unk, eow, seg
            n_alpha=config.n_alpha,
        )

        # Init only word context encoder and segment scorer (leave rest to DeBERTa and SubSegDeBERTaModel init).
        self.word_context_encoder.apply(self._init_weights)
        self.segment_scorer.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_pos: Optional[torch.LongTensor] = None,
        target_chars: Optional[torch.LongTensor] = None,
        word_lens: Optional[torch.LongTensor] = None,
        lex_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_gate: bool = False,
    ) -> SubSegDeBERTaOutput:
        """
        Forward pass for SubSegDeBERTa masked LM, computing masked subword segmental loss and returning
        SubSegDeBERTaOutput with masked LM loss, per-word logps, per-seg logps, log_alphas, etc.

        Args:
            input_ids (batch_size, max_seq_len): char ids with each masked word set to one [MASK] char
            attention_mask (batch_size, max_seq_len): encoder padding mask
            masked_pos (num_masked_words, 2): (batch_idx, position) of each [MASK] char in input_ids
            target_chars (num_masked_words, max_masked_len): padded chars + <eow> of masked words
            word_lens (num_masked_words, ): actual lengths (including <eow>) of masked words
            lex_ids (num_masked_words, max_masked_len, max_seg_len): lexicon index per (start, segment_len)
        """
        # Get context representations from encoder/backbone
        encoder_out = self.subsegdeberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = encoder_out.last_hidden_state  # (batch_size, max_seq_len, hidden_size)

        num_masked_words = 0 if masked_pos is None else masked_pos.size(0)
        if num_masked_words == 0:  # If no masked words in this batch, return zero loss with valid gradient.            
            loss = hidden_states.sum() * 0.0
            return SubSegDeBERTaOutput(loss=loss)

        # Get [MASK] context vectors.
        mask_outputs = hidden_states[masked_pos[:, 0], masked_pos[:, 1]]  # (num_masked_words, hidden_size)

        # Compute contexts for each word starting position
        word_contexts = self.word_context_encoder(mask_outputs, target_chars)  # (num_masked_words, max_masked_len, hidden_size)

        # Compute segment log-probabilities
        scorer_output = self.segment_scorer(
            word_contexts=word_contexts,
            target_chars=target_chars,
            word_lens=word_lens,
            lex_ids=lex_ids,
            return_gate=return_gate,
        )
        seg_logp = scorer_output["seg_logp"]  # (num_masked_words, max_masked_len, max_seg_len)
        
        # Compute marginal log-probabilities with dynamic programming
        log_alpha = forward_algorithm(
            seg_logp=seg_logp,
            max_seg_len=self.config.max_seg_len,
        ) # (batch_size, content_len + 1)    

        # Use per-word length to index log_alpha
        word_log_probs = log_alpha[torch.arange(num_masked_words, device=seg_logp.device), word_lens]

        # Compute loss = negative of marginal log-likelihood averaged across masked words or chars
        if self.config.loss_normalization == "char":
            loss = -(word_log_probs / word_lens.clamp_min(1).float()).mean()
        else:
            loss = -word_log_probs.mean()

        return SubSegDeBERTaOutput(
            loss=loss,
            word_log_probs=word_log_probs,
            seg_logp=seg_logp,
            log_alpha=log_alpha,
            gate=scorer_output.get("gate"),
            hidden_states=encoder_out.hidden_states if output_hidden_states else None,
            attentions=encoder_out.attentions if output_attentions else None,
        )

    @torch.no_grad()
    def get_best_segmentation(self, input_ids, attention_mask, masked_pos, target_chars, word_lens, lex_ids):
        """
        Get the best segmentation of each masked word using Viterbi decoding.

        Args:
            input_ids (batch_size, max_seq_len): char ids with each masked word set to one [MASK] char
            attention_mask (batch_size, max_seq_len): encoder padding mask
            masked_pos (num_masked_words, 2): (batch_idx, position) of each [MASK] char in input_ids
            target_chars (num_masked_words, max_masked_len): padded chars + <eow> of masked words
            word_lens (num_masked_words, ): actual lengths (including <eow>) of masked words
            lex_ids (num_masked_words, max_masked_len, max_seg_len): lexicon index per (start, segment_len)

        Returns:
            best_log_prob: (num_masked_words,) Log-probability of best segmentation per masked word.
            segmentations: List of best segmentations for each masked word.
                Each segmentation is a list of (start, end) tuples in 0-indexed content positions.
                E.g. [(0, 2), (2, 3)] means a length-2 subword followed by a length-1 subword.
        """
        # Get context representations from encoder/backbone
        encoder_out = self.subsegdeberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_out.last_hidden_state

        # Get [MASK] context vectors.
        mask_outputs = hidden_states[masked_pos[:, 0], masked_pos[:, 1]]

        # Compute contexts for each word starting position
        word_contexts = self.word_context_encoder(mask_outputs, target_chars)

        # Compute segment log-probabilities
        scorer_output = self.segment_scorer(
            word_contexts=word_contexts,
            target_chars=target_chars,
            word_lens=word_lens,
            lex_ids=lex_ids,
        )

        # Extract highest-probability segmentations with Viterbi algorithm
        return viterbi_decode(
            seg_logp=scorer_output["seg_logp"],
            max_seg_len=self.config.max_seg_len,
            content_lens=word_lens,
        )
