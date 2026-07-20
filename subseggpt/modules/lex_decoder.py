import torch
import torch.nn as nn
import torch.nn.functional as F


class LexiconDecoder(nn.Module):
    """
    Lexicon MLP decoder that computes p_lex for subword segments,
    conditioned on the context vector from the transformer backbone.

    Computes log P_lex(segment | h) over a fixed lexicon vocabulary,
    where h is the context representation from the transformer backbone.
    """

    def __init__(
        self,
        lex_vocab_size,
        context_size,
        dropout=0.1,
    ):
        super().__init__()
        self.lex_vocab_size = lex_vocab_size
        self.context_size = context_size

        # Project transformer output to learned context, project learned context to logits over the lexicon
        self.mlp = nn.Sequential(
            nn.Linear(context_size, context_size),
            nn.Dropout(dropout),
            nn.Linear(context_size, lex_vocab_size),
        )

    def forward(self, context):
        """
        Compute log-probabilities over the full lexicon vocabulary.

        Args:
            context: (..., context_size)
                Context vectors from transformer backbone.

        Returns:
            log_probs: (..., lex_vocab_size)
                Log-probabilities over the lexicon.
        """
        logits = self.mlp(context)
        return F.log_softmax(logits, dim=-1)

    def score_segments(self, context, lex_ids):
        """
        Score specific lexicon items at each context position.

        Args:
            context: (..., context_size)
                Context vectors from transformer backbone.
            lex_ids: (...)
                Lexicon indices to score (one per context position).

        Returns:
            log_probs: (...)
                Log-probabilities of the specified lexicon items.
        """
        all_log_probs = self.forward(context)  # (..., lex_vocab_size)
        log_probs = all_log_probs.gather(
            dim=-1, index=lex_ids.unsqueeze(-1)
        ).squeeze(-1)  # (...)

        return log_probs


class MixtureGate(nn.Module):
    """
    Learned mixture gate used to combine char- and lex-decoder log-probabilities
    into a single per-segment log-probability, conditioned on the context vector.
    """

    def __init__(self, context_size):
        super().__init__()
        self.context_size = context_size

        # Project context to a scalar gate logit
        self.gate_mlp = nn.Linear(context_size, 1)
        nn.init.zeros_(self.gate_mlp.bias)

    def forward(self, context, log_p_char, log_p_lex):
        """
        Computes log P(segment | h) = logsumexp(log(gate)   + log P_lex(segment | h),
                                                log(1-gate) + log P_char(segment | h))
        where gate = sigmoid(MLP(h)).

        Args:
            context: (..., context_size)
                Context vectors from transformer backbone.
            log_p_char: (...)
                Character decoder log-probabilities.
            log_p_lex: (...)
                Lexicon decoder log-probabilities.

        Returns:
            log_p_combined: (...)
                Combined log-probabilities (log mixture of char and lex probs).
            gate: (...)
                Per-segment gate values (probability of using the lexicon).
        """
        # Compute gate logit and the corresponding gate value
        gate_logit = self.gate_mlp(context).squeeze(-1)
        gate = torch.sigmoid(gate_logit)

        # Combine in log space: log(gate * P_lex + (1-gate) * P_char)
        log_gate = F.logsigmoid(gate_logit)
        log_one_minus_gate = F.logsigmoid(-gate_logit)
        log_terms = torch.stack([
            log_gate + log_p_lex,
            log_one_minus_gate + log_p_char,
        ], dim=-1)
        log_p_combined = torch.logsumexp(log_terms, dim=-1)

        return log_p_combined, gate
