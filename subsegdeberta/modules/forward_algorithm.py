"""
Dynamic programming algorithms required for subword segmental modelling
(for marginalisation and best segmentation extraction).
"""

import torch


def _safe_logsumexp(x):
    """
    logsumexp along dim=-1 that returns -inf when all inputs are -inf.
    Standard ``torch.logsumexp`` produces NaN  when all inputs are -inf.

    Args:
        x: (batch_size, num_terms) tensor.

    Returns:
        (batch_size,) logsumexp over dimension -1
    """

    # Replace all-inf rows with zeros so logsumexp gradient is finite
    all_neg_inf = (x == float('-inf')).all(dim=-1)  # (batch,)
    safe_x = torch.where(
        all_neg_inf.unsqueeze(-1).expand_as(x),
        torch.zeros_like(x),
        x,
    )

    # Safe logsumexp
    result = torch.logsumexp(safe_x, dim=-1)

    # Restore correct -inf for those rows
    restored_result = torch.where(all_neg_inf, torch.full_like(result, float('-inf')), result)
    
    return restored_result


def forward_algorithm(seg_logp, max_seg_len):
    """
    Compute marginal log-probabilities using forward algorithm.

    Notation and indexing:
        - Content sequence: c_1, c_2, ..., c_n (1-indexed)
        - seg_logp uses 0-indexed positions (0 to seq_len-1)
        - seg_logp[b, start_pos, seg_len-1] = log P(c_{start_pos+1}, ..., c_{start_pos+seg_len})

    Args:
        seg_logp: (batch_size, seq_len, max_seg_len)
            (already excludes BOS token, so 0th position is start of target sequence)
            seg_logp[b, start_pos, l-1] = log P(segment of length l starting at start_pos)
        max_seg_len: Maximum segment length.

    Returns:
        log_alpha: (batch_size, seq_len + 1)    
            log_alpha[b, t] = log P(c_1, c_2, ..., c_t) marginalised over all segmentations up to position t
            log_alpha[b, 0] = 0 (base case)
            log_alpha[b, n] = log P(entire sequence)
    """
    batch_size, seq_len, _ = seg_logp.shape
    device = seg_logp.device
    dtype = seg_logp.dtype

    log_alpha = torch.full(
        (batch_size, seq_len + 1),
        float('-inf'),
        device=device,
        dtype=dtype,
    )

    # Initialise base case
    log_alpha[:, 0] = 0.0

    # Forward pass: compute log_alpha[t] for t = 1, 2, ..., seq_len
    for t in range(1, seq_len + 1):
     
        # Loop over all segments ending at position t
        terms = []
        for seg_len in range(1, min(max_seg_len, t) + 1):
            # Segment c_{start_pos+1}, ..., c_{start_pos+seg_len} = c_{t-seg_len+1}, ..., c_t
            start_pos = t - seg_len # 0-indexed starting position

            # Get marginalisation up to last character before start of segment
            log_alpha_start_pos = log_alpha[:, start_pos] # 1-indexed ending position of preceding sequence

            # Get log-probs for segment ending at 1-indexed position t, which starts at 0-indexed position start_pos
            seg_log_prob = seg_logp[:, start_pos, seg_len - 1]

            # Compute log_alpha[t - seg_len] + log P(c_{t-seg_len+1}, ..., c_t)
            term = log_alpha_start_pos + seg_log_prob
            terms.append(term) 

        # Sum over all segments ending at position t
        terms_tensor = torch.stack(terms, dim=-1)  # (batch_size, num_terms)
        log_alpha[:, t] = _safe_logsumexp(terms_tensor)

    return log_alpha


def viterbi_decode(seg_logp, max_seg_len, content_lens=None):
    """
    Find highest-probability subword segmentations using Viterbi algorithm.
    Same structure as forward_algorithm but uses max instead of logsumexp.

    Args:
        seg_logp: (batch_size, seq_len, max_seg_len)
            (already excludes BOS token, so 0th position is start of target sequence)
            seg_logp[b, start_pos, l-1] = log P(segment of length l starting at start_pos)
        max_seg_len: Maximum segment length.
        content_lens: Optional (batch_size,) per-item content length.
            When provided, backtrace starts at each item's content_len to exclude padding.

    Returns:
        best_log_prob: (batch_size,) Log-probability of best segmentation.
        segmentations: List of best segmentations for each batch item.
            Each segmentation is a list of (start, end) tuples in 0-indexed content positions.
            E.g. [(0, 2), (2, 5)] means segments c_1..c_2 and c_3..c_5.
    """
    batch_size, seq_len, _ = seg_logp.shape
    device = seg_logp.device
    dtype = seg_logp.dtype

    # Viterbi scores: viterbi[t] = best log P(c_1, ..., c_t)
    viterbi = torch.full(
        (batch_size, seq_len + 1),
        float('-inf'),
        device=device,
        dtype=dtype,
    )
    viterbi[:, 0] = 0.0

    # Backpointers: best_seg_len[t] = length of segment ending at t in best path
    backpointers = torch.zeros(
        (batch_size, seq_len + 1),
        device=device,
        dtype=torch.long,
    )

    # Forward pass (same structure as forward_algorithm, but max instead of logsumexp)
    for t in range(1, seq_len + 1):
        # Track best segment ending at position t
        best_score = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
        best_len = torch.zeros((batch_size,), device=device, dtype=torch.long)

        for seg_len in range(1, min(max_seg_len, t) + 1):
            start_pos = t - seg_len # 0-indexed starting position
            score = viterbi[:, start_pos] + seg_logp[:, start_pos, seg_len - 1]

            better = score > best_score
            best_score = torch.where(better, score, best_score)
            best_len = torch.where(better, torch.tensor(seg_len, device=device), best_len)

        viterbi[:, t] = best_score
        backpointers[:, t] = best_len

    # Get best final position log-prob
    if content_lens is not None:
        best_log_prob = viterbi[torch.arange(batch_size, device=device), content_lens]
    else:
        best_log_prob = viterbi[:, seq_len]

    # Backtrack to get segmentations
    segmentations = []
    for b in range(batch_size):
        segments = []
        t = int(content_lens[b].item()) if content_lens is not None else seq_len
        while t > 0:
            seg_len = backpointers[b, t].item()
            if seg_len == 0:
                # No valid segmentation found
                break
            start = t - seg_len
            # Tuple (start, t): 0-indexed positions covering characters c_{start+1}..c_t (1-indexed)
            segments.append((start, t))
            t = start
        segments.reverse()
        segmentations.append(segments)

    return best_log_prob, segmentations
