from teammate_finetune import (
    load_training_data,
    category_weights,
    word_weights,
)

def align_token_weights(
    tokenizer,
    target,
    word_cats,
    eos_cats,
    category_weights,
    max_length=512,
):
    """
    Assign a loss weight to each BART target token.

    Ordinary tokens: weight 1.
    Corrected-word tokens: category-specific weight.
    EOS: deletion-related weight when applicable.
    """

    encoded = tokenizer(
        text_target=target,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]
    token_ids = encoded["input_ids"]

    # Match the space-separated word convention used by
    # the M2 reconstruction code.
    # An empty target contains zero words.
    words = target.split(" ") if target else []

    if len(words) != len(word_cats):
        raise ValueError(
            f"Word/category mismatch: "
            f"{len(words)} words vs {len(word_cats)} annotations "
            f"for target {target!r}"
        )

    # Map character positions to word indices.
    char_to_word = [-1] * len(target)

    pos = 0
    for word_idx, word in enumerate(words):
        for char_idx in range(pos, pos + len(word)):
            char_to_word[char_idx] = word_idx
        pos += len(word) + 1

    def weight_for_categories(cats):
        if not cats:
            return 1.0

        return max(
            category_weights[c]
            for c in cats
        )

    weights = []

    for token_id, (start, end) in zip(token_ids, offsets):

        # Special tokens, including BOS and EOS.
        if start == end:
            if token_id == tokenizer.eos_token_id:
                weights.append(weight_for_categories(eos_cats))
            else:
                weights.append(1.0)
            continue

        # Use the final character to identify the target word.
        char_idx = end - 1

        if char_idx >= len(char_to_word):
            raise ValueError(
                f"Invalid token offset {(start, end)} "
                f"for target {target!r}"
            )

        word_idx = char_to_word[char_idx]

        if word_idx == -1:
            weights.append(1.0)
        else:
            weights.append(
                weight_for_categories(word_cats[word_idx])
            )

    assert len(weights) == len(token_ids)

    return weights
