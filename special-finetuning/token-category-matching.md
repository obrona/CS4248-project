# Matching target tokens to error categories

This note explains how `finetune.py` decides which ERRANT error category
(and therefore which loss weight) each target token belongs to. Line numbers refer to
`finetune.py`.

## TL;DR

- Categories are attached to the **gold target**, never to the model's output. They are
  computed once, before training, from the M2 annotations.
- Pipeline: M2 edits → category per target **word** → weight per word → weight per
  BPE **token** (via character offsets) → weighted cross-entropy.
- Teacher forcing keeps every decoder position aligned with the gold target, so
  `logits[t]`, `labels[t]` and `label_weights[t]` always refer to the same position.

```
M2 edits ──apply_edits──▶ category per target WORD ──category_weights──▶ weight per WORD
                                                                              │
     target text ──tokenizer(offsets)──▶ (s, e) per TOKEN                     │
                                              │                               │
                         char2word[e-1] ──────┴──▶ word index ────────────────┘
                                                        ▼
                                              weight per TOKEN (label_weights)
```

## 1. Why the model's output never needs a category

The loss is

```
L = Σ_t w_t · CE(p_t, y_t) / Σ_t w_t
```

where `y_t` is the gold token at position `t` and `p_t` is the model's predicted
distribution there. The weight `w_t` depends only on the gold position. Suppose the
gold word is `saw` (a `VERB:TENSE` correction) and the model prefers `sees`. The loss
`−log p(saw)` is scaled by the `VERB:TENSE` weight, and we never ask what kind of
error `sees` is.

During training nothing is generated one token at a time. `DataCollatorForSeq2Seq`
builds `decoder_input_ids` by shifting the **gold** labels right by one, and the whole
sequence goes through a single forward pass with a causal mask:

```
labels            :  <s>   Ġa    Ġbig  Ġdis   aster  Ġ.    </s>
decoder_input_ids :  </s>  <s>   Ġa    Ġbig   Ġdis   aster Ġ.
```

`logits[:, t]` is scored against `labels[:, t]` (`compute_loss`, line 257). The
tensors `ce`, `labels` and `label_weights` all have shape `[batch, T]`, so they line
up by construction. A wrong prediction at `t` does not shift later positions, because
the next decoder input is still the gold token.

**Length mismatches.** In training the output is always exactly `T` positions long.
A model that wants a longer output puts low probability on `</s>` at the final
position. A model that wants to insert an extra word puts probability on a non-gold
token at some position. Both are penalised at a known gold position, so the weight is
known too. At inference (`eval-conll14st.py`) outputs can be any length, but no weights
are used there: M2Scorer counts extra edits as false positives.

## 2. Gold target words → categories (`read_m2`, `apply_edits`)

The BEA-2019 M2 files list every edit with its span in the source, its ERRANT type
and its correction:

```
S I saws the show 's advertisement hanging up of a wall ...
A 1 2|||R:VERB:TENSE|||saw|||REQUIRED|||-NONE-|||0
A 8 9|||R:PREP|||on|||REQUIRED|||-NONE-|||0
```

`read_m2` (line 116) parses the edits for annotator 0, skips `noop`, and keeps the
coarse category: `coarse_category` (line 111) drops the operation prefix, so
`R:VERB:TENSE` becomes `VERB:TENSE`. Edits are stable-sorted by start offset.

`apply_edits` (line 142) walks through the source, copies unedited words and applies
each edit. It returns the target words, a parallel list of categories, and a category
list for EOS:

| Target word   | Origin                 | Category        |
|---------------|------------------------|-----------------|
| I             | copied                 | `None` (w = 1)  |
| **saw**       | replaces `saws`        | `VERB:TENSE`    |
| the … up      | copied                 | `None` (w = 1)  |
| **on**        | replaces `of`          | `PREP`          |
| a wall …      | copied                 | `None` (w = 1)  |

Rules by edit type:

- **Replacement / insertion (`R:`, `M:`)**: every word of the correction gets the
  category. For example, `because` → `because of` gives both words `PREP`.
- **Deletion (`U:`)**: a deleted word produces no target word. Its category is held in
  `pending` and attached to the **next** target word, which is the step where the model
  must skip the deleted word instead of copying it. For example, in
  `I like the music` → `I like music`, `music` gets `DET`. If nothing follows the
  deletion, the category becomes `eos_cat` and goes on `</s>` (see §5).
- **Multiple categories on one word**: a word can carry a pending deletion and also
  come from an edit. It then keeps a list of categories, and `word_weights` (line 228)
  uses the **maximum** weight.
- **`UNK` edits**: the source span is left unchanged and copied, so it gets weight 1.

### Safety check against the baseline data

`load_training_data` (line 176) rebuilds every training pair from M2 and compares it
with the baseline's `*.src` / `*.tgt` files used by `standard-finetuning`. Sources must
match exactly (assert). If a rebuilt target differs (line 194), the baseline target is
used and **all its weights are set to 1**, because the categories can't be placed
reliably. These counts are saved as `target_mismatches` in `train_config.json`.

## 3. Categories → weights (`category_weights`, line 211)

`n_c` is the number of edited target positions of category `c`, counting both word
positions and EOS positions. Then:

```
r_c = n_c^(−β)                                  # rarer → larger
w_c = min(1 + (α − 1) · r_c / Σ_c' p_c' r_c',  w_max),   p_c = n_c / Σ n
```

Before the cap, the average weight of an edited token is `α`. With `β = 0` every edit
gets `α`, and with `α = 1` you get the standard loss back. Copied tokens always have
weight 1.

## 4. Words → BPE tokens (`tokenize` inside `main`, line 301)

BART uses byte-level BPE, so one word can become several tokens, and the loss is
computed per token. The mapping uses character offsets.

### 4.1 Join the words with single spaces (line 290)

```python
"tgt": [" ".join(t) for t in targets],
```

`targets` holds one word list per sentence, for example
`["I", "saw", "the", "show", "'s", ...]`. Joining gives the string the tokenizer needs,
which is also exactly the line in the baseline `.tgt` file. The single-space join can
be reversed exactly: `text.split(" ")` gives back the same word list, including the
empty words some Lang-8 lines contain. So word `k` of the string is word `k` of
`word_weights`.

### 4.2 Character → word index

```python
char2word = []
for k, word in enumerate(text.split(" ")):
    char2word += [k] * len(word) + [-1]
```

```
text:      a  ␣  b  i  g  ␣  d  i  s  a  s  t  e  r  ␣  .
char2word: 0 -1  1  1  1 -1  2  2  2  2  2  2  2  2 -1  3   (-1)
```

### 4.3 Token → word through its last character

The tokenizer is called with `return_offsets_mapping=True`, which gives a `(s, e)`
character span for every token:

```python
lw = [ww[char2word[e - 1]] if e > s and char2word[e - 1] >= 0 else 1.0
      for s, e in offsets]
```

| token   | (s, e)          | word         | weight          |
|---------|-----------------|--------------|-----------------|
| `<s>`   | (0, 0)          | — (special)  | 1.0             |
| `Ġa`    | (0, 1)          | 0 `a`        | 1.0             |
| `Ġbig`  | (2, 5) / (1, 5) | 1 `big`      | 1.0             |
| `Ġdis`  | (6, 9) / (5, 9) | 2 `disaster` | **w(MORPH)**    |
| `aster` | (9, 14)         | 2 `disaster` | **w(MORPH)**    |
| `Ġ.`    | (15, 16)        | 3 `.`        | 1.0             |
| `</s>`  | (0, 0)          | — (special)  | 1.0 or EOS (§5) |

(The token splits above are illustrative.)

- **Last character (`e − 1`), not the first:** depending on `trim_offsets`, a span may
  include the BPE leading space (`Ġ`). Then `char2word[s]` would be `-1`, but the last
  character is always inside the word.
- **`e > s`:** special tokens have empty spans and get 1.0.
- **Every subword of an edited word gets the full word weight.** Each piece is a step
  the model must get right to produce the corrected word.
- A token never spans two words, because byte-level BPE splits on spaces before
  merging.

### 4.4 Contractions

M2 data is PTB/spaCy-tokenized, so contractions are already separate words:
`did n't`, `ca n't`, `wo n't`, `sha n't`, `it 's`, `I 'll`. `n't` and `'s` are
ordinary words with their own index and category (often `CONTR`). For example:

```
S ... does n't mean it 's will be easy ...
A 8 9|||U:CONTR|||||| ...     → "'s" is deleted; CONTR goes on "will"
```

If BPE splits ` n't` into several pieces, all of them map to the `n't` word. The
CoNLL-14 test set uses the same tokenization, so training and evaluation match. Raw
text containing `didn't` should be tokenized the same way before inference.

## 5. Deletions at the end of the sentence → `</s>`

```python
lw[-1] = max(lw[-1], eos_w)   # line 315
```

If the last source word(s) were deleted, `apply_edits` returns their categories as
`eos_cat`. The step where the model must stop instead of copying is `</s>`, which is
always the last label token, even after truncation. `max` ensures the weight is never
lowered.

## 6. Batching and the loss

- `WeightedSeq2SeqCollator` (line 237) pads `label_weights` with **0** to the label
  length. Labels are padded with `-100`.
- `compute_loss` (line 257):

  ```python
  ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                       ignore_index=-100, reduction="none").view_as(labels)
  w  = weights * (labels != -100)
  loss = (w * ce).sum() / w.sum().clamp_min(1.0)
  ```

  Padding contributes nothing to the numerator or to the denominator. Dividing by the
  total weight, not the token count, keeps the gradient scale comparable to the
  baseline for every `α`.

## 7. Checking the alignment

At startup the script prints three weighted training examples as
`token` / `[token|weight]` (the sanity-check block at line 327). This is the quickest
way to confirm on real data that corrected words, and the word after a deletion, carry
the expected weights.
