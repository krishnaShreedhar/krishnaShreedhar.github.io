---
title: "Paper Review: Attention Is All You Need"
subtitle: >
  Vaswani et al., NeurIPS 2017. Core contribution, what I understood quickly,
  what took longer, and what to read next.
category: growth
type: paper-review
date: 2024-01-01
status: done   # done | reading | queued | notes
tags:
  - paper-review
  - transformers
  - nlp
reading_time: 7
author: "Shreedhar Kodate"
output: "blogs/growth/posts/attention-paper-review.html"
# Paper metadata
paper:
  authors: "Vaswani, Shazeer, Parmar, Uszkoreit et al."
  venue: "NeurIPS 2017"
  url: "https://arxiv.org/abs/1706.03762"
  rating: 5
---

## Core Contribution

The Transformer is the first sequence transduction model built entirely on attention mechanisms,
without any recurrence or convolution. This removes the sequential dependency that prevented
RNNs from being parallelised, and establishes an architecture family that now underlies almost
all large language models.

## What I Understood Quickly

- The motivation against RNNs: sequential computation, long-range dependency path length.
- The mechanics of scaled dot-product attention — clear once you see the Q/K/V framing.
- Multi-head attention as parallel attention sub-spaces.
- The encoder-decoder structure and why causal masking is needed in the decoder.

## What Confused Me (and How I Resolved It)

### Why divide by √d_k?

For random unit-variance Q and K vectors of dimension d_k, the dot product has variance d_k.
Dividing by √d_k restores unit variance regardless of dimension.

### Why sine/cosine for positional encoding?

sin(pos + k) can be written as a linear combination of sin(pos) and cos(pos), so the model
can learn to attend to "offset by k" positions via linear projection.

## What to Read Next

| Paper | Relationship | Priority |
|-------|-------------|----------|
| BERT (Devlin et al., 2018) | Encoder-only, bidirectional pre-training | High |
| GPT (Radford et al., 2018) | Decoder-only, autoregressive | High |
| Relative Attention (Shaw et al., 2018) | Relative positions | Medium |
| ViT (Dosovitskiy et al., 2020) | Transformers for images | High |

## References

[^1]: Vaswani et al. (2017). *Attention is all you need.* NeurIPS.
      https://arxiv.org/abs/1706.03762

[^2]: Rush, A. (2018). *The Annotated Transformer.* Harvard NLP.
