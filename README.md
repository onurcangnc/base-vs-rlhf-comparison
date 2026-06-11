# Base vs RLHF Model Comparison

A hands-on notebook comparing **base** (pretrained) and **instruct** (RLHF-tuned) language models to observe how reinforcement learning from human feedback transforms model behavior.

## What This Does

Loads Gemma 2B in both variants and runs identical prompts through each:

| Category | Purpose |
|---|---|
| Factual Q&A | Can it answer a direct question? |
| Instruction Following | Can it follow a specific task format? |
| Conversational | Does it engage in dialogue? |
| Safety-Relevant | How does it handle sensitive requests? |

Includes a **token probability analysis** showing how RLHF shifts the model's next-token distribution — the base model hedges across many continuations while the instruct model concentrates probability on helpful responses.

## Quick Start

1. Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onurcangnc/base-vs-rlhf-comparison/blob/main/base_vs_rlhf_comparison.ipynb)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Accept [Gemma's license](https://huggingface.co/google/gemma-2b) on HuggingFace
4. Run all cells

## Key Takeaways

- **Alignment is a thin layer.** The base model's capabilities are all still there after RLHF — only the behavioral surface changes.
- **Refusal is learned, not inherent.** Base models have zero concept of refusing harmful requests. Safety behavior is entirely an artifact of RLHF.
- **The layer is fragile.** Since safety is a learned overlay, it can be jailbroken, fine-tuned away, or may fail to generalize.

## Context

Created as part of [BlueDot AI Safety Fundamentals](https://course.aisafetyfundamentals.com/) (Alignment track). The exercise is based on the optional "Play with base and RLHF models" activity.

For a deeper mechanistic look at how RLHF installs refusal behavior, see [Arditi et al., 2024 — "Refusal in LLMs Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717).

## License

MIT
