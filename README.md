# Refusal in Language Models Is Mediated by a Single Direction

An implementation notebook reproducing the core method from Arditi et al.'s paper *"Refusal in Language Models Is Mediated by a Single Direction"*, applied to `Qwen/Qwen2.5-Coder-0.5B-Instruct`. The notebook identifies a single linear direction in the model's residual stream that drives refusal behavior, then demonstrates two ways to remove it: activation-level ablation at inference time, and permanent weight orthogonalization.

## What this notebook does

1. **Loads the model** — `Qwen2.5-Coder-0.5B-Instruct` via `TransformerBridge.boot_transformers`, which keeps the raw HuggingFace weights unprocessed (no LayerNorm folding or centering).

2. **Loads paired datasets**
   - *Harmful instructions* — from the [AdvBench](https://github.com/llm-attacks/llm-attacks) `harmful_behaviors.csv` dataset.
   - *Harmless instructions* — from `tatsu-lab/alpaca`, filtered to prompts with no additional input field.
   - Both are split into train/test (80/20).

3. **Finds the refusal direction** — runs the model on a batch of harmful and harmless prompts, caches residual-stream activations at an intermediate layer (`layer=14`, last token position), and computes the difference of means between the two activation sets. This difference vector, normalized to unit length, is the "refusal direction" `r̂`.

4. **Ablates the direction at inference time** — hooks into every `resid_pre`, `resid_mid`, and `resid_post` activation across all layers and projects out the component along `r̂`:
   ```
   a' = a - (a · r̂) r̂
   ```
   Generations are compared side by side: baseline (no intervention) vs. intervention (direction ablated).

5. **Orthogonalizes model weights** — instead of hooking activations at runtime, directly orthogonalizes the weight matrices that write into the residual stream (embedding, attention output, MLP output) with respect to `r̂`:
   ```
   W' = W - r̂ r̂ᵀ W
   ```
   This bakes the ablation permanently into the model, so no forward hooks are needed to reproduce the effect. Generations are then compared across all three conditions: baseline, activation-ablated, and weight-orthogonalized.

## Notebook structure

| Section | Cells | Purpose |
|---|---|---|
| Setup | 1–2 | Install & import dependencies |
| Load model | 3–4 | Boot Qwen2.5-Coder-0.5B-Instruct via TransformerBridge |
| Load data | 5–8 | Fetch and preview harmful/harmless instruction sets |
| Tokenization utils | 9–10 | Chat-template-aware batch tokenizer for Qwen |
| Generation utils | 11–16 | Hookable generation loop (`_generate_with_hooks`, `get_generations`) |
| Find refusal direction | 17–20 | Cache activations, compute mean-difference direction |
| Activation ablation | 21–24 | Hook-based intervention + baseline/intervention comparison |
| Weight orthogonalization | 25–29 | Permanent weight edit + three-way comparison |

## Requirements

Installed in the notebook itself via pip:

```
transformers transformers_stream_generator tiktoken transformer_lens einops jaxtyping colorama
```

Also required (not explicitly installed in the notebook, so add if missing):

```
torch pandas scikit-learn tqdm datasets requests
```

A CUDA-capable GPU is recommended (the notebook uses `bfloat16` and falls back to CPU automatically if no GPU is available, though generation will be much slower).

## How to run

1. Run all cells top to bottom in order — later cells depend on objects (`model`, `refusal_dir`, `harmful_inst_test`, `baseline_generations`, etc.) created earlier.
2. Adjust `N_INST_TRAIN` / `N_INST_TEST` to change how many prompts are used to compute the direction and to evaluate it.
3. Adjust `layer` in the direction-finding cell to compute the refusal direction from a different layer's residual stream.
4. Swap `MODEL_PATH` to point at a different HuggingFace model ID to reproduce the method on another model (chat template and tokenization utilities are Qwen-specific and would need adapting).

## Notes

- This is a research/interpretability reproduction, not a production tool. It's intended for studying *why* and *how* refusal behavior is represented internally, in line with the original paper's mechanistic-interpretability framing.
- Memory is freed after direction extraction (`del harmful_cache, harmless_cache, ...`) to avoid holding onto large cached activation tensors.
- The weight-orthogonalization approach is mathematically equivalent to the activation-ablation hook, but avoids the runtime overhead of hooks on every forward pass.

## Reference

Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., & Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
