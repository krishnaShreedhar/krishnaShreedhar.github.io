---
title: "Fine-Tuning LLMs"
subtitle: "Fine-tuning adapts a pre-trained language model to a specific task or domain by continuing training on curated task-specific data. Unlike prompting which works within the frozen model's capabilities, fine-tuning..."
category: technical
project: large_scale_aiml_systems
project_title: "Large Scale AI/ML Systems"
date: 2025-08-07
reading_time: 5
tags:
  - large-scale-aiml-systems
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_aiml_systems/docs/03_llmops/04_fine_tuning.html"
---
Fine-tuning adapts a pre-trained language model to a specific task or domain by continuing training on curated task-specific data. Unlike prompting which works within the frozen model's capabilities, fine-tuning modifies model weights to internalize new behavior, output formats, or domain knowledge. The decision of when to fine-tune vs use RAG vs prompt engineer is one of the most important architectural decisions in LLM system design.

## Fine-Tuning Approaches

```mermaid
graph TD
    subgraph Methods[Fine-Tuning Method Spectrum]
        subgraph FullFT[Full Fine-Tuning]
            Full[Update all model weights\nMaximum adaptation\nRequires large GPU cluster\n70B model: 8x A100 80GB minimum\nRisk of catastrophic forgetting]
            style Full fill:#fee2e2,stroke:#dc2626
        end

        subgraph PEFT[Parameter-Efficient Fine-Tuning]
            LoRA[LoRA - Low-Rank Adaptation\nFreeze base model\nAdd small trainable rank matrices\nto attention layers\nTypically rank r=8 to 64\nTrains 0.1-1% of parameters]
            QLoRA[QLoRA - Quantized LoRA\nQuantize base model to 4-bit\nApply LoRA on quantized model\n70B model fine-tunable on 2x A100\nNear-full-FT quality at fraction of cost]
            style LoRA fill:#dcfce7,stroke:#16a34a,stroke-width:2px
            style QLoRA fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        end

        subgraph Alignment[Alignment Techniques]
            SFT[Supervised Fine-Tuning\nInstruction-response pairs\nTeaches desired behavior and format]
            RLHF[RLHF - Reinforcement Learning\nfrom Human Feedback\nTrain reward model on human preferences\noptimize policy with PPO\nExpensive and complex]
            DPO[DPO - Direct Preference Optimization\nSimpler alternative to RLHF\nTrain on preference pairs chosen vs rejected\nNo separate reward model]
        end
    end
```

## LoRA Architecture

```mermaid
graph TD
    subgraph LoRADiagram[LoRA Parameter Injection]
        Input[Input Activations\ndim: d]

        subgraph FrozenPath[Frozen Base Model Path]
            W[Original Weight Matrix W\ndim: d x d\nFROZEN - no gradient updates]
        end

        subgraph LoRAPath[LoRA Path - Trainable]
            A[Matrix A\ndim: d x r\nrandom init\nrank r much less than d]
            B[Matrix B\ndim: r x d\nzero init\nso delta-W starts at zero]
            AB[A times B\nLow-rank update\ndelta-W of rank r]
        end

        Output[Output = W times x plus alpha divided by r times A times B times x\nalpha: scaling hyperparameter]

        Input --> W --> Output
        Input --> A --> B --> AB --> Output
    end
```

## Fine-Tuning Pipeline

```mermaid
graph TD
    subgraph FTPipeline[Fine-Tuning Pipeline]
        Decision[Decision: Fine-Tune vs RAG\nFine-tune if: format/style not achievable by prompting\nRAG if: knowledge base is dynamic or large]

        DataCuration[Dataset Curation\n1000-100000 instruction-response pairs\nHigh quality over quantity\nDiverse coverage of task variants\nValidation split 10-20%]

        BaseModel[Select Base Model\nLlama 3.1 8B or 70B\nMistral 7B\nPhi-3 medium\nbased on capability and cost]

        Train[QLoRA Training\nhyperparams: lr=2e-4\nbatch=4-16 grad accumulation\nepochs=1-3\nuse bf16 precision]

        Eval[Evaluate on Held-Out Set\ncompare to baseline - base model with prompt\ncheck for regression on general capabilities]

        Merge[Merge LoRA Adapters\ninto base model weights\nfor efficient serving\nor serve adapter separately]

        Deploy[Deploy Fine-Tuned Model\nto inference infrastructure]

        Decision --> DataCuration --> BaseModel --> Train --> Eval --> Merge --> Deploy
    end

    style DataCuration fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style Deploy fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **LoRA (Low-Rank Adaptation)**: Instead of modifying the full weight matrices (which are huge — a 7B model has billions of parameters), LoRA adds small trainable rank decomposition matrices alongside frozen original weights. The effective weight update W + delta-W where delta-W = AB is constrained to rank r. This makes fine-tuning feasible on modest hardware and prevents catastrophic forgetting of the base model's general capabilities.

- **QLoRA**: Combines 4-bit quantization of the base model with LoRA adapters. The base model is quantized to 4-bit (NF4 format) and kept frozen. LoRA adapters are trained in bf16. This reduces the GPU memory required to fine-tune a 70B model from 8+ A100s (full FT) to 2 A100s — a 4x memory reduction. Accuracy is near-identical to full fine-tuning in most benchmarks.

- **Instruction Tuning**: Fine-tuning on (instruction, response) pairs to teach a model to follow instructions reliably. The base pre-trained model knows a lot but doesn't know to behave helpfully. Instruction tuning (SFT on curated instruction-following datasets) produces the difference between a raw language model and a useful assistant.

- **RLHF (Reinforcement Learning from Human Feedback)**: The technique used by OpenAI, Anthropic, and others to align LLMs with human preferences. Step 1: Collect human preferences (humans rank multiple model outputs). Step 2: Train a reward model on these preferences. Step 3: Fine-tune the LLM using PPO (Proximal Policy Optimization) to maximize the reward model's score. RLHF is expensive and complex but produces highly aligned models.

- **DPO (Direct Preference Optimization)**: A simpler alignment technique that directly optimizes on preference pairs (chosen vs rejected response) without a separate reward model or RL training. DPO reformulates the RLHF objective into a classification loss on preference pairs, making it much more stable and computationally efficient. Most teams doing alignment today use DPO or its variants.

- **Data Quality over Quantity**: For fine-tuning, 1,000 high-quality, diverse, and correctly formatted instruction-response pairs typically outperforms 100,000 noisy or redundant pairs. Dataset curation — filtering, deduplicating, and quality-checking examples — is more important than collecting more data.

- **Catastrophic Forgetting**: When fine-tuning on a narrow dataset causes the model to lose its general capabilities (common knowledge, language fluency, instruction-following). Mitigations: LoRA/PEFT (modifies only a fraction of parameters), mixing in general instruction-following data during fine-tuning, and evaluating on general benchmarks alongside task-specific metrics.

## Trade-offs

| Approach | Cost | Flexibility | Capability Gain | Serving Complexity |
|---------|------|------------|----------------|-------------------|
| Prompt engineering | Very Low | High | Limited | Low |
| RAG | Low | High | Knowledge extension | Medium |
| LoRA fine-tuning | Medium | Medium | Style and format | Medium |
| Full fine-tuning | Very High | Low | Maximum | High |
| RLHF | Highest | Low | Alignment | High |

## When to Use

- **Prompt engineering first**: Always the starting point — exhausts the free option before paying for fine-tuning
- **RAG**: When the model needs access to specific up-to-date or proprietary knowledge — faster iteration than fine-tuning and knowledge can be updated without retraining
- **LoRA fine-tuning**: When the required output format or behavior cannot be achieved by prompting (JSON schemas with specific conventions, domain-specific writing style, coding in obscure languages), or when inference cost at scale justifies replacing large API calls with a smaller fine-tuned model
- **Full fine-tuning**: Rarely justified for most teams — only when PEFT doesn't achieve required quality and compute budget allows
- **DPO over RLHF**: Default for alignment when preference data is available — same quality as RLHF with significantly less engineering complexity