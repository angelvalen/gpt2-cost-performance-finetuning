# A Cost–Performance Comparison of Fine-Tuning Methods Across GPT-2 Scales and Tasks

Code and technical report for a controlled study of how **model size** and **fine-tuning method** jointly determine **performance and cost** in GPT-2. Self-directed project that follows Stanford's CS224N Default Final Project (GPT-2), using its public starter code, datasets and report structure.

**Report:** [A Cost–Performance Comparison of Fine-Tuning Methods Across GPT-2 Scales and Tasks](report.pdf)

## Overview

We compare three fine-tuning modes on GPT-2 small (125M), medium (356M) and large (774M):

| Mode | What is trained |
|---|---|
| **LLL** (last linear layer) | Only the classification head; the backbone is frozen |
| **LoRA** | Low-rank adapters on the attention projections (`r=8`, `alpha=16`); backbone frozen |
| **Full** | All parameters (not run at large scale: it did not fit on our GPU) |

on four tasks: sentiment classification (**SST-5**, **CFIMDB**), paraphrase detection (**Quora**) and **sonnet generation** (where LLL does not apply). Every configuration is repeated over 3 seeds under one shared protocol (one learning rate per mode across sizes), and we report accuracy (chrF for sonnets) together with trainable parameters, peak GPU memory and time per epoch.

## Main findings

- **Fine-tuning method.** LoRA outperforms LLL by 4–13 points on every classification task and size, and matches full fine-tuning within noise while training under 0.5% of the parameters.

- **Size.** On the choice of size, the gains from scale are task dependent: LoRA saturates at small scale on CFIMDB and at medium scale on SST and sonnet, whereas paraphrase still improves at large scale, with diminishing returns.

- **Cost.** The parameter saving brings smaller savings in peak memory and time at equal micro-batch, and cost grows roughly in proportion to model size. On a 15 GB GPU this growth becomes a feasibility limit: full fine-tuning of GPT-2 large did not fit even at micro-batch 1 on CFIMDB, whereas LoRA large ran on all four tasks.

- **Choice.** LoRA at small-to-medium scale is the best cost–accuracy default in our setup. LLL, whose memory is 4–6× below full fine-tuning, is the cheapest option, at the price of 4–13 points on classification. The best choice ultimately depends on the task and the hardware.

### Dev-set results (mean ± std over 3 seeds)

Accuracy (%) for SST, CFIMDB and paraphrase; chrF for sonnet (top-p sampling).

| Mode (LR) | Size | SST | CFIMDB | Paraphrase | Sonnet |
|---|---|---|---|---|---|
| LLL (1e-3) | small | 47.4 ± 0.5 | 86.0 ± 2.5 | 71.0 ± 1.5 | -- |
| | medium | 49.1 ± 0.6 | 86.3 ± 0.2 | 74.5 ± 0.3 | -- |
| | large | 50.4 ± 0.3 | 90.6 ± 0.0 | 76.5 ± 0.2 | -- |
| LoRA (1e-4) | small | 51.6 ± 0.1 | 97.7 ± 0.6 | 84.4 ± 0.3 | 26.3 ± 0.5 |
| | medium | 55.2 ± 0.5 | 97.6 ± 0.4 | 86.3 ± 0.3 | 26.9 ± 0.2 |
| | large | 54.9 ± 0.5 | 97.3 ± 0.6 | 87.2 ± 0.1 | 27.0 ± 0.4 |
| Full (1e-5) | small | 50.1 ± 0.9 | 98.0 ± 0.7 | 84.5 ± 0.1 | 26.3 ± 0.5 |
| | medium | 54.8 ± 1.1 | 97.7 ± 0.5 | 85.8 ± 0.4 | 27.0 ± 0.2 |
| | large | -- | -- | -- | -- |

Cost tables (trainable parameters, peak memory, time per epoch) and learning-rate / regularization sensitivity runs are in the [report](report.pdf).

## Repository structure

```
.
├── report.pdf                # Main technical report
├── train.ipynb               # Example notebook: every task and mode for one model size and seed
├── explore_results.ipynb     # Exploration and aggregation of the results
├── classifier.py             # SST-5 / CFIMDB classification
├── paraphrase_detection.py   # Quora paraphrase detection
├── sonnet_generation.py      # Sonnet generation: top-p sampling and beam search
├── models/                   # GPT-2 model
├── modules/                  # Masked multi-head self-attention, transformer layer
├── optimizer.py              # AdamW
├── datasets.py               # Dataset loading
├── evaluation.py             # Evaluation utilities
├── visualization.py          # Plotting
├── utils.py, config.py       # Shared utilities and configuration
├── sanity_check.py           # Starter-code check of the GPT-2 implementation against Hugging Face
├── optimizer_test.py         # Starter-code test of the optimizer (with optimizer_test.npy)
├── data/                     # Datasets
├── results/                  # Per-run results summaries
├── output/                   # Dev and test predictions / generated sonnets, training curves (git-ignored)
├── checkpoints/              # Saved model checkpoints (git-ignored)
├── requirements.txt          # Python dependencies
├── README.md
└── LICENSE
```

## Setup

Experiments ran on a single **Tesla T4 (15 GB)** on Kaggle, so absolute memory and time values are specific to that hardware.

**Use Python 3.12.**

```bash
git clone https://github.com/angelvalen/gpt2-cost-performance-finetuning.git
cd gpt2-cost-performance-finetuning
pip install -r requirements.txt
```

LoRA uses the Hugging Face `peft` library; pretrained GPT-2 weights are downloaded from Hugging Face.

## Data

Datasets come from the public CS224N Default Final Project starter repository: SST-5, CFIMDB, Quora Question Pairs (we use a subsample) and Shakespeare's sonnets. Training and development data is included in `data/`.

## Usage

### Run the full grid for one model size and seed

[`train.ipynb`](train.ipynb) runs every task and fine-tuning mode for a fixed model size and seed. Set the model size (`gpt2`, `gpt2-medium` or `gpt2-large`) and the seed in the cells. Per-run results are written to `results/`, training curves and predictions (or generated sonnets) on the development and test sets are written to `output/`. 

### Run a single experiment

Each task has its own script; run `--help` for the full list of options.

```bash
python classifier.py --help               # SST-5 and CFIMDB
python paraphrase_detection.py --help     # Quora
python sonnet_generation.py --help        # sonnets
```

### Hyperparameters

| Parameter | Scope | Value |
|---|---|---|
| Weight decay | all | 0.01 |
| Head dropout | all | 0.1 |
| LoRA r, alpha, dropout | all | 8, 16, 0.05 |
| LoRA target modules, bias | all | query, key, value, attention_dense; none |
| Learning rate | mode | LLL 1e-3; LoRA 1e-4; Full 1e-5 |
| Effective batch size | task | SST 32, CFIMDB 8, Para. 32, Sonnet 4 |
| Early-stopping patience | task | Classification 5, Sonnet 10 |
| Top-p sampling | task | p = 0.9, temperature = 0.9 |
| Beam search | task | 5 beams, length penalty 0.6 |

### Reproducing the report

The full grid (4 tasks x 3 modes x 3 sizes x 3 seeds) does not fit in one Kaggle session, so the report's results were produced by running `train.ipynb` separately for each model size on a Kaggle Tesla T4 (15 GB), with three seeds (11711, 1, 2). The per-run results are in `results/`, and `explore_results.ipynb` helps visualising them.

#### Micro-batch sizes on a 15 GB GPU

The effective batch size is fixed per task (SST 32, CFIMDB 8, paraphrase 32, sonnet 4). Most configurations run with a micro-batch equal to the effective batch, but five did not fit in the T4's 15 GB at that size. For these we used the **largest power-of-two micro-batch that fit** and gradient accumulation, so that micro-batch x accumulation steps equals the effective batch:

| Mode | Size | Task | Micro-batch x accumulation steps |
|---|---|---|---|
| LoRA | medium | Paraphrase | 16 x 2 |
| LoRA | large | CFIMDB | 4 x 2 |
| LoRA | large | Paraphrase | 4 x 8 |
| Full | medium | CFIMDB | 4 x 2 |
| Full | medium | Paraphrase | 8 x 4 |

If you reproduce the results on the same GPU, set these in the notebook's configuration (in the code, `batch_size` is the micro-batch and `grad_accum_steps` the accumulation steps); otherwise those runs will run out of memory. Full fine-tuning of GPT-2 large was not run, since it did not fit even at micro-batch 1 on CFIMDB.

Accumulation leaves the optimization settings unchanged, so accuracy is unaffected, but **peak memory and time per epoch depend on the micro-batch**. These five cells are therefore not comparable with the rest in the report's cost analysis, and on a GPU with more memory you can use the effective batch size directly.

## Implementation notes

- **Built on the public CS224N starter code.** The starter provides the scaffolding; the GPT-2 components (attention, transformer layers, embeddings), the AdamW step and the training and evaluation pipelines are our own implementation. LoRA uses the `peft` library.
- **Efficiency techniques:** FP16 mixed precision (autocast with loss scaling, fp32 weights), KV cache for generation, and gradient accumulation.
- **Early stopping** is decided on dev-set performance, so absolute numbers are slightly optimistic.
- **Cost measurement:** peak memory is `torch.cuda.max_memory_allocated`; time per epoch is wall-clock and includes the per-epoch evaluation used for early stopping.

## Limitations

- One learning rate per mode across sizes, a fixed LoRA configuration, three seeds, dev-set results only.
- No full fine-tuning at large scale.
- Five configurations needed a reduced micro-batch (the largest power of two that fit) plus gradient accumulation, so their memory and time are not comparable with the rest; quantitative cost comparisons use the other cells.

## Citation

```bibtex
@misc{valencia2026costperformance,
  title        = {A Cost--Performance Comparison of Fine-Tuning Methods Across {GPT-2} Scales and Tasks},
  author       = {Valencia Padilla, {\'A}ngel},
  year         = {2026},
  howpublished = {\url{https://github.com/angelvalen/gpt2-cost-performance-finetuning}},
  note         = {Technical report}
}
```

## Acknowledgements

Stanford CS224N staff for the public project handout, starter code and datasets; Hugging Face for the pretrained GPT-2 weights, and the `peft` library for LoRA.

## License
Released under the [Apache License 2.0](LICENSE). This repository contains code derived from the [CS224N Default Final Project starter code](https://github.com/stanfordnlp/cs224n_gpt) (Apache License 2.0). The starter files were modified by Ángel Valencia, 2026: the missing GPT-2 components and the AdamW step were implemented, and LoRA, gradient accumulation, mixed-precision training, KV-cache decoding, beam search and the experiment pipeline were added. Datasets and pretrained weights are not covered by this license and are subject to their own terms.

## Author

Ángel Valencia Padilla · [GitHub](https://github.com/angelvalen) · [LinkedIn](https://linkedin.com/in/angelvalenciapadilla) · angelvalenciabusiness@gmail.com
