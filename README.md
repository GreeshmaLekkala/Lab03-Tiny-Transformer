# Tiny Transformer — Causal Attention, Masked Modeling & LoRA

## Project Overview

This project explores the fundamentals of Transformer-based language models using PyTorch. It implements scaled dot-product attention, causal masking, next-token prediction, masked language modeling, and Low-Rank Adaptation (LoRA).

The experiments use synthetic sequences generated from a small finite grammar to study model training, adaptation, and attention behavior.

## Key Implementations

- Scaled dot-product attention with causal masking
- Causal next-token cross-entropy loss
- Masked-position cross-entropy loss
- Rank-2 LoRA adapter with frozen base parameters
- Bidirectional attention and right-context sensitivity testing

## Experimental Results

| Metric | Result |
|---|---:|
| Initial causal development loss | 2.5913 |
| Causal test loss after training | 0.4654 |
| LoRA trainable parameters | 72 |
| Adaptation loss before LoRA | 2.4281 |
| Adaptation loss after LoRA | 0.5069 |
| Masked test loss | 0.0000 |
| Right-context maximum logit difference | 0.2375 |

The experiments demonstrated a reduction in causal prediction loss and successful adaptation using a small number of trainable LoRA parameters.

The right-context perturbation test showed that changing a future token affected the masked position's output, supporting the expected behavior of bidirectional attention.

## Technologies Used

- Python
- PyTorch
- NumPy
- SciPy
- Scikit-learn
- Google Colab

## How to Run

1. Download the `.ipynb` notebook from this repository.
2. Open it in Google Colab.
3. Run the environment setup cells.
4. Restart the session if prompted.
5. Select **Runtime → Run all**.
6. Review the experiment outputs and analysis.

## Limitations

This project uses a small synthetic grammar rather than natural-language datasets. The results demonstrate distribution fitting and attention behavior but do not establish general language understanding or compositional generalization.

## Academic Context

Completed as part of COMP-672 Natural Language Processing coursework. The lab structure and starter materials were provided by the course instructor.

Original course repository: https://github.com/glossner/COMP-672-Labs
