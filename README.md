# Image Captioning & MLP Regularization Experiments (PyTorch)

This repository contains two deep learning and computer vision experimental notebooks implemented in PyTorch (`A_G13`):

1. **Task 1 — Image Captioning using CNN-LSTM & Bahdanau Attention on Flickr8k** ([`A_G13_Task1_Notebook.ipynb`](./A_G13_Task1_Notebook.ipynb))
2. **Task 2 — Overfitting & Regularization Analysis in MLPs on Fashion-MNIST** ([`A_G13_Task2_Notebook.ipynb`](./A_G13_Task2_Notebook.ipynb))

---

## Repository Structure

```text
.
├── A_G13_Task1_Notebook.ipynb   # CNN-LSTM & Attention image captioning on Flickr8k
├── A_G13_Task2_Notebook.ipynb   # MLP overfitting & regularization study on Fashion-MNIST
├── .gitignore
└── README.md
```

---

## Task 1: Image Captioning using CNN-LSTM & Bahdanau Attention (`Flickr8k`)

### Objective
Automatically generate natural language descriptions for images using an Encoder–Decoder architecture in PyTorch, progressing from fixed-feature CNN-LSTM baselines (V0) to a spatial Bahdanau Attention model (V1).

### Dataset & Preprocessing
- **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (loaded via `kagglehub`)
  - **8,091** unique images, **5** captions per image (**40,455** total records)
  - Caption length: mean **11.78** words (min 1, max 38)
- **Splits** (`random_state=42`):
  - Train: **32,364** | Validation: **4,045** | Test: **4,046**
- **Vocabulary:** Words occurring $\ge 5$ times in training captions plus `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` (**2,676 tokens**)
- **Image Transforms:** Resize to `224 × 224`, `ToTensor()`, and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

### Architectures & Experimental Results
- **V0 Baselines** (`EncoderCNN` frozen ResNet + `DecoderLSTM`, embed dim `256`, hidden size `512`, batch size `32`, `10` epochs, Adam `lr=0.001`, evaluated on 500 test samples):
  - `Exp 1`: ResNet18 + 1-Layer LSTM
  - `Exp 2`: ResNet50 + 1-Layer LSTM
  - `Exp 3`: ResNet18 + 2-Layer LSTM (`dropout=0.3`)
- **V1 Attention Model** (`EncoderCNN` spatial feature maps `[B, 2048, 7, 7]` + `BahdanauAttention` + `AttentionDecoder` LSTMCell, embed/hidden/attention dim `512`, batch size `64`, `15` epochs, Adam `lr=1e-3`, multi-reference BLEU on 1,000 test images):
  - `Exp 4`: ResNet50 + Bahdanau Attention (weights saved to `exp4_resnet50_attention.pth`)

| Experiment | Architecture | Final Train Loss | Final Val Loss | BLEU Score |
| --- | --- | :---: | :---: | :---: |
| Exp 1 (V0) | ResNet18 + 1-Layer LSTM | 1.2995 | 2.5800 | 0.047484 |
| Exp 2 (V0) | ResNet50 + 1-Layer LSTM | 1.1236 | 2.4810 | 0.071494 |
| Exp 3 (V0) | ResNet18 + 2-Layer LSTM | 1.7387 | 2.3979 | 0.046436 |
| **Exp 4 (V1)** | **ResNet50 + Bahdanau Attention** | **0.3874** | **3.1217** | **0.251987** |

- **Key Takeaway:** Introducing spatial feature maps and Bahdanau Attention improved the BLEU score by **252.46%** over the best V0 baseline (`0.071494` → `0.251987`).

---

## Task 2: Overfitting & Regularization Analysis in MLPs (`Fashion-MNIST`)

### Objective
Study overfitting in a Multi-Layer Perceptron (MLP) by intentionally restricting training data, and evaluate how **Dropout**, **Weight Decay (L2 Regularization)**, and their combination affect generalization and the train-validation accuracy gap.

### Dataset & Setup
- **Dataset:** Fashion-MNIST (`torchvision.datasets.FashionMNIST`) — `28 × 28` grayscale images across 10 classes
  - **Training subset:** Intentionally limited to **1,000** samples (`Subset(train_dataset, range(1000))`) to induce overfitting
  - **Evaluation set:** Full **10,000** test images
- **Base Architecture:** `784 → 512 (ReLU) → 256 (ReLU) → 10`
- **Training Config:** `30` epochs, batch size `64`, `CrossEntropyLoss`, Adam optimizer (`lr=0.001`)

### Experiments & Results (Epoch 30)

| Model | Regularization Configuration | Train Accuracy | Validation Accuracy | Train–Val Gap |
| --- | --- | :---: | :---: | :---: |
| `BaselineMLP` | None | 97.90% | 80.42% | 17.48% |
| `DropoutMLP` | `Dropout(0.5)` after each hidden layer | 92.00% | 80.21% | 11.79% *(11.59% in summary)* |
| `WeightDecayMLP` | `weight_decay=1e-4` (Adam) | 97.80% | 79.86% | 17.94% |
| **`CombinedMLP`** | **`Dropout(0.5)` + `weight_decay=1e-4`** | **89.00%** | **79.52%** | **9.48%** |

- **Key Findings:**
  - Without regularization, the baseline MLP heavily memorized the 1,000-sample training set (`97.90%` train vs. `80.42%` validation).
  - `weight_decay=1e-4` alone had minimal effect on reducing the generalization gap (`17.94%`).
  - `Dropout(0.5)` substantially reduced overfitting (gap reduced by ~5.89 percentage points).
  - Combining **Dropout + Weight Decay** achieved the smallest train–validation gap (**9.48%**, an **8.00%** gap reduction compared to the executed baseline).

---

## Requirements & Usage

### Dependencies
- Python 3.x
- `torch`, `torchvision`
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- `Pillow`, `tqdm`, `nltk`, `kagglehub` (for Task 1)

### Running the Notebooks
1. Clone the repository:
   ```bash
   git clone https://github.com/JATINCHULETT/Image-Captioning-and-MLP-Regularization.git
   cd Image-Captioning-and-MLP-Regularization
   ```
2. Open either notebook in JupyterLab, VS Code, or Google Colab:
   - `A_G13_Task1_Notebook.ipynb` (GPU recommended; downloads Flickr8k via `kagglehub`)
   - `A_G13_Task2_Notebook.ipynb` (runs on CPU or GPU; downloads Fashion-MNIST automatically via `torchvision`)
