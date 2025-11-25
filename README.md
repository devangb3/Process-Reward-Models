# A Process Supervision Approach to Correcting Logical Errors in On-Device LLMs

This repository contains the implementation and experiments for benchmarking Process Reward Models (PRMs) in constrained-compute environments for multi-step mathematical reasoning.

## Overview

Large Language Models (LLMs) frequently struggle with multi-step mathematical reasoning due to **"logical hallucinations"**, where a single intermediate error propagates to the final answer. To mitigate this problem, recent works have demonstrated that **Process Supervision**, where each step in the reasoning chain is verified, provides better guidance and higher accuracy than Outcome Supervision (where only the final answer is verified) and other simple consensus methods like Self-Consistency.

### The Challenge

Using Process Supervision requires training a verifier model (Process Reward Model - PRM) on step-level annotated datasets, which imposes a prohibitively high annotation cost. An open question remains: **Does this upfront investment yield positive returns for small, compute-constrained models (sub-10B parameters), or are cheaper consensus-based methods sufficient?**

### Our Contribution

In this work, we benchmark the **inference sample efficiency of PRMs in constrained-compute environments**. Our key findings demonstrate that investing in step-level annotated datasets is justified even for small models, as it significantly improves accuracy while reducing the number of required generations.

## Key Results

Our experiments on the **GSM8K benchmark** show that the Process-Guided Best-of-N strategy consistently outperforms the Self-Consistency baseline across different constrained compute budgets:

- **At N=5 generations**: PRM achieves 60.0% accuracy vs. Self-Consistency's 50.8% (18.1% improvement)
- **Peak improvement**: 26.8% improvement at N=4 (56.8% vs. 44.8%)
- **Efficiency gain**: PRM at N=5 matches Self-Consistency at N=64, representing a **12.8× reduction** in required generations

> This demonstrates that for constrained environments, the upfront cost of creating step-level annotated datasets yields significantly improved accuracy and computational efficiency.

## Methodology

### Models Used

- **Generator Model**: Qwen/Qwen3-0.6B (600M parameters, lightweight model for generating candidate reasoning chains)
- **Verifier Model**: Qwen/Qwen3-8B (8B parameters) finetuned on PRM800K dataset (Process Reward Model)
- **Dataset**: GSM8K (grade-school mathematical reasoning benchmark)

### PRM Architecture

- **Base Model**: Qwen/Qwen3-8B (8 billion parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
- **PRM Head**: Single linear layer for binary step classification
- **Training Data**: PRM800K dataset (563,181 labeled steps from 97,782 problems)
- **Training Performance**: 94.54% test accuracy on step-level verification

### Experimental Setup

We employ three main experimental approaches:

1. **Best-of-N Sampling**: Generate N reasoning chains and select the best using PRM scores
2. **Beam Search**: Use PRM scores to guide beam search through the solution space
3. **Fixed Pool Evaluation**: Compare PRM selection vs. Self-Consistency on fixed candidate sets

## Repository Structure

```
Process-Reward-Models/
├── README.md
├── requirements.txt                   # Python dependencies
├── Training/
│   ├── PRM_New_Run.ipynb             # Main PRM training pipeline
│   └── checkpoint_to_hf.ipynb        # Convert checkpoints to HuggingFace format
└── Experiments/
    ├── PRM_Sampling_experiment.ipynb  # Best-of-N efficiency curve experiments
    ├── prm_step_scores.jsonl         # Step-level PRM score outputs
    ├── Beam_Search/
    │   ├── PRM_Beam_Search_Experiment.ipynb  # Beam search evaluation
    │   └── beam_search_results (1).jsonl     # Beam search results
    └── Fixed_Pool/
        ├── PRM_Fixed_Pool.ipynb              # Fixed pool evaluation
        └── fixed_pool_v2_fewshot_full_test.jsonl  # Pre-generated candidates
```

## Getting Started

### Hardware Requirements

This project was developed and tested in **Google Colab with NVIDIA A100 GPU (80GB VRAM)**.

**Recommended Specifications:**

- **GPU**: NVIDIA A100 (80GB) for training, T4/V100 for inference
- **VRAM**:
  - Training: 24GB+ (A100 recommended, ~7 hours for 2 epochs)
  - Inference: 16GB+ for experiments
- **RAM**: 16GB+ system memory
- **Storage**: ~50GB free space for models and datasets

**Google Colab Tiers:**

- **Free Tier (T4 GPU)**: Sufficient for running all experiments
- **Colab Pro (V100/A100)**: Recommended for training (~7 hours on A100)

### Prerequisites

This project was developed and tested in **Google Colab** (latest version as of November 2024). The following packages are required:

#### Environment Requirements

```bash
# Core Deep Learning
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
accelerate>=0.24.0

# Data Processing
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0

# HuggingFace Integration
huggingface_hub>=0.19.0

# Utilities
wget>=3.2
matplotlib>=3.7.0

# Optional: For Jupyter notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0
```

#### Installation

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch>=2.0.0 transformers>=4.35.0 peft>=0.7.0 accelerate>=0.24.0
pip install datasets>=2.14.0 numpy>=1.24.0 tqdm>=4.65.0
pip install huggingface_hub>=0.19.0 wget>=3.2 matplotlib>=3.7.0
```

**Note**: Many of these packages come pre-installed in Google Colab. This project does NOT use `vllm` or `wandb`.

### Quick Start

#### 1. Clone the Repository

```bash
git clone https://github.com/devangb3/Process-Reward-Models.git
cd Process-Reward-Models
```

#### 2. Set Up Environment

If running locally (not in Colab), install dependencies:

```bash
pip install -r requirements.txt
```

If using **Google Colab** (recommended), mount your Google Drive and install missing packages:

```python
from google.colab import drive
drive.mount('/content/drive')

# Install any missing packages
!pip install wget tqdm datasets
```

#### 3. Authenticate with HuggingFace

Get access to the required models by authenticating:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Get your HuggingFace token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### 4. Run Your First Experiment

Open and run the Best-of-N sampling experiment:

```bash
# In Colab or Jupyter:
# Navigate to Experiments/PRM_Sampling_experiment.ipynb
# Run all cells to evaluate PRM vs Self-Consistency
```

### HuggingFace Authentication

Before running training or experiments, you need to authenticate with HuggingFace to access the models:

1. Get a HuggingFace access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. In your Colab notebook, run:

   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

3. Enter your token when prompted

This is required to access:

- Base models (Qwen/Qwen3-8B, Qwen/Qwen3-0.6B)
- Pre-trained PRM adapters (devangb4/prm-qwen3-8b-bf16-6k, devangb4/prm-qwen3-8b-bf16-full)

### Available Pre-trained Models

We provide several trained PRM checkpoints on HuggingFace:

1. **devangb4/prm-qwen3-8b-bf16-6k** (Intermediate Checkpoint)
   - Training steps: 6,000
   - Use case: Experimentation and quick testing
   - Contains: LoRA adapter + PRM head + tokenizer

2. **devangb4/prm-qwen3-8b-bf16-full** (Full Training)
   - Training steps: 14,628 (2 full epochs)
   - Test accuracy: 94.54% on step verification
   - Use case: Best performing model for evaluation
   - Contains: LoRA adapter + PRM head + tokenizer

3. **Kaubitech/prm-qwen3-8b-bf16-final** (Alternative)
   - Complete final model checkpoint
   - Use case: Alternative trained checkpoint

**Base Models** (auto-downloaded from HuggingFace):

- **Qwen/Qwen3-8B**: PRM base model (8B parameters)
- **Qwen/Qwen3-0.6B**: Generator model (600M parameters)

### Training a PRM

1. Navigate to the `Training/` directory
2. Open `PRM_New_Run.ipynb` for the complete training pipeline
3. **Important**: Authenticate with HuggingFace first (see above)
4. Configure your project path:

   ```python
   PROJECT_PATH = "/content/drive/MyDrive/your_project_path"
   BASE_MODEL_NAME = "Qwen/Qwen3-8B"
   ```

5. The notebook includes:
   - Automatic dataset download from PRM800K
   - LoRA fine-tuning configuration
   - Training with balanced class sampling
   - Checkpoint saving every 2,000 steps
   - Model export to HuggingFace format

**Training Configuration:**

- Epochs: 2
- Batch size: 4 per device
- Gradient accumulation: 4 steps
- Learning rate: 2e-4
- Scheduler: Cosine with 3% warmup
- Precision: bfloat16
- Training time: ~7 hours on A100 GPU

### Dataset Information

**PRM800K Dataset:**

- Source: OpenAI ([GitHub](https://github.com/openai/prm800k))
- Training: 97,782 problems → 563,181 labeled steps
- Test: 2,762 problems → 16,153 labeled steps
- Labels: Binary (correct/incorrect) per reasoning step
- Class balancing: 70% positive, 30% negative (from original 93.8% positive)

The dataset is automatically downloaded by the training notebook.

**After training**, upload your model to HuggingFace:

```python
from huggingface_hub import login, create_repo, upload_folder

login()  # Login with your HF token

HF_REPO_ID = "your-username/prm-qwen3-8b"
create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)

upload_folder(
    folder_path=EXPORT_DIR,
    repo_id=HF_REPO_ID,
    commit_message="Upload PRM model"
)
```

### Running Experiments

#### Experiment 1: Baseline Evaluations

**Notebook**: `Experiments/Fixed_Pool/PRM_Fixed_Pool.ipynb`

**Objective**: Rigorous controlled comparison using fixed candidate pools.

**Setup**:

- Dataset: Full GSM8K test set (1,319 problems)
- Pool size: 16 pre-generated candidates per problem
- Temperature: 0.6 with few-shot prompting
- Methods: Self-Consistency vs PRM Best-of-N on same pool

**Run the experiment**:

```python
# Open Experiments/Fixed_Pool/PRM_Fixed_Pool.ipynb  
# Uses: fixed_pool_v2_fewshot_full_test.jsonl
# Most rigorous evaluation - controls for generation variance
```

#### Experiment 2: Search Strategies

**Notebook**: `Experiments/Beam_Search/PRM_Beam_Search_Experiment.ipynb`

**Objective**: Evaluate PRM-guided beam search for structured exploration.

**Setup**:

- Dataset: 500 problems from GSM8K
- Beam width (K): 4
- Beam expansion (M): 4 candidates per position
- Max steps: 25

**Algorithm**:

1. Generate K diverse initial steps
2. For each beam, generate M continuations
3. Score all K×M candidates with PRM
4. Keep top K scoring beams
5. Repeat until solution or max steps

**Run the experiment**:

```python
# Open Experiments/Beam_Search/PRM_Beam_Search_Experiment.ipynb
# Results saved to: beam_search_results (1).jsonl
```

#### Experiment 3: Best-of-N Sampling (Efficiency Curve)

**Notebook**: `Experiments/Compute_Efficiency/PRM_Sampling_experiment.ipynb`

**Objective**: Compare PRM-guided Best-of-N against Self-Consistency across different compute budgets.

**Setup**:

- Dataset: 250 random problems from GSM8K test set
- Budgets: [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 32, 64]
- Generator: Qwen3-0.6B with temperature=0.7
- PRM: devangb4/prm-qwen3-8b-bf16-6k

**Methods**:

- **Self-Consistency**: Generate N solutions, majority vote on final answers
- **PRM Best-of-N**: Generate N solutions, select highest PRM-scored chain

**Run the experiment**:

```python
# Open Experiments/Compute_Efficiency/PRM_Sampling_experiment.ipynb
# Results saved to: efficiency_curve_data_250.jsonl
# Plot generated as: efficiency_curve.png
```

## Results

Our experiments demonstrate that Process Reward Models provide significant advantages in constrained-compute scenarios:

| N | Self-Consistency | PRM | Improvement |
|---|------------------|-----|-------------|
| 1 | 36.8% | 36.8% | 0.0% |
| 2 | 42.0% | 48.4% | **15.2%** |
| 3 | 40.4% | 51.2% | **26.7%** |
| 4 | 44.8% | 56.8% | **26.8%** |
| 5 | 50.8% | 60.0% | **18.1%** |
| 7 | 50.8% | 60.4% | **18.9%** |
| 8 | 54.8% | 60.8% | **11.0%** |
| 9 | 52.8% | 60.0% | **13.6%** |
| 11 | 52.8% | 62.4% | **18.2%** |
| 13 | 55.6% | 63.2% | **13.7%** |
| 15 | 56.8% | 62.8% | **10.6%** |
| 16 | 57.2% | 62.0% | **8.4%** |
| 32 | 58.0% | 63.2% | **9.0%** |
| 64 | 58.4% | 63.6% | **8.9%** |

*Accuracy on GSM8K dataset*

### Key Findings

- **At N=5**: PRM achieves 60.0% accuracy vs. 50.8% for Self-Consistency (18.1% improvement)
- **Maximum improvement**: PRM shows **26.8% improvement** at N=4
- **Consistent gains**: PRM outperforms Self-Consistency across all budget levels N≥2
- **Efficiency**: PRM at N=5 matches or exceeds Self-Consistency at N=64, representing a **12.8× reduction** in required generations

The PRM approach achieves superior accuracy with far fewer generations, making it ideal for deployment in resource-constrained environments.

## Related Work

- **Let's Verify Step by Step (PRM800K)**: [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- **LoRA: Low-Rank Adaptation of Large Language Models**: [ICLR 2022](https://openreview.net/forum?id=nZeVKeeFYf9)
- **Training Verifiers to Solve Math Word Problems (GSM8K)**: [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**: [NeurIPS 2022](https://arxiv.org/abs/2201.11903)
- **Solving Math Word Problems with Process- and Outcome-based Feedback**: [arXiv:2211.14275](https://arxiv.org/abs/2211.14275)
- **Self-Consistency Improves Chain of Thought Reasoning**: [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- **Qwen3 Technical Report**: [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built upon the Qwen model family
- Uses the PRM800K dataset for training
- Evaluated on the GSM8K benchmark
