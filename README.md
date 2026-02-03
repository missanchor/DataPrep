# DataPrep

A comprehensive tool package designed for streamlined **data preparation**, cleaning, and preprocessing. This toolkit provides a robust environment for handling various data quality tasks, including integration with specialized error correction modules.

## 🛠 Installation & Environment Setup

Follow these steps to set up the environment and install the necessary dependencies. We recommend using **Conda** for environment management.

### 1. Create and Activate Environment
```bash
conda create -n dataprep python=3.10
conda activate dataprep
```

### 2. Clone the Repository
```bash
git clone https://github.com/missanchor/DataPrep.git
cd DataPrep
```

### 3. Install Dependencies
Install PyTorch with CUDA 11.8 support, followed by the project-specific requirements:
```bash
# Install PyTorch ecosystem
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 🚀 Getting Started

To understand how to use the different modules within this toolkit, please refer to the provided scripts in the `examples/` directory.
```bash
cd examples/
python imputation.py
python detection.py
python correction
```


## 🔍 Additional Setup for ZeroEC

If you intend to use the **ZeroEC** (Zero-shot Error Correction) module, an additional model asset is required:

1. Visit the [ZeroEC Repository](https://github.com/YangChen32768/ZeroEC.git).
2. Download the `all-MiniLM-L6-v2` folder.
3. Place the folder within your local `ZeroEC` directory or the designated path in your configuration to ensure the embedding-based correction works correctly.