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

### 1.Imputation
This module recovers missing values in tabular data using advanced generative models (e.g., GAIN, VAEGAIN, SCIS).

### How to use:
You need a dataset with missing values and a corresponding boolean mask (where 1 indicates observed values and 0 indicates missing values).

```bash
import pandas as pd
from dataprep.tabular.imputation.GAIN import GAIN

# 1. Load Data and Mask (NumPy arrays expected)
data_missing = pd.read_csv('datasets/imputation/weather_raw.csv').values
missing_mask = pd.read_csv('datasets/imputation/weather_missing_mask.csv').values

# 2. Initialize the Model
# You can swap GAIN with VAEGAIN or SCIS
model = GAIN(
    batch_size=128, 
    hint_rate=0.9, 
    alpha=100, 
    epoch=1000,
    device='cuda' # or 'cpu'
)

# 3. Train and Predict
imputed_data = model.train_and_predict(data_missing, missing_mask)

# Save results
pd.DataFrame(imputed_data).to_csv('imputed_results.csv', index=False)
```

### 2.Detection
This module identifies dirty or anomalous cells within a dataset. It includes LLM-assisted zero-shot detection (ZeroED) alongside standard baselines like Isolation Forest and LOF.

### How to use:
For LLM-based detection, you need to configure your API base and model name (e.g., qwen2.5-7b or gpt-3.5-turbo).

```bash
import pandas as pd
from dataprep.tabular.detection.ZeroED import ZeroED

# 1. Load Dirty Data
df_raw = pd.read_csv('datasets/detection/rayyan_dirty_100.csv')

# 2. Initialize Detector
detector = ZeroED(
    model_name="qwen2.5-7b", 
    api_use=True,
    base_url="http://localhost:8000/v1", # Your LLM API endpoint
    api_key="YOUR_API_KEY",
    n_method="agglomerative"
)

# 3. Train and Predict 
error_mask = detector.train_and_predict(df_raw) # Returns a boolean matrix

# Save the generated mask
pd.DataFrame(error_mask).to_csv('generated_error_mask.csv', index=False)
```

### 3.Correction
Once errors are detected, this module repairs the specific dirty cells. ZeroEC utilizes LLMs and local embeddings to smartly correct data based on context.

> ⚠️ **Prerequisite for ZeroEC:**
> The embedding-based correction requires the `all-MiniLM-L6-v2` model.
> 
> 1. Visit the [ZeroEC Repository](https://github.com/YangChen32768/ZeroEC.git) or HuggingFace.
> 2. Download the `all-MiniLM-L6-v2` folder.
> 3. Place it within your local directory and point to it using the `embedding_model_path` parameter.

### How to use:
You need a dataset with missing values and a corresponding boolean mask (where 1 indicates observed values and 0 indicates missing values).

```bash
import pandas as pd
from dataprep.tabular.correction.ZeroEC import ZeroEC

# 1. Initialize Corrector
corrector = ZeroEC(
        model_name="qwen2.5-7b",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="EMPTY",
        embedding_model_path=f'tabular/correction/all-MiniLM-L6-v2',
        human_repair_num=10,
        output_dir=f'./runs_output',
        clean_data_path="datasets/detection/rayyan_clean_100.csv",
        dirty_data_path="datasets/detection/rayyan_dirty_100.csv",
        detection_path="datasets/detection/rayyan_dirty_error_detection_100.csv",
        prompt_dir=f'{BASE_DIR}/prompt_templates',
        max_workers=3
    )

# 2. Run Correction Pipeline
# The model will target only the cells flagged as True in the detection path
cleaned_df = corrector.train_and_predict()

# 3. Save Corrected Data
cleaned_df.to_csv('final_corrected_data.csv')
```

## 🖥️ Interactive Web Console

If you prefer to use DataPrep via an interactive graphical interface, please refer to the **DataPrep Console User Manual**.md for detailed setup and usage instructions.
