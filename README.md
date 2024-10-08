# ML-Econometrics

This repository contains the code for the course "Machine Learning in Econometrics" at the Erasmus University Rotterdam. The code is written in Python. The code is created by the "free jacamar of mastery" group by Sem van Embden, Anna Grefhorst, Jaap Jansen & Luuk Omvlee.

## Installation
Make sure to install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

Additionally, to ensure PyTorch installs with CUDA support, you can specify it when running the installation command manually, as PyTorch installation with CUDA versions depends on your specific system configuration.
For example, to install PyTorch with CUDA 11.8, you can use the following command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You can replace cu118 with the appropriate CUDA version for your system (e.g., cu117, cu116). If CUDA is not available, PyTorch will fall back to the CPU version.

## Usage
For the final submission of the assignment, only the XGBOOST model (part 4) in ```main.ipynb``` is used. The code can be run to train a new model, or a saved model can be loaded.
