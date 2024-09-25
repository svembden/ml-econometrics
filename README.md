# ML-Econometrics

This repository contains the code for the course "Machine Learning in Econometrics" at the Erasmus University Rotterdam. This code is written in Python using pytorch. The code is created by the "free jacamar of mastery" team. AKA: ...

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

You can replace cu118 with the appropriate CUDA version for your system (e.g., cu117, cu116). If CUDA is not available, PyTorch will fall back to the CPU version (which is honestly fine for this assignment).
