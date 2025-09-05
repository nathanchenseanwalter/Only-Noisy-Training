# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of "Only-Noisy Training" (ONT), a self-supervised speech denoising method that uses only noisy audio signals. The project implements three model variants: ONT (basic), ONT-rTSTM (real-valued transformer), and ONT-cTSTM (complex-valued transformer). It was written with oudated packages and code. Your task is to update this to be compatible with the latest version of python, 3.13, and the latest version of pytorch while maintaining core functionality.

## Dependencies and Environment

The project uses `uv` for dependency management. Dependencies are defined in `pyproject.toml`:
- Core ML: PyTorch, TorchAudio, NumPy
- Audio processing: librosa, scipy  
- Metrics: pesq, pystoi
- Utilities: tqdm, matplotlib, numba

Install dependencies:
```bash
uv sync
```

Run Python scripts:
```bash
uv run python <script_name>.py
```

## Key Components

### Core Architecture
- `DCUnet10_TSTM/DCUnet.py`: Complex-valued U-Net models (DCUnet10, DCUnet10_rTSTM, DCUnet10_cTSTM)
- `DCUnet10_TSTM/Dual_Transformer.py`: Transformer modules for enhanced models
- `loss.py` + `loss_utils.py`: Custom regularized loss function combining time and frequency domain losses
- `dataset_utils.py`: Audio dataset handling and subsampling functions (subsample2, subsample4)

### Training Pipeline
- `train.py`: Main training script with complete training loop, evaluation metrics, and model checkpointing
- Audio processing: 48kHz sampling, 1022-point FFT, 256 hop length
- Training uses paired subsampling from single noisy audio samples

### Dataset Generation
- `whitenoise_dataset_generator.py`: Generate synthetic noisy datasets using white Gaussian noise
- `urbansound_dataset_generator.py`: Generate real-world noisy datasets using UrbanSound8K

### Evaluation
- `metrics.py` + `metrics_utils.py`: Audio quality metrics (PESQ-WB/NB, SNR, SSNR, STOI)
- Supports resampling to different rates for metric computation

## Training Commands

Generate datasets:
```bash
uv run python whitenoise_dataset_generator.py
uv run python urbansound_dataset_generator.py
```

Train model:
```bash
uv run python train.py
```

## Model Configuration

Key parameters in `train.py`:
- Sample rate: 48000 Hz
- FFT size: 1022
- Hop length: 256
- Training pairs are 1/2 length of original samples
- Loss weights: α=0.8, β=1/200, γ=2 (synthetic) or γ=1 (real-world)
- Optimizer: Adam with 0.001 learning rate
- 6 transformer blocks in TSTM variants

## Dataset Structure

Expected directory structure:
```
Datasets/
├── WhiteNoise_Train_Input/
├── WhiteNoise_Train_Output/
├── WhiteNoise_Test_Input/
├── US_Class{N}_Train_Input/  (for UrbanSound)
├── US_Class{N}_Train_Output/
├── US_Class{N}_Test_Input/
└── clean_testset_wav/
```

Models and results are saved to `SNA-DF/DCUnet10_complex_TSTM_subsample2/` directory structure.

## Hardware Requirements

- CUDA 11.4.48 recommended
- 1-4 GPUs supported
- Automatic CPU fallback if no GPU available