# RhythmFormer — rPPG Extraction via Periodic Sparse Attention

## Project Overview
Deep learning model for extracting remote photoplethysmography (rPPG) signals from face video.
Predicts blood volume pulse (BVP) waveform from 128x128 face crops, from which heart rate is derived via FFT.

- **Paper**: Zou et al. (2025), Pattern Recognition, Vol. 164
- **License**: MIT
- **Built on**: rPPG-Toolbox (Liu et al., 2023)

## Architecture

| Component | Details |
|-----------|---------|
| **Params** | 3.33M |
| **Input** | `(N, 160, 3, 128, 128)` — NDCHW: batch, frames, RGB, height, width |
| **Output** | `(N, 160)` — predicted rPPG waveform at frame rate |
| **Fusion_Stem** | Dual-stream: RGB frames + temporal differences → 64-ch features at 1/4 spatial |
| **Patch Embedding** | Conv3d(64, 64, k=(1,4,4), s=(1,4,4)) → `(N, 64, 160, 8, 8)` |
| **3 TPT Stages** | Temporal Periodic Transformer with sparse attention (topk=40), t_patch=(2,4,8) |
| **Output Head** | Global spatial mean → Conv1d(64,1) → squeeze → `(N, 160)` |

### Key Extraction Points
- **features_last**: `(N, 64, 160)` — after spatial pooling, before Conv1d output head
- **rPPG output**: `(N, 160)` — final predicted waveform

## Data Pipeline

### Input Format
- Face crops: 128x128 RGB, z-score standardized per chunk (global mean/std over all pixels)
- Chunk length: 160 frames at 30 fps (5.33 seconds)
- `DATA_FORMAT: NDCHW` — (batch, frames, channels, height, width)

### Preprocessing (BaseLoader)
1. Face detection: Haar cascade (`haarcascade_frontalface_default.xml`)
2. Enlarge box: `LARGE_BOX_COEF: 1.5` (used in ALL configs)
3. Crop + resize to 128x128 (`cv2.INTER_AREA`)
4. Z-score: `data = (data - np.mean(data)) / np.std(data)` — single scalar mean/std
5. Chunk into 160-frame non-overlapping segments
6. Save as .npy files

### Supported Datasets
MMPD, PURE, UBFC-rPPG, SCAMPS, COHFACE, BP4D+, UBFC-PHYS, VIPL-HR

## Training

```bash
python main.py --config_file configs/train_configs/intra/0MMPD_RHYTHMFORMER.yaml
```

- **Optimizer**: AdamW, weight_decay=0, LR=9e-3
- **Scheduler**: OneCycleLR (per-step)
- **Epochs**: 30
- **Batch size**: 4
- **Data augmentation** (AUG=1): HR-aware temporal resampling + horizontal flip

### Loss Function (`RhythmFormer_Loss`)
```
L = 0.2 * NegPearson + 1.0 * FrequencyDomain_CE + 1.0 * HR_KL
```
- NegPearson: temporal waveform correlation
- FrequencyDomain: cross-entropy on FFT power spectrum (HR range [45, 150] BPM)
- HR_KL: KL divergence on PSD distributions (Gaussian around GT HR, std=3 BPM)

### Output Normalization (REQUIRED)
After forward pass, predictions must be z-score normalized per sample:
```python
pred = (pred - pred.mean(dim=-1, keepdim=True)) / pred.std(dim=-1, keepdim=True)
```

## Inference

```bash
python main.py --config_file configs/infer_configs/UBFC-rPPG_MMPD_RHYTHMFORMER.yaml
```

### Pre-trained Checkpoints (in `PreTrainedModels/`)
| Checkpoint | Training Data | Size |
|-----------|--------------|------|
| MMPD_intra_RhythmFormer.pth | MMPD (most diverse) | 13.5 MB |
| PURE_cross_RhythmFormer.pth | PURE | 13.5 MB |
| UBFC_cross_RhythmFormer.pth | UBFC-rPPG | 13.5 MB |
| COHFACE_intra_RhythmFormer.pth | COHFACE | 13.5 MB |
| VIPL_RhythmFormer_fold1-5.pth | VIPL-HR (5-fold) | 13.5 MB |

**IMPORTANT**: Checkpoints saved with DataParallel — all keys have `module.` prefix.
Strip prefix when loading without DataParallel:
```python
state_dict = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
```

### HR Extraction from Predicted Waveform
1. Detrend (Whittaker smoother, lambda=100)
2. Bandpass filter: Butterworth 1st order, [0.75, 2.5] Hz
3. Welch PSD: `nfft=1e5/sr`, `nperseg=min(len(y)-1, 256)`
4. Peak in [45, 150] BPM → HR

## Evaluation Metrics
- HR MAE, RMSE, MAPE (BPM)
- Pearson correlation (predicted HR vs GT HR)
- SNR (dB): harmonic power vs noise in cardiac band

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point (train/test/unsupervised) |
| `config.py` | YACS-based configuration schema |
| `neural_methods/model/RhythmFormer.py` | Model architecture |
| `neural_methods/model/base/video_bra.py` | Periodic sparse attention blocks |
| `neural_methods/trainer/RhythmFormerTrainer.py` | Training/validation/test loops |
| `neural_methods/loss/TorchLossComputer.py` | Loss functions (Pearson + Freq + HR) |
| `dataset/data_loader/BaseLoader.py` | Data loading, face detection, preprocessing |
| `evaluation/post_process.py` | HR extraction (FFT, peak detection), SNR, detrending |
| `evaluation/metrics.py` | Metric computation (MAE, RMSE, Pearson, etc.) |

## MSVT Integration Context
This fork is being evaluated for integration with the MSVT project (`/home/tomerf/devel/msvt`):
- **Phase 1**: Zero-shot benchmark on MSVT webcam data (headset occlusion domain)
- **Phase 2**: Fine-tune on MSVT data if zero-shot MAE is 5-12 BPM
- **Phase 3**: Teacher distillation — generate pseudo-PPG labels for MSVT P-token training
- Benchmark script lives in MSVT repo: `scripts/analysis/rhythmformer_baseline.py`
- RhythmFormer repo stays read-only (imported via sys.path)
