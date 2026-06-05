# F1 Physics Engine

**Language-conditioned video generation for Formula 1 racing dynamics.**

Given a starting frame and a plain-English description of a manoeuvre, the model generates a short video clip predicting how the car will move — braking, turning, accelerating — all grounded in real telemetry data.

> *"The driver brakes aggressively and turns sharply right at high speed."*
> → Model generates the next 8 frames showing the resulting motion.

---

## How It Works

The system fine-tunes [Stable Video Diffusion (SVD-XT)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) using **LoRA adapters** and a learned **CLIP text conditioning** layer. Only ~0.1% of parameters are trained — the frozen SVD backbone provides strong video priors, while the LoRA layers steer it toward F1-specific physics.

```
Text prompt  →  CLIP encoder (frozen)  →  Linear projection (768→1024)  ─┐
                                                                           ▼
Start frame  →  VAE encoder (frozen)   →  Latents  →  UNet + LoRA (r=16) → Denoised latents → VAE decoder → Video
```

Real F1 telemetry (speed, throttle, brake, steering, gear, DRS) is automatically converted into natural-language prompts via `TelemetryPromptGenerator`, so training supervision comes directly from on-car sensor data.

---

## Project Structure

```
f1_physics_engine/
├── train.py                   # Training loop — Accelerate, min-SNR loss, cosine LR
├── inference.py               # Generate a video from a prompt + start frame
├── requirements.txt
├── configs/
│   └── default.yaml           # All hyperparameters in one place
├── models/
│   ├── __init__.py
│   └── lora_svd.py            # LanguageConditionedSVD, TextConditioner, build_model
├── data/
│   ├── __init__.py
│   └── dataset.py             # F1VideoDataset, SyntheticF1Dataset, TelemetryPromptGenerator
├── utils/
│   ├── __init__.py
│   └── evaluation.py          # OpticalFlowEvaluator, AttentionHeatmapExtractor, VideoMetrics
├── checkpoints/
│   └── config.yaml
└── slurm/
    └── train.sh               # SLURM script for HPC cluster
```

---

## Setup

**1. Clone and enter the repo**

```bash
git clone https://github.com/maahipatel05/f1-physics-engine.git
cd f1-physics-engine
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA 12.1 (recommended for training):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**3. Authenticate with HuggingFace**

The SVD model requires accepting Stability AI's license once:

```bash
huggingface-cli login
```

Then visit [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and click **"Agree and access repository"**.

---

## Data Preparation

### Quick start — no data needed

The built-in `SyntheticF1Dataset` generates random clips with realistic prompts. Use it to verify the pipeline runs end-to-end:

```bash
python train.py --config configs/default.yaml --synthetic
```

### Real F1 data

**Extract frames from video:**
```python
from data import F1VideoDataset

F1VideoDataset.extract_frames_from_video(
    video_path="race.mp4",
    output_dir="data/f1_dataset/sessions/2023_Monaco_VER/frames/",
    fps=10.0,
    image_size=128,
)
```

**Download and export telemetry:**
```python
from data import FastF1TelemetryLoader

loader = FastF1TelemetryLoader(year=2023, gp="Monaco", session_type="R", driver="VER")
loader.save_csv("data/f1_dataset/sessions/2023_Monaco_VER/telemetry.csv")
```

Expected layout:
```
data/f1_dataset/sessions/
└── 2023_Monaco_VER/
    ├── frames/
    │   ├── frame_000000.png
    │   └── ...
    └── telemetry.csv
```

---

## Training

**Single GPU:**
```bash
python train.py --config configs/default.yaml
```

**Multi-GPU with Accelerate:**
```bash
accelerate config   # one-time setup
accelerate launch --num_processes=4 train.py --config configs/default.yaml
```

**Resume from checkpoint:**
```bash
python train.py --config configs/default.yaml --resume ./checkpoints/checkpoint-0005000
```

**With WandB logging:**
```bash
export WANDB_API_KEY="your_key"
python train.py --config configs/default.yaml --wandb_run_name "run-001"
```

**On a SLURM cluster:**
```bash
sbatch slurm/train.sh
```

### Key hyperparameters (`configs/default.yaml`)

| Parameter | Default | Notes |
|---|---|---|
| `model.lora_r` | 16 | LoRA rank — higher uses more VRAM |
| `model.lora_alpha` | 32 | Effective scale = alpha / r = 2.0 |
| `training.batch_size` | 1 | Per-GPU |
| `training.gradient_accumulation_steps` | 8 | Effective batch size = 8 |
| `training.learning_rate` | 1e-4 | AdamW |
| `training.num_epochs` | 50 | |
| `training.snr_gamma` | 5.0 | Min-SNR loss weighting; set null to disable |
| `data.num_frames` | 8 | Frames per clip |
| `data.image_size` | 128 | Spatial resolution |

---

## Inference

```bash
python inference.py \
    --checkpoint ./checkpoints/checkpoint-0005000 \
    --start_frame ./data/test_frame.png \
    --prompt "The driver brakes aggressively and turns sharply right at high speed." \
    --output ./outputs/predicted_clip.mp4 \
    --num_frames 8 \
    --num_inference_steps 25 \
    --guidance_scale 3.5
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Checkpoint directory |
| `--start_frame` | required | Conditioning PNG frame |
| `--prompt` | required | Natural language action description |
| `--num_inference_steps` | 25 | More steps = sharper output |
| `--guidance_scale` | 3.5 | CFG scale — 1.0 disables guidance |
| `--save_heatmaps` | flag | Save cross-attention visualisations |
| `--seed` | 42 | Reproducibility |

The script also saves individual frames as PNGs and runs a self-evaluation comparing optical flow against a static baseline.

---

## Evaluation

### Optical flow MSE

```python
from utils import OpticalFlowEvaluator

evaluator = OpticalFlowEvaluator()
metrics = evaluator.compute_flow_mse(predicted_frames, ground_truth_frames)
# Returns: {"mse_u": ..., "mse_v": ..., "mse_total": ..., "epe": ...}
```

### Cross-attention heatmaps

Visualises which spatial regions the model attends to for each text token:

```python
from utils import AttentionHeatmapExtractor

extractor = AttentionHeatmapExtractor(model)
extractor.install_hooks(max_layers=4)
# run a forward pass
maps = extractor.collect_maps()
extractor.plot(maps, frames, prompt, save_dir="./eval/heatmaps")
extractor.remove_hooks()
```

### PSNR / SSIM / flow MSE over a dataset

```python
from utils import VideoMetrics

vm = VideoMetrics()
vm.update(pred_frames, gt_frames, prompt="...", clip_id="clip_001")
vm.save("./eval/metrics.json")
```

---

## Prompt Engineering

`TelemetryPromptGenerator` converts raw fastf1 telemetry into natural language automatically:

```python
from data import TelemetryPromptGenerator

gen = TelemetryPromptGenerator()
prompt = gen.generate(speed=310.0, throttle=0.0, brake=85.0, steering=45.0, gear=3, drs=0)
# → "The driver brakes aggressively and turns right at very high speed, in gear 3."
```

You can also write custom prompts directly for inference — no specific format is required.

---

## Hardware

| Hardware | VRAM usage | Approx. time / epoch (1024 clips) |
|---|---|---|
| A100 80 GB | ~40 GB | ~15 min |
| RTX 3090 24 GB | ~22 GB (batch=1) | ~45 min |
| Apple M1/M2 MPS | Unified memory | ~3 hrs (debugging only) |

Enable `mixed_precision: fp16` in `configs/default.yaml` to roughly halve VRAM on NVIDIA GPUs. Leave it as `"no"` on Apple Silicon.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- [Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) — Stability AI
- [fastf1](https://docs.fastf1.dev/) — open-source F1 telemetry
- [PEFT](https://github.com/huggingface/peft) — HuggingFace LoRA
- [Diffusers](https://github.com/huggingface/diffusers) — HuggingFace

---

Built by [Maahi Patel](https://github.com/maahipatel05) and [Jinay Shah](https://github.com/Jinay2205)