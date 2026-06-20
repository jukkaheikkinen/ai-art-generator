# AI Art Generator

Generate digital art locally using Stable Diffusion — no API costs, no subscriptions, unlimited images.

Art styles are defined as **themes** (JSON files you edit freely), and per-machine GPU optimizations are stored in **machine profiles** — neither your art style nor your hardware setup requires touching Python code.

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [First run](#first-run)
4. [Generating images](#generating-images)
5. [Themes](#themes)
6. [Machine profiles](#machine-profiles)
7. [Full CLI reference](#full-cli-reference)
8. [Troubleshooting](#troubleshooting)

---

## Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | [python.org](https://python.org) |
| NVIDIA GPU | 4 GB VRAM | CPU works too — much slower |
| CUDA driver | 11.8+ | Checked automatically by `setup.ps1` |
| Disk space | ~6 GB | ~4 GB model cache + ~2.5 GB PyTorch |

> **No GPU?** Use `.\setup.ps1 --cpu` — a single image takes minutes instead of seconds.

---

## Installation

```powershell
git clone https://github.com/jukkaheikkinen/ai-art-generator
cd ai-art-generator
.\setup.ps1
```

`setup.ps1` handles everything: checks Python, creates a `.venv`, detects your GPU and CUDA version, installs the correct PyTorch build, installs all dependencies, and verifies the result.

```powershell
.\setup.ps1            # recommended — auto-detects GPU
.\setup.ps1 --cpu      # force CPU-only (no GPU required, but slow)
.\setup.ps1 --no-venv  # skip virtual environment creation
```

Activate the environment before each session:

```powershell
.venv\Scripts\Activate.ps1
```

---

## First run

Detect your GPU and save the optimal settings to `machine.json`:

```powershell
python generator.py machine detect
```

The first time you generate an image, Stable Diffusion 1.5 (~4 GB) is downloaded from Hugging Face and cached in `~/.cache/huggingface/`. Subsequent runs load from cache instantly.

---

## Generating images

### Batch — generate many images

```powershell
python generator.py batch --theme cycling             # up to 50 images
python generator.py batch --theme cycling --count 10  # limit to 10
python generator.py batch --theme cycling --steps 40  # higher quality
python generator.py batch --theme cycling --filter subject="road cyclist,sprinter"
```

Images are saved to the folder defined in the theme and named `image_0001.png`, `image_0002.png`, … Numbering continues from where you left off — re-running never overwrites existing images.

### Single — one image

```powershell
python generator.py single --theme cycling                                # random variation
python generator.py single --theme cycling --prompt "custom prompt text"  # custom prompt
```

### List — preview prompts without generating

```powershell
python generator.py list --theme cycling               # all combinations
python generator.py list --theme cycling --count 20    # first 20
python generator.py list --theme cycling --filter subject="solo climber"
```

---

## Themes

Themes are JSON files in the `themes/` folder. The built-in `cycling` theme is created automatically on first run.

```powershell
python generator.py themes list              # list all themes
python generator.py themes show cycling      # inspect a theme
python generator.py themes new               # create a new theme interactively
python generator.py themes edit cycling      # open theme file in your editor
python generator.py themes delete cycling    # delete a theme
```

### Creating a theme

`themes new` walks you through building a theme step by step — no JSON editing required:

```
python generator.py themes new

  Theme display name  > Fantasy Character Art
  Slug (filename)     > fantasy_characters
  Output folder       > fantasy_output

  Variation key       > subject
  Values for subject  > knight, wizard, rogue, ranger

  Variation key       > environment
  Values for environment > enchanted forest, ancient ruins, misty mountains

  Variation key       > style
  Values for style    > oil painting, watercolor, concept art

  Available placeholders: {subject}, {environment}, {style}
  Prompt template     > detailed fantasy portrait of a {subject} in {environment}, {style}, dramatic lighting

  ✓ Theme saved: themes/fantasy_characters.json   (36 total combinations)
```

### Theme file format

```json
{
  "name": "Fantasy Character Art",
  "description": "Painterly fantasy characters in various settings",
  "output_dir": "fantasy_output",
  "model": "runwayml/stable-diffusion-v1-5",
  "prompt_template": "detailed fantasy portrait of a {subject} in {environment}, {style}, dramatic lighting",
  "negative_prompt": "blurry, ugly, distorted, watermark, text, modern, photorealistic",
  "variations": {
    "subject":     ["knight", "wizard", "rogue", "ranger"],
    "environment": ["enchanted forest", "ancient ruins", "misty mountains"],
    "style":       ["oil painting", "watercolor", "concept art"]
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✓ | Display name shown in the CLI |
| `prompt_template` | ✓ | Format string — `{key}` placeholders must match variation keys |
| `variations` | ✓ | Keys mapped to lists of values; every combination = one image |
| `negative_prompt` | — | What the model should avoid — has a big impact on quality |
| `output_dir` | — | Output folder (default: `<slug>_output`) |
| `model` | — | HuggingFace model ID (default: `runwayml/stable-diffusion-v1-5`) |

### Filtering variations at runtime

```powershell
# Only generate images for two subjects, all other variations unchanged
python generator.py batch --theme cycling --filter subject="road cyclist,sprinter"

# Combine multiple filters
python generator.py batch --theme cycling --filter subject="solo climber" --filter palette="muted earth tones"
```

---

## Machine profiles

Different GPUs need different memory optimizations. Profiles store these settings so you do not need to pass flags manually every time.

```powershell
python generator.py machine detect      # auto-detect GPU and save to machine.json
python generator.py machine show        # show active profile and all its settings
python generator.py machine set high    # manually set a profile
python generator.py machine profiles    # list all profiles
```

### Available profiles

| Profile | For | VRAM | Size | Steps | Optimizations |
|---------|-----|------|------|-------|---------------|
| `high` | RTX 3090/4090, A100 | 16+ GB | 768 px | 40 | none needed |
| `medium` | RTX 3070/3080/4070 | 8-12 GB | 512 px | 30 | attention + VAE slicing |
| `low` | Quadro P620, GTX 1060 | 4-6 GB | 512 px | 30 | attention + VAE slicing |
| `minimal` | GT 1030, integrated | < 4 GB | 512 px | 25 | + VAE tiling + CPU offload |
| `cpu` | No GPU | n/a | 512 px | 20 | CPU mode (float32) |

`machine detect` saves `machine.json` next to `generator.py`. This file is **gitignored** — each machine has its own. On a new machine, just run `machine detect` after setup.

**Profile priority** (highest wins):
1. `--profile NAME` on the command line — one-off override
2. Profile saved in `machine.json` — set by `machine detect` or `machine set`
3. Auto-detected from VRAM — fallback when no `machine.json` exists

### Customising profiles

Edit `profiles.json` to tweak settings. You only need to include keys you want to override:

```json
{
  "low": {
    "steps": 35,
    "guidance_scale": 8.0
  }
}
```

---

## Full CLI reference

```
python generator.py batch   --theme NAME [--count N] [--filter KEY=V1,V2]
                            [--profile NAME] [--steps N] [--size N] [--guidance F]
                            [--model MODEL_ID] [--output DIR] [--cpu]

python generator.py single  --theme NAME [--prompt TEXT]
                            [--profile NAME] [--steps N] [--size N] [--guidance F]
                            [--model MODEL_ID] [--output DIR] [--cpu]

python generator.py list    --theme NAME [--count N] [--filter KEY=V1,V2]

python generator.py themes  list | show NAME | new | edit NAME | delete NAME

python generator.py machine detect | show | set NAME | profiles
```

| Flag | Description |
|------|-------------|
| `--count N` | Max images per batch run (default: 50) |
| `--steps N` | Inference steps — higher = sharper, slower (default: from profile) |
| `--size N` | Image size in pixels, square (default: from profile) |
| `--guidance F` | Prompt guidance scale — higher = more literal (default: from profile) |
| `--filter KEY=V1,V2` | Narrow a variation list for this run (repeatable) |
| `--profile NAME` | Override machine profile for this run |
| `--cpu` | Force CPU mode |

---

## Troubleshooting

**`CUDA out of memory`**
Switch to a more conservative profile:
```powershell
python generator.py machine set minimal
```

**Slow first run**
The model (~4 GB) is downloading and will be cached after the first run.

**`torch` not found**
Activate the virtual environment first: `.venv\Scripts\Activate.ps1`

**Images look blurry or low quality**
Increase steps (`--steps 40`) or review the theme`s `negative_prompt`.

**PowerShell "running scripts is disabled"**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
