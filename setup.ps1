# AI Art Generator — Setup Script
# Detects your GPU/CUDA version and installs the correct dependencies automatically.
#
# Usage:
#   .\setup.ps1            # standard setup
#   .\setup.ps1 --cpu      # CPU-only (no GPU required)
#   .\setup.ps1 --no-venv  # skip virtual environment creation

param(
    [switch]$cpu,
    [switch]$NoVenv
)

$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Header($text) {
    Write-Host ""
    Write-Host "── $text " -ForegroundColor Cyan -NoNewline
    Write-Host ("─" * [Math]::Max(0, 60 - $text.Length)) -ForegroundColor DarkGray
}

function Write-Ok($text)   { Write-Host "  ✓ $text" -ForegroundColor Green  }
function Write-Warn($text) { Write-Host "  ! $text" -ForegroundColor Yellow }
function Write-Fail($text) { Write-Host "  ✗ $text" -ForegroundColor Red    }

function Exit-WithError($text) {
    Write-Fail $text
    exit 1
}

# ── Python check ──────────────────────────────────────────────────────────────

Write-Header "Checking Python"

try {
    $pyVersion = python --version 2>&1
} catch {
    Exit-WithError "Python not found. Install Python 3.10+ from https://python.org"
}

$versionMatch = $pyVersion -match "Python (\d+)\.(\d+)"
if (-not $versionMatch) { Exit-WithError "Could not parse Python version: $pyVersion" }

$major = [int]$Matches[1]
$minor = [int]$Matches[2]

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    Exit-WithError "Python 3.10+ required. Found: $pyVersion"
}

Write-Ok "$pyVersion"

# ── Virtual environment ────────────────────────────────────────────────────────

if (-not $NoVenv) {
    Write-Header "Virtual environment"

    if (Test-Path ".venv") {
        Write-Ok ".venv already exists — skipping creation"
    } else {
        Write-Host "  Creating .venv..." -ForegroundColor DarkGray
        python -m venv .venv
        Write-Ok ".venv created"
    }

    # Activate
    & ".venv\Scripts\Activate.ps1"
    Write-Ok "Activated .venv"
} else {
    Write-Warn "Skipping virtual environment (--no-venv)"
}

# ── Detect GPU / CUDA ─────────────────────────────────────────────────────────

Write-Header "Detecting GPU"

$torchIndex = $null
$torchLabel = $null

if ($cpu) {
    Write-Warn "CPU-only mode requested (--cpu)"
    $torchIndex = "https://download.pytorch.org/whl/cpu"
    $torchLabel = "CPU"
} else {
    try {
        $nvidiaSmi = nvidia-smi 2>&1
        $cudaLine  = $nvidiaSmi | Select-String "CUDA Version"
        $gpuLine   = $nvidiaSmi | Select-String "^\|.*%" | Select-Object -First 1

        if ($cudaLine -match "CUDA Version:\s+(\d+)\.(\d+)") {
            $cudaMajor = [int]$Matches[1]
            $cudaMinor = [int]$Matches[2]
            $cudaVer   = "$cudaMajor.$cudaMinor"

            Write-Ok "CUDA $cudaVer detected"

            # Map CUDA version to PyTorch index
            # PyTorch ships wheels for specific CUDA versions; pick the highest supported
            if     ($cudaMajor -gt 12 -or ($cudaMajor -eq 12 -and $cudaMinor -ge 4)) {
                $torchIndex = "https://download.pytorch.org/whl/cu124"
                $torchLabel = "CUDA 12.4"
            } elseif ($cudaMajor -eq 12) {
                $torchIndex = "https://download.pytorch.org/whl/cu121"
                $torchLabel = "CUDA 12.1"
            } elseif ($cudaMajor -eq 11 -and $cudaMinor -ge 8) {
                $torchIndex = "https://download.pytorch.org/whl/cu118"
                $torchLabel = "CUDA 11.8"
            } else {
                Write-Warn "CUDA $cudaVer is older than 11.8 — falling back to CPU build"
                $torchIndex = "https://download.pytorch.org/whl/cpu"
                $torchLabel = "CPU (CUDA too old)"
            }

            # Print GPU name if we can parse it
            if ($nvidiaSmi -match "(\w[\w\s]+(?:RTX|GTX|Quadro|Tesla|A\d|H\d|V\d)[\w\s\d]*)") {
                Write-Ok "GPU: $($Matches[1].Trim())"
            }
        } else {
            Write-Warn "nvidia-smi found but CUDA version not detected — using CPU build"
            $torchIndex = "https://download.pytorch.org/whl/cpu"
            $torchLabel = "CPU"
        }
    } catch {
        Write-Warn "No NVIDIA GPU detected — using CPU-only PyTorch build"
        $torchIndex = "https://download.pytorch.org/whl/cpu"
        $torchLabel = "CPU"
    }
}

Write-Ok "PyTorch build: $torchLabel"

# ── Install PyTorch ────────────────────────────────────────────────────────────

Write-Header "Installing PyTorch ($torchLabel)"
Write-Warn "This may download up to ~2.5 GB — please be patient…"

pip install torch torchvision torchaudio --index-url $torchIndex --quiet
if ($LASTEXITCODE -ne 0) { Exit-WithError "PyTorch installation failed" }
Write-Ok "PyTorch installed"

# ── Install other dependencies ─────────────────────────────────────────────────

Write-Header "Installing dependencies"

pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Exit-WithError "Dependency installation failed" }
Write-Ok "All dependencies installed"

# ── Verify ────────────────────────────────────────────────────────────────────

Write-Header "Verifying installation"

$verifyScript = @"
import torch
from diffusers import StableDiffusionPipeline
from rich.console import Console

c = Console()
cuda = torch.cuda.is_available()
device = 'CUDA' if cuda else 'CPU'
gpu = torch.cuda.get_device_name(0) if cuda else 'n/a'
vram = f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda else 'n/a'
c.print(f'  torch      [green]{torch.__version__}[/green]')
c.print(f'  device     [cyan]{device}[/cyan]')
c.print(f'  GPU        [cyan]{gpu}[/cyan]  VRAM: [cyan]{vram}[/cyan]')
"@

python -c $verifyScript
if ($LASTEXITCODE -ne 0) { Exit-WithError "Verification failed — check the errors above" }

# ── Done ──────────────────────────────────────────────────────────────────────

Write-Header "Setup complete"

if (-not $NoVenv) {
    Write-Host ""
    Write-Host "  Activate the environment before running:" -ForegroundColor DarkGray
    Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
}

Write-Host "  Try it out:" -ForegroundColor DarkGray
Write-Host "    python generator.py themes list" -ForegroundColor White
Write-Host "    python generator.py single --theme cycling" -ForegroundColor White
Write-Host "    python generator.py batch  --theme cycling --count 5" -ForegroundColor White
Write-Host ""
