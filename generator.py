#!/usr/bin/env python3
"""
AI Art Generator
================
Generates digital art using local Stable Diffusion.
Themes define the style, prompt template, and variation lists — no script editing needed.

Usage:
  python generator.py themes list                       # list available themes
  python generator.py themes show cycling               # inspect a theme
  python generator.py themes new                        # create a new theme interactively
  python generator.py themes edit cycling               # open theme file in your editor

  python generator.py batch --theme cycling             # batch-generate with a theme
  python generator.py batch --theme cycling --count 10  # limit to 10 images
  python generator.py batch --theme cycling --filter subject="road cyclist,sprinter"
  python generator.py single --theme cycling            # one image with a random prompt
  python generator.py single --theme cycling --prompt "custom prompt text"
  python generator.py list --theme cycling              # preview prompts (no generation)
"""

import json
import os
import re
import subprocess
import sys
import itertools
import random
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

console = Console()

# ==========================
# PATHS & DEFAULTS
# ==========================

SCRIPT_DIR   = Path(__file__).parent
THEMES_DIR   = SCRIPT_DIR / "themes"
DEFAULT_THEME    = "cycling"
DEFAULT_MODEL    = "runwayml/stable-diffusion-v1-5"
DEFAULT_COUNT    = 50
DEFAULT_STEPS    = 30
DEFAULT_SIZE     = 512
DEFAULT_GUIDANCE = 7.5
PROFILES_FILE = SCRIPT_DIR / "profiles.json"
MACHINE_FILE  = SCRIPT_DIR / "machine.json"

BUILTIN_PROFILES = {
    "high":    {"label": "High-end GPU (16+ GB VRAM) — RTX 3090/4090, A100",      "dtype": "float16", "image_size": 768, "steps": 40, "guidance_scale": 7.5, "attention_slicing": False, "vae_slicing": False, "vae_tiling": False, "cpu_offload": "none"},
    "medium":  {"label": "Mid-range GPU (8–12 GB VRAM) — RTX 3070/3080/4070",     "dtype": "float16", "image_size": 512, "steps": 30, "guidance_scale": 7.5, "attention_slicing": True,  "vae_slicing": True,  "vae_tiling": False, "cpu_offload": "none"},
    "low":     {"label": "Low-end GPU (4–6 GB VRAM) — Quadro P620, GTX 1060",     "dtype": "float16", "image_size": 512, "steps": 30, "guidance_scale": 7.5, "attention_slicing": True,  "vae_slicing": True,  "vae_tiling": False, "cpu_offload": "none"},
    "minimal": {"label": "Very low VRAM (< 4 GB) — GT 1030, integrated GPU",      "dtype": "float16", "image_size": 512, "steps": 25, "guidance_scale": 7.0, "attention_slicing": True,  "vae_slicing": True,  "vae_tiling": True,  "cpu_offload": "model"},
    "cpu":     {"label": "CPU only — no GPU required, very slow",                  "dtype": "float32", "image_size": 512, "steps": 20, "guidance_scale": 7.0, "attention_slicing": True,  "vae_slicing": True,  "vae_tiling": False, "cpu_offload": "none"},
}

def load_profiles() -> dict:
    """Built-in profiles merged with any user overrides from profiles.json."""
    profiles = {k: dict(v) for k, v in BUILTIN_PROFILES.items()}
    if PROFILES_FILE.exists():
        user = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        for k, v in user.items():
            profiles[k] = {**profiles.get(k, {}), **v}
    return profiles

def detect_profile_name() -> str:
    """Auto-select profile based on available VRAM."""
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 16:
        return "high"
    if vram_gb >= 8:
        return "medium"
    if vram_gb >= 4:
        return "low"
    return "minimal"

def load_machine_config() -> dict:
    if MACHINE_FILE.exists():
        return json.loads(MACHINE_FILE.read_text(encoding="utf-8"))
    return {}

def save_machine_config(data: dict):
    MACHINE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def resolve_profile(profile_arg: str | None) -> tuple[str, dict]:
    """Priority: CLI --profile > machine.json > auto-detect."""
    profiles = load_profiles()
    name = profile_arg
    if not name or name == "auto":
        name = load_machine_config().get("profile")
    if not name or name == "auto":
        name = detect_profile_name()
    if name not in profiles:
        console.print(f"[red]Unknown profile '[bold]{name}[/bold]'. Available: {', '.join(profiles)}[/red]")
        sys.exit(1)
    return name, profiles[name]

# ==========================
# BUILT-IN CYCLING THEME
# (written to themes/cycling.json on first use)
# ==========================

CYCLING_THEME = {
    "name": "Retro Cycling Posters",
    "description": "1960s Tour de France style vintage cycling poster art",
    "output_dir": "retro_tdf_posters",
    "model": DEFAULT_MODEL,
    "prompt_template": (
        "retro 1960s Tour de France cycling poster, {subject} riding through {environment}, "
        "{composition}, {palette}, bold flat colors, simplified geometric shapes, "
        "grainy vintage print texture, muted retro palette, dramatic composition, "
        "vintage travel poster art, flat design illustration"
    ),
    "negative_prompt": (
        "blurry, ugly, distorted, noisy, low quality, watermark, text, signature, "
        "bad anatomy, extra limbs, modern, photorealistic, 3d render"
    ),
    "variations": {
        "subject": [
            "road cyclist", "gravel cyclist", "vintage steel bike rider",
            "peloton", "solo climber", "breakaway rider",
            "sprinter", "time trialist", "MTB rider", "silhouette cyclist",
        ],
        "environment": [
            "Col du Galibier", "Mont Ventoux", "the Pyrenees",
            "a Finnish pine forest", "a lakeside road", "an alpine village",
            "coastal cliffs", "rolling hills", "lavender fields",
            "a snowy forest", "a gravel road", "a mountain pass",
            "countryside", "city streets", "a desert canyon",
        ],
        "palette": [
            "warm retro colors, yellow orange red",
            "cool retro colors, blue green cream",
            "faded pastel retro tones",
            "high-contrast vintage colors",
            "muted earth tones",
        ],
        "composition": [
            "diagonal climb composition",
            "winding road composition",
            "centered silhouette composition",
            "distant rider composition",
            "close-up profile composition",
        ],
    },
}

# ==========================
# THEME HELPERS
# ==========================

def ensure_themes_dir():
    THEMES_DIR.mkdir(exist_ok=True)
    cycling_path = THEMES_DIR / "cycling.json"
    if not cycling_path.exists():
        cycling_path.write_text(json.dumps(CYCLING_THEME, indent=2))

def list_theme_names():
    ensure_themes_dir()
    return sorted(p.stem for p in THEMES_DIR.glob("*.json"))

def load_theme(name: str) -> dict:
    ensure_themes_dir()
    path = THEMES_DIR / f"{name}.json"
    if not path.exists():
        console.print(f"[red]Theme '[bold]{name}[/bold]' not found.[/red]")
        console.print(f"Available themes: {', '.join(list_theme_names()) or 'none'}")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))

def save_theme(name: str, data: dict):
    ensure_themes_dir()
    path = THEMES_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path

def theme_slug(display_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_")

# ==========================
# PROMPT BUILDER
# ==========================

def build_prompt(template: str, combo: dict) -> str:
    return template.format(**combo)

def all_combos(theme: dict, filters: dict | None = None) -> list[dict]:
    """
    Return every combination of variation values as a list of dicts.
    `filters` maps variation key → list of allowed values.
    """
    variations = theme["variations"]
    filtered = {
        key: [v for v in values if v in filters[key]] if filters and key in filters else values
        for key, values in variations.items()
    }
    keys   = list(filtered.keys())
    values = list(filtered.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def random_combo(theme: dict) -> dict:
    return {key: random.choice(vals) for key, vals in theme["variations"].items()}

def parse_filters(raw: list[str] | None) -> dict:
    """Parse --filter key=val1,val2 flags into {key: [val1, val2]}."""
    result = {}
    for item in (raw or []):
        if "=" not in item:
            console.print(f"[red]Invalid --filter '[bold]{item}[/bold]' — expected key=val1,val2[/red]")
            sys.exit(1)
        key, _, vals = item.partition("=")
        result[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    return result

# ==========================
# PIPELINE
# ==========================

def load_pipeline(model_id: str, profile: dict, force_cpu: bool = False):
    use_cpu = force_cpu or not torch.cuda.is_available()
    dtype   = torch.float32 if (use_cpu or profile.get("dtype", "float16") == "float32") else torch.float16
    offload = profile.get("cpu_offload", "none")
    device  = "cpu" if use_cpu else "cuda"

    with console.status(
        f"[bold cyan]Loading model [yellow]{model_id}[/yellow]…[/bold cyan]\n"
        "[dim](First run downloads ~4 GB — cached afterwards)[/dim]",
        spinner="dots",
    ):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None,
        )
        if not use_cpu and offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif not use_cpu and offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        if profile.get("attention_slicing"):
            pipe.enable_attention_slicing()
        if profile.get("vae_slicing"):
            pipe.enable_vae_slicing()
        if profile.get("vae_tiling"):
            pipe.enable_vae_tiling()

    gpu_label = ""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram     = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_label = f" · [dim]{gpu_name} ({vram:.1f} GB)[/dim]"
        except Exception:
            pass

    console.print(
        f"[bold green]✓ Pipeline ready[/bold green]  "
        f"[cyan]{device.upper()}[/cyan] / [cyan]{dtype}[/cyan]{gpu_label}\n"
    )
    return pipe

# ==========================
# IMAGE GENERATION
# ==========================

def next_index(output_dir: Path) -> int:
    existing = [
        int(f.stem.rsplit("_", 1)[-1])
        for f in output_dir.glob("*.png")
        if f.stem.rsplit("_", 1)[-1].isdigit()
    ]
    return max(existing, default=-1) + 1

def generate_image(pipe, prompt, negative_prompt, output_dir, index, steps, size, guidance):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=size,
        height=size,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"image_{index:04d}.png"
    image.save(filename)
    return filename

# ==========================
# COMMANDS — batch / single / list
# ==========================

def cmd_batch(args):
    theme   = load_theme(args.theme)
    filters = parse_filters(getattr(args, "filter", None))
    combos  = all_combos(theme, filters or None)
    total   = min(args.count, len(combos))
    profile_name, profile = resolve_profile(getattr(args, "profile", None))

    output_dir = Path(args.output or theme.get("output_dir", args.theme + "_output"))
    model_id   = args.model or theme.get("model", DEFAULT_MODEL)
    steps      = args.steps if args.steps is not None else profile.get("steps", DEFAULT_STEPS)
    size       = args.size if args.size is not None else profile.get("image_size", DEFAULT_SIZE)
    guidance   = args.guidance if args.guidance is not None else profile.get("guidance_scale", DEFAULT_GUIDANCE)

    console.print(Panel.fit(
        f"[bold]{theme['name']}[/bold]\n"
        f"[dim]{theme.get('description', '')}[/dim]\n\n"
        f"Theme  : [yellow]{args.theme}[/yellow]   "
        f"Profile: [magenta]{profile_name}[/magenta]   "
        f"Count: [cyan]{total}[/cyan]   "
        f"Steps: [cyan]{steps}[/cyan]   "
        f"Size: [cyan]{size}×{size}[/cyan]   "
        f"Guidance: [cyan]{guidance}[/cyan]\n"
        f"Output : [cyan]{output_dir}[/cyan]" +
        (f"\nFilters: [magenta]{filters}[/magenta]" if filters else ""),
        title="🎨 AI Art Generator — batch",
        border_style="cyan",
    ))

    pipe      = load_pipeline(model_id, profile, args.cpu)
    start_idx = next_index(output_dir)
    errors    = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating…", total=total)

        for i, combo in enumerate(combos[:total]):
            label = " · ".join(str(v)[:20] for v in list(combo.values())[:2])
            progress.update(task, description=f"[bold cyan]{label}")
            prompt = build_prompt(theme["prompt_template"], combo)

            try:
                path = generate_image(
                    pipe, prompt, theme.get("negative_prompt", ""),
                    output_dir, start_idx + i, steps, size, guidance,
                )
                console.print(f"  [green]✓[/green] [dim]{path}[/dim]")
            except Exception as exc:
                console.print(f"  [red]✗ image {i} failed:[/red] {exc}")
                errors += 1

            progress.advance(task)

    saved = total - errors
    console.print(
        f"\n[bold green]Done![/bold green] "
        f"[cyan]{saved}/{total}[/cyan] images saved to [yellow]{output_dir}/[/yellow]"
    )


def cmd_single(args):
    theme  = load_theme(args.theme)
    profile_name, profile = resolve_profile(getattr(args, "profile", None))
    output_dir = Path(args.output or theme.get("output_dir", args.theme + "_output"))
    model_id   = args.model or theme.get("model", DEFAULT_MODEL)
    steps      = args.steps if args.steps is not None else profile.get("steps", DEFAULT_STEPS)
    size       = args.size if args.size is not None else profile.get("image_size", DEFAULT_SIZE)
    guidance   = args.guidance if args.guidance is not None else profile.get("guidance_scale", DEFAULT_GUIDANCE)

    if args.prompt:
        prompt = args.prompt
        label  = prompt[:80]
    else:
        combo  = random_combo(theme)
        prompt = build_prompt(theme["prompt_template"], combo)
        label  = "  ".join(f"[cyan]{k}[/cyan]: {v}" for k, v in combo.items())

    console.print(Panel.fit(
        f"[bold]{theme['name']}[/bold] — single image\n\n"
        f"{label}\n\n"
        f"Profile: [magenta]{profile_name}[/magenta]   "
        f"Steps: [cyan]{steps}[/cyan]   "
        f"Size: [cyan]{size}×{size}[/cyan]   "
        f"Guidance: [cyan]{guidance}[/cyan]\n\n"
        f"[dim italic]{prompt[:120]}…[/dim italic]",
        title="🎨 AI Art Generator — single",
        border_style="cyan",
    ))

    pipe = load_pipeline(model_id, profile, args.cpu)
    idx  = next_index(output_dir)

    with console.status("[bold cyan]Generating image…[/bold cyan]", spinner="dots"):
        path = generate_image(
            pipe, prompt, theme.get("negative_prompt", ""),
            output_dir, idx, steps, size, guidance,
        )

    console.print(f"\n[bold green]✓ Saved:[/bold green] [yellow]{path}[/yellow]")


def cmd_list(args):
    theme   = load_theme(args.theme)
    filters = parse_filters(getattr(args, "filter", None))
    combos  = all_combos(theme, filters or None)
    total   = min(args.count, len(combos)) if args.count else len(combos)

    keys = list(theme["variations"].keys())
    table = Table(
        "#", *keys,
        title=f"🎨 [bold]{theme['name']}[/bold]  [dim]{total} of {len(combos)} combos[/dim]",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_lines=False,
    )
    for i, combo in enumerate(combos[:total]):
        table.add_row(str(i + 1), *combo.values())

    console.print(table)
    if filters:
        console.print(f"[dim]Active filters: {filters}[/dim]")
    console.print(
        f"\n[dim]Run [bold]batch --theme {args.theme} --count {total}[/bold] to generate these.[/dim]"
    )

# ==========================
# COMMANDS — themes
# ==========================

def cmd_themes(args):
    sub = args.themes_command

    if sub == "list":
        names = list_theme_names()
        if not names:
            console.print("[yellow]No themes found.[/yellow] Run [bold]themes new[/bold] to create one.")
            return
        table = Table("#", "Slug", "Name", "Description", "Variations",
                      box=box.SIMPLE_HEAD, header_style="bold cyan")
        for i, name in enumerate(names, 1):
            try:
                t = load_theme(name)
                var_summary = ", ".join(
                    f"{k} ({len(v)})" for k, v in t.get("variations", {}).items()
                )
                table.add_row(str(i), name, t.get("name", ""), t.get("description", ""), var_summary)
            except Exception:
                table.add_row(str(i), name, "[red]invalid JSON[/red]", "", "")
        console.print(table)

    elif sub == "show":
        theme = load_theme(args.name)
        console.print(Panel(
            f"[bold]{theme['name']}[/bold]\n"
            f"[dim]{theme.get('description', '')}[/dim]\n\n"
            f"[cyan]Model:[/cyan]  {theme.get('model', DEFAULT_MODEL)}\n"
            f"[cyan]Output:[/cyan] {theme.get('output_dir', args.name + '_output')}\n\n"
            f"[cyan]Prompt template:[/cyan]\n[italic]{theme['prompt_template']}[/italic]\n\n"
            f"[cyan]Negative prompt:[/cyan]\n[dim italic]{theme.get('negative_prompt', '')}[/dim italic]",
            title=f"Theme: {args.name}",
            border_style="cyan",
        ))
        for key, values in theme.get("variations", {}).items():
            table = Table(key, box=box.MINIMAL, header_style="bold yellow", show_header=True)
            for v in values:
                table.add_row(v)
            console.print(table)

    elif sub == "new":
        _interactive_new_theme()

    elif sub == "edit":
        path = THEMES_DIR / f"{args.name}.json"
        if not path.exists():
            console.print(f"[red]Theme '[bold]{args.name}[/bold]' not found.[/red]")
            sys.exit(1)
        editor = os.environ.get("EDITOR", "notepad")
        console.print(f"Opening [yellow]{path}[/yellow] in [cyan]{editor}[/cyan]…")
        subprocess.run([editor, str(path)])

    elif sub == "delete":
        path = THEMES_DIR / f"{args.name}.json"
        if not path.exists():
            console.print(f"[red]Theme '[bold]{args.name}[/bold]' not found.[/red]")
            sys.exit(1)
        if Confirm.ask(f"[yellow]Delete theme '[bold]{args.name}[/bold]'?[/yellow]"):
            path.unlink()
            console.print(f"[green]Deleted.[/green]")

def cmd_machine(args):
    sub = args.machine_command

    if sub == "detect":
        name = detect_profile_name()
        profiles = load_profiles()
        profile = profiles[name]
        mc = {"profile": name}
        if torch.cuda.is_available():
            mc["gpu"] = torch.cuda.get_device_name(0)
            mc["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        else:
            mc["gpu"] = "CPU only"
        save_machine_config(mc)

        console.print(Panel.fit(
            f"[bold green]Detected profile: [cyan]{name}[/cyan][/bold green]\n"
            f"{profile['label']}\n\n"
            f"GPU   : [cyan]{mc['gpu']}[/cyan]\n"
            f"VRAM  : [cyan]{mc.get('vram_gb', 'n/a')} GB[/cyan]\n\n"
            + "\n".join(f"  {k:<20} [yellow]{v}[/yellow]" for k, v in profile.items() if k != "label"),
            title="🖥️  Machine Profile",
            border_style="green",
        ))
        console.print(f"[dim]Saved to [yellow]{MACHINE_FILE}[/yellow][/dim]")

    elif sub == "show":
        mc = load_machine_config()
        profile_name, profile = resolve_profile(None)
        console.print(Panel.fit(
            f"Profile : [bold cyan]{profile_name}[/bold cyan]  [dim]{profile.get('label', '')}[/dim]\n"
            f"GPU     : [cyan]{mc.get('gpu', 'unknown')}[/cyan]\n"
            f"VRAM    : [cyan]{mc.get('vram_gb', 'n/a')} GB[/cyan]\n\n"
            + "\n".join(f"  {k:<20} [yellow]{v}[/yellow]" for k, v in profile.items() if k != "label"),
            title="🖥️  Current Machine Config",
            border_style="cyan",
        ))
        if not MACHINE_FILE.exists():
            console.print("[dim yellow]No machine.json found — using auto-detected profile.[/dim yellow]")

    elif sub == "set":
        profiles = load_profiles()
        if args.name not in profiles:
            console.print(f"[red]Unknown profile '[bold]{args.name}[/bold]'.[/red]")
            console.print(f"Available: {', '.join(profiles)}")
            sys.exit(1)
        mc = load_machine_config()
        mc["profile"] = args.name
        save_machine_config(mc)
        console.print(f"[bold green]✓ Profile set to '[cyan]{args.name}[/cyan]'[/bold green]")
        console.print(f"[dim]Saved to {MACHINE_FILE}[/dim]")

    elif sub == "profiles":
        profiles = load_profiles()
        table = Table("#", "Name", "Label", "Size", "Steps", "dtype", "Offload",
                      box=box.SIMPLE_HEAD, header_style="bold cyan")
        mc_profile = load_machine_config().get("profile") or detect_profile_name()
        for i, (name, p) in enumerate(profiles.items(), 1):
            marker = " ◄ current" if name == mc_profile else ""
            table.add_row(
                str(i), name + marker, p.get("label", ""),
                str(p.get("image_size", "")), str(p.get("steps", "")),
                p.get("dtype", ""), p.get("cpu_offload", "none"),
            )
        console.print(table)


def _interactive_new_theme():
    """Walk the user through creating a new theme interactively."""
    console.print(Panel(
        "Answer the prompts below to build your theme.\n"
        "The [bold]prompt template[/bold] is a Python format string — use [cyan]{key}[/cyan] "
        "placeholders matching your variation keys.\n\n"
        "Example template:\n"
        "[italic]retro poster of a {subject} in {environment}, {style}, {mood}[/italic]",
        title="✨ Create new theme",
        border_style="cyan",
    ))

    display_name = Prompt.ask("[cyan]Theme display name[/cyan]")
    slug         = Prompt.ask("[cyan]Slug (filename, no spaces)[/cyan]", default=theme_slug(display_name))
    description  = Prompt.ask("[cyan]Short description[/cyan]", default="")
    output_dir   = Prompt.ask("[cyan]Output folder[/cyan]", default=slug + "_output")
    model        = Prompt.ask("[cyan]HuggingFace model ID[/cyan]", default=DEFAULT_MODEL)

    console.print("\n[bold]Define variation groups[/bold] (e.g. key=subject, values=knight,wizard,rogue)")
    variations = {}
    while True:
        key = Prompt.ask("[cyan]Variation key[/cyan] (or leave blank to finish)", default="")
        if not key:
            if not variations:
                console.print("[yellow]You need at least one variation.[/yellow]")
                continue
            break
        raw_vals = Prompt.ask(f"  Values for [yellow]{key}[/yellow] (comma-separated)")
        variations[key] = [v.strip() for v in raw_vals.split(",") if v.strip()]

    keys_hint = ", ".join(f"{{{k}}}" for k in variations)
    console.print(f"\n[dim]Available placeholders: {keys_hint}[/dim]")
    template = Prompt.ask("[cyan]Prompt template[/cyan]")

    neg_default = CYCLING_THEME["negative_prompt"]
    neg_prompt  = Prompt.ask("[cyan]Negative prompt[/cyan]", default=neg_default)

    theme = {
        "name":            display_name,
        "description":     description,
        "output_dir":      output_dir,
        "model":           model,
        "prompt_template": template,
        "negative_prompt": neg_prompt,
        "variations":      variations,
    }

    path = save_theme(slug, theme)
    combo_count = 1
    for v in variations.values():
        combo_count *= len(v)

    console.print(
        f"\n[bold green]✓ Theme saved:[/bold green] [yellow]{path}[/yellow]\n"
        f"  [cyan]{combo_count}[/cyan] total prompt combinations\n\n"
        f"  [dim]python generator.py list --theme {slug}[/dim]\n"
        f"  [dim]python generator.py batch --theme {slug}[/dim]"
    )

# ==========================
# ARGUMENT PARSER
# ==========================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="generator",
        description="AI Art Generator — local Stable Diffusion with themes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python generator.py themes list\n"
            "  python generator.py themes new\n"
            "  python generator.py themes edit cycling\n"
            "  python generator.py batch --theme cycling\n"
            "  python generator.py batch --theme cycling --count 10 --filter subject=\"road cyclist,sprinter\"\n"
            "  python generator.py single --theme cycling\n"
            "  python generator.py single --theme cycling --prompt \"custom prompt\"\n"
            "  python generator.py list --theme cycling\n"
        ),
    )

    # Shared generation options
    gen_shared = argparse.ArgumentParser(add_help=False)
    gen_shared.add_argument("--theme",    default=DEFAULT_THEME, metavar="NAME",     help=f"theme to use (default: {DEFAULT_THEME})")
    gen_shared.add_argument("--model",    default=None,          metavar="MODEL_ID", help="override the theme's HuggingFace model")
    gen_shared.add_argument("--output",   default=None,          metavar="DIR",      help="override the theme's output directory")
    gen_shared.add_argument("--steps",    default=None, type=int,   help="inference steps (default: from profile)")
    gen_shared.add_argument("--size",     default=None, type=int,   help="image size in px (default: from profile)")
    gen_shared.add_argument("--guidance", default=None, type=float, help="guidance scale (default: from profile)")
    gen_shared.add_argument("--profile",  default=None, metavar="NAME", help="machine profile to use (auto-detect if omitted)")
    gen_shared.add_argument("--cpu",      action="store_true",                       help="force CPU mode")

    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # batch
    p_batch = sub.add_parser("batch", parents=[gen_shared],
        help="generate a batch of images from all prompt combinations")
    p_batch.add_argument("--count", default=DEFAULT_COUNT, type=int, metavar="N",
                         help=f"max images to generate (default: {DEFAULT_COUNT})")
    p_batch.add_argument("--filter", nargs="*", metavar="KEY=VAL1,VAL2",
                         help="narrow one or more variation keys, e.g. --filter subject=\"road cyclist,sprinter\"")

    # single
    p_single = sub.add_parser("single", parents=[gen_shared],
        help="generate one image (random variation or --prompt)")
    p_single.add_argument("--prompt", default=None, metavar="TEXT",
                          help="custom prompt (omit to use a random variation)")

    # list
    p_list = sub.add_parser("list",
        help="preview all prompt combinations as a table (no generation)")
    p_list.add_argument("--theme",  default=DEFAULT_THEME, metavar="NAME", help=f"theme to use (default: {DEFAULT_THEME})")
    p_list.add_argument("--count",  default=None, type=int, metavar="N",   help="limit rows shown (default: all)")
    p_list.add_argument("--filter", nargs="*", metavar="KEY=VAL1,VAL2",    help="narrow variation keys")

    # themes
    p_themes = sub.add_parser("themes", help="manage themes")
    themes_sub = p_themes.add_subparsers(dest="themes_command", metavar="subcommand")
    themes_sub.required = True
    themes_sub.add_parser("list", help="list all available themes")

    p_show = themes_sub.add_parser("show",   help="show full details of a theme")
    p_show.add_argument("name", help="theme slug")

    themes_sub.add_parser("new", help="create a new theme interactively")

    p_edit = themes_sub.add_parser("edit",   help="open a theme JSON file in your editor")
    p_edit.add_argument("name", help="theme slug")

    p_del = themes_sub.add_parser("delete", help="delete a theme")
    p_del.add_argument("name", help="theme slug")

    p_machine = sub.add_parser("machine", help="configure machine-specific optimization profile")
    machine_sub = p_machine.add_subparsers(dest="machine_command", metavar="subcommand")
    machine_sub.required = True
    machine_sub.add_parser("detect", help="auto-detect GPU and save recommended profile to machine.json")
    machine_sub.add_parser("show", help="show current machine.json and effective profile settings")
    p_mset = machine_sub.add_parser("set", help="manually set a profile in machine.json")
    p_mset.add_argument("name", help="profile name (high/medium/low/minimal/cpu)")
    machine_sub.add_parser("profiles", help="list all available profiles and their settings")

    return parser

# ==========================
# ENTRY POINT
# ==========================

def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "batch":  cmd_batch,
        "single": cmd_single,
        "list":   cmd_list,
        "themes": cmd_themes,
        "machine": cmd_machine,
    }
    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    main()
