from pathlib import Path


DEFAULT_FIGURE_FORMATS = ["png", "pdf", "svg"]
RASTER_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff", "webp"}


def normalize_formats(formats):
    normalized = []
    for fmt in formats:
        fmt = str(fmt).strip().lower().lstrip(".")
        if fmt and fmt not in normalized:
            normalized.append(fmt)
    if not normalized:
        raise ValueError("At least one output format is required.")
    return normalized


def panel_label(index, dataset):
    return f"（{chr(ord('a') + index)}）{dataset}"


def add_panel_label_below(ax, label, y=-0.24, fontsize=10, fontweight="semibold"):
    return ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight=fontweight,
        clip_on=False,
    )


def apply_common_vector_settings(plt):
    plt.rcParams.update(
        {
            "font.family": [
                "Times New Roman",
                "SimSun",
                "SimHei",
                "Microsoft YaHei",
                "DejaVu Serif",
            ],
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def save_figure_formats(fig, output_base, formats=None, dpi=320, pad_inches=0.04):
    formats = normalize_formats(formats or DEFAULT_FIGURE_FORMATS)
    output_base = Path(output_base)
    if output_base.suffix:
        output_base = output_base.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        output_path = output_base.with_suffix(f".{fmt}")
        save_kwargs = {"facecolor": "white", "pad_inches": pad_inches}
        if fmt in RASTER_FORMATS:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)
    return saved_paths


def save_figure_paths(fig, output_paths, dpi=320, pad_inches=0.04):
    saved_paths = []
    for raw_path in output_paths:
        output_path = Path(raw_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower().lstrip(".")
        save_kwargs = {"facecolor": "white", "pad_inches": pad_inches}
        if fmt in RASTER_FORMATS:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        saved_paths.append(output_path)
    return saved_paths
