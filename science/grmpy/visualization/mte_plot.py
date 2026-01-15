"""
MTE visualization module.

Design Decision: Separates plotting logic from estimation to enable
independent testing and customization of visualizations.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from grmpy.core.contracts import EstimationResult


def plot_mte_curve(
    result: EstimationResult,
    config_path: Optional[Union[str, Path]] = None,
    output_file: Optional[str] = None,
    show_confidence: bool = True,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot the Marginal Treatment Effect curve.

    Args:
        result: EstimationResult containing MTE data
        config_path: Optional config for additional settings
        output_file: Optional path to save figure
        show_confidence: Whether to show min/max bands
        figsize: Figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    quantiles = result.quantiles
    mte = result.mte

    # Main MTE curve
    ax.plot(quantiles, mte, "b-", linewidth=2, label="MTE")

    # Confidence/variation bands if available in metadata
    if show_confidence and "mte_min" in result.metadata and "mte_max" in result.metadata:
        mte_min = result.metadata["mte_min"]
        mte_max = result.metadata["mte_max"]
        ax.fill_between(
            quantiles,
            mte_min,
            mte_max,
            alpha=0.2,
            color="blue",
            label="X variation"
        )

    # Formatting
    ax.set_xlabel("Unobserved Resistance ($u_D$)", fontsize=12)
    ax.set_ylabel("Marginal Treatment Effect", fontsize=12)
    ax.set_title("Marginal Treatment Effect Curve", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
