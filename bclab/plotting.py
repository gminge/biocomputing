"""画图辅助函数：统一实验图的风格。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bclab.neurons import SimResult

# 让中文标题/图例正常显示；找不到 CJK 字体时回退到 DejaVu Sans。
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SPIKE_TOP_MARGIN = 3.0  # 脉冲标记画在阈值上方多少 mV


def plot_voltage_trace(
    result: SimResult,
    ax: plt.Axes | None = None,
    current: np.ndarray | None = None,
    title: str | None = None,
) -> plt.Axes:
    """画膜电位轨迹：阈值虚线 + 脉冲时刻标记。

    若传入 current（每步电流数组），则在共享 x 轴上方加一个电流子图。
    调用方负责 plt.show() 或 fig.savefig()。
    """
    if ax is None:
        if current is not None:
            fig, (ax_i, ax) = plt.subplots(
                2, 1, figsize=(9, 5), sharex=True,
                gridspec_kw={"height_ratios": [1, 3]},
            )
            ax_i.plot(result.t, current, color="tab:orange", lw=1)
            ax_i.set_ylabel("I (nA)")
            ax_i.spines[["top", "right"]].set_visible(False)
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    p = result.params
    ax.plot(result.t, result.v, color="tab:blue", lw=1.2, label="V(t)")
    ax.axhline(p.v_th, ls="--", color="tab:red", lw=1,
               label=f"threshold {p.v_th} mV")
    ax.axhline(p.e_l, ls=":", color="gray", lw=1,
               label=f"rest {p.e_l} mV")

    # 用竖线/点标记每次发放
    marker_y = p.v_th + SPIKE_TOP_MARGIN
    ax.plot(result.spikes, np.full_like(result.spikes, marker_y),
            "|", color="tab:red", ms=10)

    ax.set_ylim(min(result.v.min() - 2, p.e_l - 5), marker_y + 8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("V (mV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return ax
