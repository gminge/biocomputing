"""实验 1.3：脉冲序列输入 —— 神经元把一串输入脉冲"整合"成自己的发放。

Phase 2 才会引入真正的突触；这里先手工构造一个等效输入：
每个输入脉冲贡献一个指数衰减的突触后电流（PSC），

    I(t) = Σ_k  w · exp(-(t - t_k) / τ_s) · Θ(t - t_k)

这正是 Phase 2 中突触模型的雏形。调节输入脉冲频率或权重 w，
可以观察"时间整合"现象：单个脉冲不足以发放，密集的一串才可以。

运行：
    python experiments/phase01_single_neuron/exp03_spike_input.py
输出：
    figures/exp03_spike_input.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from bclab import LIFNeuron
from bclab.plotting import plot_voltage_trace

DT = 0.1          # ms
DURATION = 500.0  # ms
TAU_S = 5.0       # 突触电流衰减时间常数 (ms)
WEIGHT = 2.0      # 每个输入脉冲的电流幅度 (nA)

FIG_DIR = Path(__file__).parent / "figures"


def poisson_spikes(rate_hz: float, duration_ms: float, dt: float,
                   rng: np.random.Generator) -> np.ndarray:
    """生成泊松脉冲序列，返回脉冲出现的时刻 (ms)。"""
    prob = rate_hz / 1000.0 * dt  # 每步发放概率
    mask = rng.random(int(round(duration_ms / dt))) < prob
    return np.nonzero(mask)[0] * dt


def psc_current(spike_times: np.ndarray, n_steps: int, dt: float,
                tau: float, weight: float) -> np.ndarray:
    """把脉冲序列转成指数衰减的突触后电流数组。"""
    current = np.zeros(n_steps)
    kernel_len = int(round(10 * tau / dt))          # 核只算到 10τ
    kernel = weight * np.exp(-np.arange(kernel_len) * dt / tau)
    for t_spike in spike_times:
        i0 = int(round(t_spike / dt))
        i1 = min(i0 + kernel_len, n_steps)
        current[i0:i1] += kernel[: i1 - i0]
    return current


def main() -> None:
    rng = np.random.default_rng(seed=7)
    n_steps = int(round(DURATION / DT))

    input_spikes = poisson_spikes(rate_hz=300.0, duration_ms=DURATION,
                                  dt=DT, rng=rng)
    current = psc_current(input_spikes, n_steps, DT, TAU_S, WEIGHT)

    neuron = LIFNeuron()
    result = neuron.simulate(current, dt=DT)

    # --- 画图：输入脉冲栅格 + 电流 + 膜电位 ---
    fig, (ax_raster, ax_i, ax_v) = plt.subplots(
        3, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2, 3]},
    )

    ax_raster.plot(input_spikes, np.ones_like(input_spikes), "|",
                   color="black", ms=8)
    ax_raster.set_yticks([])
    ax_raster.set_ylabel("input\nspikes")
    ax_raster.spines[["top", "right", "left"]].set_visible(False)

    ax_i.plot(result.t, current, color="tab:orange", lw=0.8)
    ax_i.set_ylabel("I (nA)")
    ax_i.spines[["top", "right"]].set_visible(False)

    plot_voltage_trace(
        result, ax=ax_v,
        title=(f"Poisson input 300 Hz, PSC τ_s={TAU_S} ms, w={WEIGHT} nA  "
               f"->  {result.n_spikes} output spikes "
               f"({result.firing_rate():.0f} Hz)"),
    )
    # 把输出脉冲也标在栅格行下方
    ax_raster.plot(result.spikes, np.zeros_like(result.spikes), "|",
                   color="tab:red", ms=10)
    ax_raster.set_ylim(-0.5, 1.5)
    ax_raster.set_yticks([0, 1])
    ax_raster.set_yticklabels(["out", "in"])

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "exp03_spike_input.png"
    fig.savefig(out, dpi=150)
    print(f"{len(input_spikes)} input spikes -> "
          f"{result.n_spikes} output spikes")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
