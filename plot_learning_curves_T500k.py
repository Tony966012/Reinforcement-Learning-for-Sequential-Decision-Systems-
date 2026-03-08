import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_returns(log_path):
    text = Path(log_path).read_text()
    pattern = r"Episode done return=([\-0-9\.]+)"
    vals = [float(x) for x in re.findall(pattern, text)]
    return np.array(vals, dtype=float)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])


def stack_seeds(files):
    runs = [extract_returns(f) for f in files]
    min_len = min(len(r) for r in runs)
    arr = np.stack([r[:min_len] for r in runs], axis=0)  # [S, T]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean, std, min_len


def main():
    # adjust names if you changed them
    base_files = [
        "ppo_base_T500k_s0.txt",
        "ppo_base_T500k_s1.txt",
        "ppo_base_T500k_s2.txt",
    ]
    expshape_files = [
        "ppo_expshape_T500k_s0.txt",
        "ppo_expshape_T500k_s1.txt",
        "ppo_expshape_T500k_s2.txt",
    ]
    bls_files = [
        "ppo_bls_T500k_s0.txt",
        "ppo_bls_T500k_s1.txt",
        "ppo_bls_T500k_s2.txt",
    ]

    base_mean, base_std, T_base = stack_seeds(base_files)
    exp_mean, exp_std, T_exp = stack_seeds(expshape_files)
    bls_mean, bls_std, T_bls = stack_seeds(bls_files)

    print(f"Baseline episodes (min across seeds): {T_base}")
    print(f"Expert+shaping episodes (min across seeds): {T_exp}")
    print(f"BLS critic episodes (min across seeds): {T_bls}")

    # use min length over all methods so x-axis aligns
    T = min(T_base, T_exp, T_bls)
    x = np.arange(1, T + 1)

    base_ma = moving_average(base_mean[:T], window=20)
    exp_ma = moving_average(exp_mean[:T], window=20)
    bls_ma = moving_average(bls_mean[:T], window=20)

    plt.figure()

    # mean curves with std bands
    plt.plot(x, base_ma, linewidth=2.0, label="Baseline PPO (mean, 3 seeds)")
    plt.fill_between(
        x,
        base_mean[:T] - base_std[:T],
        base_mean[:T] + base_std[:T],
        alpha=0.15,
    )

    plt.plot(x, exp_ma, linewidth=2.0, label="Expert+Shaping (mean, 3 seeds)")
    plt.fill_between(
        x,
        exp_mean[:T] - exp_std[:T],
        exp_mean[:T] + exp_std[:T],
        alpha=0.15,
    )

    plt.plot(x, bls_ma, linewidth=2.0, label="BLS critic (mean, 3 seeds)")
    plt.fill_between(
        x,
        bls_mean[:T] - bls_std[:T],
        bls_mean[:T] + bls_std[:T],
        alpha=0.15,
    )

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("LunarLander-v2 (500k steps): PPO vs Expert+Shaping vs BLS (3 seeds)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig("learning_curves_T500k_3seeds.png", dpi=200)
    print("Saved figure to learning_curves_T500k_3seeds.png")
    plt.show()


if __name__ == "__main__":
    main()
