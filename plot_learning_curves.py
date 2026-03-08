import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_returns(log_path):
    text = Path(log_path).read_text()
    pattern = r"Episode done return=([\-0-9\.]+)"
    returns = [float(x) for x in re.findall(pattern, text)]
    return np.array(returns, dtype=float)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])


def main():
    # logs to compare
    baseline_log = "ppo_baseline.txt"
    expert_log = "ppo_expert_shaping.txt"
    bls_log = "ppo_bls.txt"  # make sure you ran this

    baseline_ret = extract_returns(baseline_log)
    expert_ret = extract_returns(expert_log)
    bls_ret = extract_returns(bls_log)

    print(f"Baseline episodes: {len(baseline_ret)}")
    print(f"Expert+shaping episodes: {len(expert_ret)}")
    print(f"BLS critic episodes: {len(bls_ret)}")

    print(f"Baseline mean return: {baseline_ret.mean():.2f}")
    print(f"Expert+shaping mean return: {expert_ret.mean():.2f}")
    print(f"BLS critic mean return: {bls_ret.mean():.2f}")

    ep_baseline = np.arange(1, len(baseline_ret) + 1)
    ep_expert = np.arange(1, len(expert_ret) + 1)
    ep_bls = np.arange(1, len(bls_ret) + 1)

    ma_baseline = moving_average(baseline_ret, window=20)
    ma_expert = moving_average(expert_ret, window=20)
    ma_bls = moving_average(bls_ret, window=20)

    plt.figure()
    # raw curves (faint)
    plt.plot(ep_baseline, baseline_ret, alpha=0.2, label="Baseline PPO (raw)")
    plt.plot(ep_expert, expert_ret, alpha=0.2, label="Expert+Shaping (raw)")
    plt.plot(ep_bls, bls_ret, alpha=0.2, label="BLS critic (raw)")
    # smoothed curves
    plt.plot(ep_baseline, ma_baseline, linewidth=2.0, label="Baseline PPO (MA-20)")
    plt.plot(ep_expert, ma_expert, linewidth=2.0, label="Expert+Shaping (MA-20)")
    plt.plot(ep_bls, ma_bls, linewidth=2.0, label="BLS critic (MA-20)")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("LunarLander-v2: PPO vs Expert+Shaping vs BLS Critic")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig("learning_curves_3way.png", dpi=200)
    print("Saved figure to learning_curves_3way.png")
    plt.show()


if __name__ == "__main__":
    main()
