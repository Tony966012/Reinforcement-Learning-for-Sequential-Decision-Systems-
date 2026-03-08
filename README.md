// =========================
// Repo: drl5100-final
// =========================
// This canvas contains a complete, runnable project scaffold with baseline PPO for LunarLander,
// hooks for expert demonstrations + reward shaping, and a Broad-Learning-System (BLS) critic module.
// Copy these files into a local folder with the same layout.


// -------------------------
// File: README.md
// -------------------------
# DRL5100 Final Project – Efficient RL for LunarLander


This project explores improving PPO on the LunarLander-v2 task via (1) expert-demo assisted learning and reward shaping, and (2) a Broad-Learning-System (BLS) value approximator ("broad critic").


## Quickstart
```bash
# 1) Create env (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


# 2) Train a baseline PPO agent
python src/train.py --algo ppo --env LunarLander-v2 --total-steps 200000
# 3) Train with expert+shaping
python src/train.py --algo ppo --env LunarLander-v2 --total-steps 200000 \
--use-expert --expert-path data/expert_lander.pkl --use-shaping


# 4) Train with BLS critic
python src/train.py --algo ppo --env LunarLander-v2 --total-steps 200000 \
--critic bls


# 5) Evaluate
python src/eval.py --ckpt runs/ppo_LunarLander-v2/latest.pt --episodes 20
```


## Project Layout
```
├─ README.md
├─ requirements.txt
├─ configs/
│ ├─ ppo.yaml
│ └─ ddpg.yaml
├─ data/
│ └─ expert_lander.pkl # (optional) saved expert trajectories
├─ runs/ # checkpoints & logs
├─ src/
│ ├─ algos/
│ │ ├─ ppo.py
│ │ ├─ ddpg.py
│ │ └─ buffers.py
│ ├─ critics/
│ │ ├─ bls.py
│ │ └─ nn_value.py
│ ├─ env/
│ │ ├─ wrappers.py
│ │ └─ shaping.py
│ ├─ expert/
│ │ └─ heuristic_lander.py
│ ├─ nets/
│ │ ├─ policy_mlp.py
│ │ └─ q_mlp.py
│ ├─ utils/
│ │ ├─ serial.py
│ │ ├─ schedules.py
│ │ └─ torch_utils.py
│ ├─ train.py
│ └─ eval.py
└─ REPORT_outline.md
```


## Notes
- Works on CPU or single GPU. For WashU cluster, use `--device cuda` and save/restore checkpoints from `runs/`.
- BLS critic uses ridge regression on a broad feature map. It can replace the NN critic in PPO.
- Reward shaping is optional and designed to be potential-based (keeps optimal policy).


---