// -------------------------
// File: REPORT_outline.md
// -------------------------
# Report Outline (Template)


## 1. Introduction
- Problem statement: LunarLander control, delayed rewards, stability.
- Goal: improve sample-efficiency and stability vs. baseline PPO.


## 2. Related Work (2024+)
- Ahmad et al. 2024 (hybrid offline-online PPO + TWTL shaping) — summarize contributions and relevance.
- Thalagala et al. 2025 (Broad Critic Deep Actor) — summarize and relevance.


## 3. Methods
- Baseline PPO (brief equations).
- Expert-mixing and potential-based shaping (formulas; show shaped reward preserves optimality).
- BLS critic (feature map, ridge regression closed form).
- Combined method.


## 4. Experiments
- Environments, seeds, total steps, hardware.
- Ablations: baseline / +expert+shaping / +BLS / combined.
- Metrics: average return, success rate, sample-efficiency (steps to 200 reward).


## 5. Results
- Learning curves, tables of returns, variance analysis.
- Discussion.


## 6. Conclusion & Future Work


## References