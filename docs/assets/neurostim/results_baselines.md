| method | family | regime A | regime B | regime C | regime D |
| --- | --- | ---: | ---: | ---: | ---: |
| zero | reference | 0.575 | 0.574 | 0.591 | 2.728 |
| Oracle LQG * | ceiling | **0.015** | 0.013 | 0.039 | 24.444 (60% fail) |
| iLQR-NMPC * | ceiling | 0.017 | **0.010** | **0.009** | **0.011** |
| MPPI * | sampling | 0.051 | 0.043 | 0.017 | 0.065 |
| LQG (nominal) | classical | **0.015** | 0.262 (40% fail) | 0.220 (25% fail) | 23.670 (70% fail) |
| PI | classical | 0.050 | 0.205 (40% fail) | 3.357 (90% fail) | 4.285 (60% fail) |
| H∞ robust | robust | **0.016** | 0.149 (30% fail) | 0.360 (45% fail) | 27.399 (75% fail) |
| Adaptive MPC (RLS) | adaptive | 0.030 (5% fail) | 0.025 | 0.025 | 86.585 (85% fail) |
| DeePC | data-driven | 0.236 (5% fail) | 0.311 (30% fail) | 0.189 | 2.149 (40% fail) |
| Koopman-MPC | learned dynamics | 0.043 (5% fail) | 0.029 (10% fail) | 0.027 | 98.550 (95% fail) |
| SAC | model-free RL | 0.073 | 0.109 (5% fail) | 0.050 | 2.720 (10% fail) |
| TD-MPC2-style | model-based RL | 0.050 | 0.128 (10% fail) | 0.070 | 0.237 (5% fail) |
| JEPA-style + MPPI | representation | 3.597 (100% fail) | 4.591 (100% fail) | 0.637 (55% fail) | 1.571 (30% fail) |
| Diffusion policy | generative | 0.055 | 0.473 (40% fail) | 0.234 (20% fail) | 0.942 |
| BC (MSE) | imitation (control) | 0.034 | 0.226 (35% fail) | 0.213 (15% fail) | 0.949 |
