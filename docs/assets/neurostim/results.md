| setting | policy | return | steady-state error | violation rate | charge |
| --- | --- | ---: | ---: | ---: | ---: |
| NeuroStim-Easy-v0 | zero | -406.0 ± 1.7 | 2.026 | 0.000 | 0.00 |
| NeuroStim-Easy-v0 | random | -448.7 ± 5.7 | 2.193 | 0.000 | 0.99 |
| NeuroStim-Easy-v0 | bandit (UCB) | -257.4 ± 12.0 | 1.248 | 0.000 | 0.79 |
| NeuroStim-Easy-v0 | nominal PI | -3.2 ± 0.3 | 0.001 | 0.000 | 1.04 |
| NeuroStim-Easy-v0 | online sysID | -47.7 ± 1.5 | 0.008 | 0.000 | 0.94 |
| NeuroStim-Easy-v0 | oracle greedy * | -3.1 ± 0.2 | 0.002 | 0.000 | 1.01 |
| NeuroStim-v0 | zero | -406.2 ± 1.5 | 2.032 | 0.000 | 0.00 |
| NeuroStim-v0 | random | -438.3 ± 4.9 | 2.125 | 0.000 | 0.99 |
| NeuroStim-v0 | bandit (UCB) | -197.8 ± 12.0 | 0.897 | 0.000 | 0.79 |
| NeuroStim-v0 | nominal PI | -911.8 ± 114.0 | 4.382 | 0.029 | 2.00 |
| NeuroStim-v0 | online sysID | -88.3 ± 8.4 | 0.235 | 0.203 | 1.58 |
| NeuroStim-v0 | oracle greedy * | -38.1 ± 6.8 | 0.209 | 0.000 | 1.73 |
| NeuroStim-Hard-v0 | zero | -404.1 ± 2.1 | 2.009 | 0.000 | 0.00 |
| NeuroStim-Hard-v0 | random | -439.1 ± 7.4 | 2.162 | 0.000 | 0.99 |
| NeuroStim-Hard-v0 | bandit (UCB) | -588.2 ± 39.7 | 2.583 | 0.000 | 0.66 |
| NeuroStim-Hard-v0 | nominal PI | -1176.2 ± 135.5 | 5.828 | 0.047 | 2.00 |
| NeuroStim-Hard-v0 | online sysID | -508.7 ± 56.5 | 1.649 | 0.077 | 1.51 |
| NeuroStim-Hard-v0 | oracle greedy * | -40.3 ± 7.3 | 0.188 | 0.000 | 1.73 |
| v0, fixed B | nominal PI | -27.6 ± 0.5 | 0.124 | 0.056 | 1.97 |
| v0, fixed B | online sysID | -75.5 ± 4.9 | 0.167 | 0.553 | 1.79 |
| v0, fixed B | oracle greedy * | -22.8 ± 0.4 | 0.116 | 0.000 | 1.97 |
| v0, fixed B | PPO (fixed patient) | -19.4 ± 0.3 | 0.090 | 0.946 | 2.56 |
| v0, fixed B | PPO-Lagrangian (fixed patient) | -44.3 ± 0.5 | 0.227 | 0.000 | 1.07 |
| NeuroStim-v0 | PPO (random patients) | -518.9 ± 65.3 | 2.270 | 0.850 | 2.64 |
| NeuroStim-v0 | PPO + true B (random patients) * | -157.0 ± 16.2 | 0.747 | 0.532 | 1.80 |
| v0, fixed B | DAgger (fixed patient) | -22.7 ± 0.4 | 0.116 | 0.000 | 1.94 |
| NeuroStim-v0 | DAgger (random patients) | -298.4 ± 12.9 | 1.497 | 0.003 | 0.88 |
| NeuroStim-v0 | DAgger + calibration (random patients) | -168.9 ± 26.2 | 0.832 | 0.016 | 1.24 |
