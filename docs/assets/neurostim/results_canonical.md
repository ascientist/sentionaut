| level | controller | median error | mean error | failure rate |
| --- | --- | ---: | ---: | ---: |
| L0 known patient | zero | 0.5874 | 0.8140 | 0.00 |
| L0 known patient | population LQG | 0.0149 | 0.0180 | 0.00 |
| L0 known patient | adaptive (RLS + CE) | 0.0251 | 0.6663 | 0.03 |
| L0 known patient | oracle LQG * | 0.0149 | 0.0180 | 0.00 |
| L1 random patients | zero | 0.5738 | 0.8164 | 0.00 |
| L1 random patients | population LQG | 0.3661 | 9.8554 | 0.40 |
| L1 random patients | adaptive (RLS + CE) | 0.0252 | 0.0476 | 0.00 |
| L1 random patients | oracle LQG * | 0.0133 | 0.0324 | 0.00 |
| L2 + partial observation | zero | 0.5630 | 0.7915 | 0.00 |
| L2 + partial observation | population LQG | 0.5692 | 10.4198 | 0.53 |
| L2 + partial observation | adaptive (RLS + CE) | 0.0405 | 0.3486 | 0.07 |
| L2 + partial observation | oracle LQG * | 0.0149 | 0.0332 | 0.00 |
| L3 + drift | zero | 0.5972 | 0.8130 | 0.00 |
| L3 + drift | population LQG | 5.4302 | 17.1547 | 0.70 |
| L3 + drift | adaptive (RLS + CE) | 0.1357 | 4.1422 | 0.30 |
| L3 + drift | oracle LQG * | 0.0158 | 0.0330 | 0.00 |
