| setting | policy | pixel MSE | classifier accuracy | charge | h > h_max |
| --- | --- | ---: | ---: | ---: | ---: |
| any | ceiling D(E(y)) (latent reached exactly) | 0.0106 | 0.950 | 0.00 | 0.000 |
| fixed patient | no stimulation | 0.0738 | 0.089 | 0.00 | 0.000 |
| fixed patient | policy via world model (round 1) | 0.0156 | 0.934 | 4.42 | 0.000 |
| fixed patient | policy via world model (round 2) | 0.0143 | 0.935 | 4.35 | 0.000 |
| fixed patient | policy via true brain * | 0.0139 | 0.929 | 3.95 | 0.000 |
| random patients | no stimulation | 0.0738 | 0.089 | 0.00 | 0.000 |
| random patients | fixed-patient policy (no personalisation) | 0.1020 | 0.103 | 4.35 | 0.000 |
| random patients | world model + calibration context | 0.0470 | 0.601 | 2.51 | 0.000 |
| random patients | true brain + calibration context * | 0.0417 | 0.664 | 1.91 | 0.000 |
