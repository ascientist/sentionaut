# References

[Home](index.md) · [Getting started](getting-started.md) · [User guide](user-guide.md) · [Nomenclature](nomenclature.md) · [Models](models/index.md) · [References](references.md)

Full citations for papers linked from the [model pages](models/index.md).

## Beyeler et al. 2019

Michael Beyeler, Devyani Nanduri, James D. Weiland, Ariel Rokem, Geoffrey M. Boynton, Ione Fine.
*A model of ganglion axon pathways accounts for percepts elicited by retinal implants.*
Scientific Reports 9, 9199 (2019).
[doi:10.1038/s41598-019-45416-4](https://doi.org/10.1038/s41598-019-45416-4)

Source of the axon-map spatial model and the Gaussian scoreboard baseline.

## Granley & Beyeler 2021

Jacob Granley, Michael Beyeler.
*A computational model of phosphene appearance for epiretinal prostheses.*
IEEE EMBC 2021.
[doi:10.1109/EMBC46164.2021.9629663](https://doi.org/10.1109/EMBC46164.2021.9629663)

Biphasic pulse-parameter effects (brightness, size, streak) on top of the axon map.
Implemented here as `BiphasicAxonMapTorch`.

## Polimeni et al. 2006

Jonathan R. Polimeni, Mukund Balasubramanian, Eric L. Schwartz.
*Multi-area visuotopic map complexes in macaque striate and extra-striate cortex.*
Vision Research 46(20), 3336–3359 (2006).
[doi:10.1016/j.visres.2006.03.006](https://doi.org/10.1016/j.visres.2006.03.006)

Wedge-dipole visuotopic map used for cortical electrode ↔ visual-field
coordinates (scoreboard and dynaphos).

## van der Grinten et al. 2024

Maureen van der Grinten, Jaap de Ruyter van Steveninck, Antonio Lozano, et al.
*Towards biologically plausible phosphene simulation for the differentiable
optimization of visual cortical prostheses.*
eLife 13, e85812 (2024).
[doi:10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812)

Dynaphos cortical phosphene model (temporal charge / activation dynamics).
Implemented here as `DynaphosTorch`.

## Ha and Schmidhuber 2018

David Ha, Jürgen Schmidhuber.
*World Models.*
arXiv:1803.10122 (2018).
[doi:10.5281/zenodo.1207631](https://doi.org/10.5281/zenodo.1207631)

The `f(s_t, a_t) → s_{t+1}` interface used by every percept model here.

## Dosovitskiy et al. 2020

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.*
ICLR 2021.
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

Patch tokens. The axon-map student uses parameter-free 2D sin/cos positions
because the percept grid size changes.

## Jaegle et al. 2021

Andrew Jaegle, Felix Gimeno, Andrew Brock, et al.
*Perceiver: General Perception with Iterative Attention.*
ICML 2021.
[arXiv:2103.03206](https://arxiv.org/abs/2103.03206)

Cross-attention over a variable set. Electrodes are that set.

## Peebles and Xie 2023

William Peebles, Saining Xie.
*Scalable Diffusion Models with Transformers.*
ICCV 2023.
[arXiv:2212.09748](https://arxiv.org/abs/2212.09748)

Adaptive layer norm, used here to condition on `rho` and `axlambda`. The
student is not a diffusion model.

## Bruce et al. 2024

Jake Bruce, Michael Dennis, Ashley Edwards, et al.
*Genie: Generative Interactive Environments.*
ICML 2024.
[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)

The spatiotemporal transformer block of `AxonVideoWorld`: spatial attention
inside a frame, causal temporal attention across frames. Its latent action
model and VQ tokens are not used, because the stimulation is known and the
percept is continuous.

## Valevski et al. 2024

Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter.
*Diffusion Models Are Real-Time Game Engines.*
ICLR 2025.
[arXiv:2408.14837](https://arxiv.org/abs/2408.14837)

Noise on context frames during training, so an autoregressive model tolerates
its own errors when it runs free.
