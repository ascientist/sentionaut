# References

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
