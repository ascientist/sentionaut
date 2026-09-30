UV ?= uv
ENV_FLAGS ?= --extra dev
DATASET ?= data/axon_map.h5
SAMPLES ?= 128
WORLD_DATASET ?= data/world.h5
EPISODES ?= 256
SEQ_LEN ?= 16
MODEL ?= axonmap
OUTDIR ?= artifacts

.PHONY: setup setup-jax dataset world demo animate demos train ablate neurostim neurostim-percept neurostim-canonical test lint format docs docs-serve clean

setup:
	$(UV) sync $(ENV_FLAGS)

setup-jax:
	$(UV) sync $(ENV_FLAGS) --extra jax

dataset:
	$(UV) run sentionaut-generate --output $(DATASET) --samples $(SAMPLES)

world:
	$(UV) run sentionaut-world --output $(WORLD_DATASET) --episodes $(EPISODES) --sequence-length $(SEQ_LEN)

demo:
	$(UV) run streamlit run src/sentionaut/demo_app.py

animate:
	$(UV) run sentionaut-animate --model $(MODEL) --outdir $(OUTDIR)

demos:
	$(UV) run sentionaut-animate --model all --scenario sequence --outdir docs/assets/demos
	$(UV) run sentionaut-animate --model dynaphos_axonmap --outdir artifacts
	cp artifacts/dynaphos_axonmap.gif docs/assets/demos/dynaphos_axonmap_sweep.gif
	$(UV) run python examples/dynaphos_axonmap_figures.py

train:
	$(UV) run sentionaut-train train --dataset $(WORLD_DATASET) --config configs/train.yaml

ablate:
	$(UV) run sentionaut-train ablate --dataset $(WORLD_DATASET) --config configs/ablation.yaml

neurostim:
	$(UV) run python examples/neurostim_tutorial.py

neurostim-percept:
	$(UV) run python examples/neurostim_percept_tutorial.py

neurostim-canonical:
	$(UV) run python examples/neurostim_canonical_tutorial.py

test:
	$(UV) run pytest -m "not slow"

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format --check src tests

docs:
	$(UV) run zensical build

docs-serve:
	$(UV) run zensical serve

clean:
	rm -f $(DATASET) $(WORLD_DATASET)
	rm -rf site
