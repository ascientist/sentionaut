UV ?= uv
ENV_FLAGS ?= --extra dev
DATASET ?= data/axon_map.h5
SAMPLES ?= 128
WORLD_DATASET ?= data/world.h5
EPISODES ?= 256
SEQ_LEN ?= 16
MODEL ?= axonmap
OUTDIR ?= artifacts

.PHONY: setup dataset world demo animate train ablate test lint format clean

setup:
	$(UV) sync $(ENV_FLAGS)

dataset:
	$(UV) run sentionaut-generate --output $(DATASET) --samples $(SAMPLES)

world:
	$(UV) run sentionaut-world --output $(WORLD_DATASET) --episodes $(EPISODES) --sequence-length $(SEQ_LEN)

demo:
	$(UV) run streamlit run src/sentionaut/demo_app.py

animate:
	$(UV) run sentionaut-animate --model $(MODEL) --outdir $(OUTDIR)

train:
	$(UV) run sentionaut-train train --dataset $(WORLD_DATASET) --config configs/train.yaml

ablate:
	$(UV) run sentionaut-train ablate --dataset $(WORLD_DATASET) --config configs/ablation.yaml

test:
	$(UV) run pytest -m "not slow"

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format --check src tests

clean:
	rm -f $(DATASET) $(WORLD_DATASET)
