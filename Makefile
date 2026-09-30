UV ?= uv
ENV_FLAGS ?= --extra dev
DATASET ?= data/axon_map.h5
SAMPLES ?= 128
WORLD_DATASET ?= data/world.h5
EPISODES ?= 256
SEQ_LEN ?= 16
MODEL ?= axonmap
OUTDIR ?= artifacts
AXON_DATASET ?= data/axon_full.h5
AXON_CKPT ?= data/axon_world.pt
AXON_EPISODES ?= 512
AXON_EPOCHS ?= 50
AXON_BATCH ?= 256
AXON_DIM ?= 64
AXON_DEPTH ?= 2
AXON_WORKERS ?= 3
STREAM_DATASET ?= data/axon_stream.h5

.PHONY: setup dataset world demo animate demos train ablate axon-world axon-video neurostim neurostim-percept neurostim-canonical test lint format docs docs-serve clean

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

demos:
	$(UV) run sentionaut-animate --model all --scenario sequence --outdir docs/assets/demos

train:
	$(UV) run sentionaut-train train --dataset $(WORLD_DATASET) --config configs/train.yaml

ablate:
	$(UV) run sentionaut-train ablate --dataset $(WORLD_DATASET) --config configs/ablation.yaml

# Axon-map distillation: teacher transitions, student, then the docs figures.
axon-world:
	$(UV) run sentionaut-world --output $(AXON_DATASET) --model axonmap --episodes $(AXON_EPISODES) \
		--sequence-length 4 --silent-tail 2 --xrange -12 12 --yrange -12 12 --xystep 0.25 \
		--timing data/timing_gen.json
	$(UV) run sentionaut-distill-axon --dataset $(AXON_DATASET) --epochs $(AXON_EPOCHS) \
		--batch-size $(AXON_BATCH) --dim $(AXON_DIM) --depth $(AXON_DEPTH) --num-workers $(AXON_WORKERS) \
		--ckpt $(AXON_CKPT) --timing data/timing_train.json
	$(UV) run sentionaut-axon-report --dataset $(AXON_DATASET) --ckpt $(AXON_CKPT) \
		--timing-gen data/timing_gen.json --timing-train data/timing_train.json \
		--out-dir docs/assets/axon-world

# Genie-style video model on teacher streams, with the exact-fade student as baseline.
axon-video:
	$(UV) run sentionaut-world --output $(STREAM_DATASET) --model axonmap --stream --episodes 512 \
		--sequence-length 32 --silent-tail 8 --xrange -12 12 --yrange -12 12 --xystep 0.5 \
		--timing data/timing_gen_stream.json
	$(UV) run sentionaut-axon-video train --dataset $(STREAM_DATASET) --ckpt data/axon_video.pt \
		--timing data/timing_train_video.json --epochs 40 --batch-size 16 --num-workers 2
	$(UV) run sentionaut-distill-axon --dataset $(STREAM_DATASET) --ckpt data/axon_stream_markov.pt \
		--timing data/timing_train_markov.json --epochs 40 --batch-size 16 --lr 5e-4 --rollout-k 16 \
		--train-stride 8 --val-stride 16 --dim 128 --depth 4 --num-workers 2
	$(UV) run sentionaut-axon-video report --dataset $(STREAM_DATASET) --ckpt data/axon_video.pt \
		--baseline-ckpt data/axon_stream_markov.pt --timing-train data/timing_train_video.json \
		--timing-baseline data/timing_train_markov.json --timing-gen data/timing_gen_stream.json \
		--out-dir docs/assets/axon-video

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
