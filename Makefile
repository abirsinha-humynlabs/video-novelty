PY ?= python3
INDEX ?= .novelty
CLIPS ?= data/clips

.PHONY: help install install-gpu test lint clips demo index calibrate report clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

install:        ## CPU install (no torch, no weights, runs anywhere)
	$(PY) -m pip install -e ".[fit,dev]"

install-gpu:    ## add torch + transformers for the DINOv2 / V-JEPA2 path
	$(PY) -m pip install -e ".[gpu,fit,dev]"

test:           ## unit tests (synthetic video, ~30s)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check novelty tests

clips:          ## cut the demo clips out of a source video (see script header)
	bash scripts/prepare_eval_clips.sh

index:          ## build signatures from $(CLIPS)
	$(PY) -m novelty.cli index $(CLIPS) --index $(INDEX) --window 30 --hop 15

calibrate:      ## fit the null model + fusion weights
	$(PY) -m novelty.cli calibrate --index $(INDEX) --labels eval/pairs.yaml

report:         ## write novelty-report.html
	$(PY) -m novelty.cli report --index $(INDEX) --out novelty-report.html

demo: clips index calibrate report  ## the whole thing, end to end
	$(PY) -m novelty.cli select --index $(INDEX)

clean:
	rm -rf $(INDEX) novelty-report.html selection.json .pytest_cache .ruff_cache
