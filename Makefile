ARGS ?=

# bash for the `time` keyword used by the rules that run the real program
# (test_rag, test_prl, test_corpus): it prints how long each command took,
# in the format below.
SHELL := /bin/bash
export TIMEFORMAT := Time: %1Rs

# On 42 computers the home is too small for torch and the model, so the
# virtualenv, the uv cache and the Hugging Face cache go to sgoinfre (shared
# by every computer of the campus). Exported for every rule, so `uv run`
# also finds the venv there. Elsewhere (no sgoinfre) nothing changes.
# Variables already set in the environment win. Other folder: STORE=/path
STORE ?= $(if $(USER),$(wildcard /sgoinfre/students/$(USER)))
ifneq ($(STORE),)
export UV_PROJECT_ENVIRONMENT ?= $(STORE)/rag-venv
export UV_CACHE_DIR           ?= $(STORE)/uv-cache
export HF_HOME                ?= $(STORE)/hf
# Hardlinks do not work on NFS.
export UV_LINK_MODE           ?= copy
endif

all: install

install:
	@uv sync
	@echo "Everything is up to date now."
	@if [ -n "$(STORE)" ]; then echo "venv: $(UV_PROJECT_ENVIRONMENT)"; fi

run:
	@uv run -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

# Local HTTP API (bonus 5) until Ctrl+C; try it at http://127.0.0.1:8000/docs
# Options for `serve`: SERVE_ARGS="--port 8080 --retriever bm25"
SERVE_ARGS ?=

serve:
	@uv run -m src serve $(SERVE_ARGS)

# Graphical interface for the posture analysis until Ctrl+C; open
# http://127.0.0.1:8000 (this computer only). Ctrl+C stops every camera
# and video, and each one saves its report first.
# The detailed reports (written by themselves when an analysis ends) and
# the summaries are written by Gemini (needs
# GEMINI_API_KEY): LLM=lite (Gemini 3.5 Flash Lite, default, seconds),
# LLM=hydrogen (Gemini 4 Hydrogen, the best report, about a minute: for a
# demonstration) or LLM=flash (Gemini 3.8 Flash, minutes):
# make app LLM=hydrogen
# Recording the cameras' original video is off unless RECORD=1 (for
# calibration videos): make app RECORD=1
# The work at a screen (distance, neck, blinks, rests, breaks) is only
# analysed with SCREEN=1; by default only the body: make app SCREEN=1
# The load of the lifts is not seen: LOAD kg are assumed (5 by default,
# said so in the reports; LOAD=0 for none): make app LOAD=8
# A session saves a report every REPORT_HOURS hours and goes on (2 by
# default; 0: only when it ends): make app REPORT_HOURS=1
# Private mode, the default: no picture of any camera is made or shown,
# only its name, that it is active and what is measured. PRIVATE=0 shows
# the pictures; RECORD=1 (calibration) needs them and turns it off:
# make app PRIVATE=0
# Options for `app`: APP_ARGS="--port 8080 --limit 2 --model full"
LLM      ?= lite
RECORD   ?= 0
SCREEN   ?= 0
LOAD     ?= 5
REPORT_HOURS ?= 2
PRIVATE  ?= 1
APP_ARGS ?=

# Demonstration: add two cameras; they wait for each other, start
# together 5 s after both are ready, stop after 30 s and their reports
# are written at once with the best model (Gemini 4 Hydrogen):
# make demo   (DEMO_CAMERAS=2 DEMO_SECONDS=30 DEMO_COUNTDOWN=5)
# Each run writes one report per camera: rehearse with the fast model,
# so the best one's quota is kept for the real take:
# make demo DEMO_LLM=lite
DEMO_LLM       ?= hydrogen
# The demonstration shows the pictures (to see what happens);
# DEMO_PRIVATE=1 hides them as in normal use.
DEMO_PRIVATE   ?= 0
DEMO_CAMERAS   ?= 2
DEMO_SECONDS   ?= 30
DEMO_COUNTDOWN ?= 5
# The reports are written at the end; to rehearse without spending
# calls: make demo DEMO_WRITE=0 (each can still be written by its button).
DEMO_WRITE     ?= 1

demo:
	@uv run -m src app --llm $(DEMO_LLM) --demo --demo_cameras $(DEMO_CAMERAS) --demo_seconds $(DEMO_SECONDS) --demo_countdown $(DEMO_COUNTDOWN) --load_kg $(LOAD) $(if $(filter 1,$(RECORD)),--record) $(if $(filter 1,$(DEMO_PRIVATE)),--private) $(if $(filter 0,$(DEMO_WRITE)),--demo_write=False) $(APP_ARGS)

app:
	@uv run -m src app --llm $(LLM) $(if $(filter 1,$(RECORD)),--record) $(if $(filter 1,$(SCREEN)),--screen) --load_kg $(LOAD) --report_hours $(REPORT_HOURS) $(if $(filter 1,$(RECORD)),,$(if $(filter 1,$(PRIVATE)),--private)) $(APP_ARGS)

# Each rule keeps its own mypy cache: the cache is only valid for the
# flags that built it, so sharing one made every switch between lint and
# lint-strict re-check torch, transformers... from scratch (~50 s, not ~2 s).
lint:
	uv run -m flake8 . && uv run -m mypy --cache-dir .mypy_cache/lint \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	.

lint-strict:
	uv run -m flake8 . && uv run -m mypy --cache-dir .mypy_cache/strict --strict .

test:
	uv run pytest

# How test_rag, test_prl and test_corpus run `index` (bonus 3):
#   FULL=1 (default): --full, always the whole corpus; removes the query
#                     caches.
#   FULL=0:           incremental, only the files changed since the last
#                     index; nothing is written if none changed. To show
#                     the difference: make test_rag, touch a file of
#                     tests/data/raw, then make test_rag FULL=0.
FULL       ?= 1
INDEX_FULL := $(if $(filter 1,$(FULL)),--full)

# Full pipeline (index -> search_dataset -> evaluate) on the mini corpus in
# tests/data. Generated files go to tests/data/processed and
# tests/data/output (ignored by git, removed by `make clean`).
# Options: TEST_DATASET=dataset_mini_hard.json TEST_K=10 TEST_CHUNK=2000
TEST_DATA    := tests/data
TEST_INDEX   := $(TEST_DATA)/processed
TEST_OUTPUT  := $(TEST_DATA)/output/search_results
TEST_DATASET ?= dataset_mini.json
TEST_K       ?= 5
TEST_CHUNK   ?= 500

test_rag:
	time uv run -m src index --raw_dir $(TEST_DATA)/raw --index_dir $(TEST_INDEX) \
		--max_chunk_size $(TEST_CHUNK) $(INDEX_FULL)
	time uv run -m src search_dataset \
		--dataset_path $(TEST_DATA)/datasets/UnansweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K) --save_directory $(TEST_OUTPUT) --index_dir $(TEST_INDEX)
	time uv run -m src evaluate \
		--student_search_results_path $(TEST_OUTPUT)/$(TEST_DATASET) \
		--dataset_path $(TEST_DATA)/datasets/AnsweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K)

# Retrieval of the regulation (the RAG of the reports): the prevention
# corpus in data/raw/prl, indexed in data/processed/prl (already in the
# repository), searched with each question of data/datasets/prl and
# evaluated: dataset_prl.json (the questions it was tuned on) and
# dataset_prl_holdout.json (questions kept apart). The results go to
# data/output/prl (ignored by git). The index is not rebuilt: INDEX=1
# rebuilds it first. Options: PRL_K=10 PRL_RETRIEVER=bm25
PRL_RAW       := data/raw/prl
PRL_INDEX     := data/processed/prl
PRL_DATA      := data/datasets/prl
PRL_DATASETS  := dataset_prl.json dataset_prl_holdout.json
PRL_K         ?= 5
PRL_RETRIEVER ?= hybrid
PRL_OUTPUT    := data/output/prl/$(PRL_RETRIEVER)
INDEX         ?= 0

test_prl:
	@if [ "$(INDEX)" = 1 ]; then \
		time uv run -m src index --raw_dir $(PRL_RAW) \
			--index_dir $(PRL_INDEX) $(INDEX_FULL) || exit 1; \
	fi
	@for dataset in $(PRL_DATASETS); do \
		echo "===== $$dataset ====="; \
		uv run -m src search_dataset \
			--dataset_path $(PRL_DATA)/UnansweredQuestions/$$dataset \
			--k $(PRL_K) --save_directory $(PRL_OUTPUT) \
			--index_dir $(PRL_INDEX) --retriever $(PRL_RETRIEVER) || exit 1; \
		uv run -m src evaluate \
			--student_search_results_path $(PRL_OUTPUT)/$$dataset \
			--dataset_path $(PRL_DATA)/AnsweredQuestions/$$dataset \
			--k $(PRL_K) || exit 1; \
	done

# Same pipeline on a second, independent corpus: every project in
# tests/test_corpus/raw, indexed together, with handwritten datasets (exact
# source ranges), each reported apart: Fly-in (a drone-routing simulator)
# with dataset_flyin_docs.json (the READMEs and the map files) and
# dataset_flyin_code.json (src/), A-Maze-ing (a maze generator and game)
# with dataset_amazeing_docs.json (its two READMEs and config files) and
# dataset_amazeing_code.json (mazegen/ and src/), and Dones (Python
# exercises P0-P10) with dataset_dones_code.json. Checks that the system is
# not tuned to vLLM only; each project is also noise for the others'
# questions. Same layout as data/, so every path is an option. The corpus
# is indexed once, then search_dataset + evaluate run for each dataset.
#   CORPUS_DATASETS=... dataset file names (default: all five)
#   CORPUS_K=5          k for search and evaluate (default 10, as the
#                       moulinette)
#   CORPUS_CHUNK=500    max_chunk_size (default 2000)
#   ANSWER=1            also generate the answers with Qwen (answer_dataset:
#                       slow, ~20 s per question, and needs the model's
#                       memory); off by default
CORPUS_DATA     := tests/test_corpus
CORPUS_INDEX    := $(CORPUS_DATA)/processed
CORPUS_OUTPUT   := $(CORPUS_DATA)/output/search_results/UnansweredQuestions
CORPUS_ANSWERS  := $(CORPUS_DATA)/output/search_results_and_answer/UnansweredQuestions
CORPUS_DATASETS ?= dataset_flyin_docs.json dataset_flyin_code.json \
                   dataset_amazeing_docs.json dataset_amazeing_code.json \
                   dataset_dones_code.json
CORPUS_K        ?= 10
CORPUS_CHUNK    ?= 2000
ANSWER          ?= 0

test_corpus:
	time uv run -m src index --raw_dir $(CORPUS_DATA)/raw \
		--index_dir $(CORPUS_INDEX) --max_chunk_size $(CORPUS_CHUNK) $(INDEX_FULL)
	@for dataset in $(CORPUS_DATASETS); do \
		echo "===== $$dataset ====="; \
		time uv run -m src search_dataset \
			--dataset_path $(CORPUS_DATA)/datasets/UnansweredQuestions/$$dataset \
			--k $(CORPUS_K) --save_directory $(CORPUS_OUTPUT) \
			--index_dir $(CORPUS_INDEX) || exit 1; \
		time uv run -m src evaluate \
			--student_search_results_path $(CORPUS_OUTPUT)/$$dataset \
			--dataset_path $(CORPUS_DATA)/datasets/AnsweredQuestions/$$dataset \
			--k $(CORPUS_K) || exit 1; \
		if [ "$(ANSWER)" = 1 ]; then \
			time uv run -m src answer_dataset \
				--student_search_results_path $(CORPUS_OUTPUT)/$$dataset \
				--save_directory $(CORPUS_ANSWERS) \
				--index_dir $(CORPUS_INDEX) || exit 1; \
		fi; \
	done

clean:
	@find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache $(TEST_INDEX) $(TEST_DATA)/output \
		$(CORPUS_INDEX) $(CORPUS_DATA)/output
	@echo "Cache cleaned"

# Removes what the commands generated in data/output: the analyses and
# reports of the sessions (measures of real people) and the search
# results. The regulation index in data/processed/prl stays: the
# application needs it (`index` builds it again).
cleandata:
	@rm -rf data/output
	@echo "Generated outputs removed"

.PHONY: all install run debug serve app demo lint lint-strict test test_rag test_prl test_corpus clean cleandata
