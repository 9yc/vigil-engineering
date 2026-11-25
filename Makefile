CXX ?= c++
PYTHON ?= python3
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror -pedantic

.PHONY: all test

ifeq ($(strip $(ARTIFACT_DIR)),)
$(error Set ARTIFACT_DIR to an absolute directory for build and test artifacts)
endif
ifneq ($(filter /%,$(ARTIFACT_DIR)),$(ARTIFACT_DIR))
$(error ARTIFACT_DIR must be an absolute path)
endif

SAMPLER = $(ARTIFACT_DIR)/sample_nodes
PROC_WATCH = $(ARTIFACT_DIR)/proc_tree_watch
TRANSFORMER_REF = $(ARTIFACT_DIR)/transformer_ref_cli
ARENA_VERIFIER = $(ARTIFACT_DIR)/arena_verifier

BINARIES = \
	$(SAMPLER) \
	$(PROC_WATCH) \
	$(TRANSFORMER_REF) \
	$(ARENA_VERIFIER)

all: $(BINARIES)

$(SAMPLER): tools/sample_nodes.cpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) "$<" -o "$@"

$(PROC_WATCH): tools/proc_tree_watch.cpp include/proc_watchdog.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

$(TRANSFORMER_REF): tools/transformer_ref_cli.cpp include/transformer_ops.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -pthread -Iinclude "$<" -o "$@"

$(ARENA_VERIFIER): tools/arena_verifier.cpp include/slab_arena.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

test: all
	mkdir -p "$(ARTIFACT_DIR)/test-tmp"
	ARTIFACT_DIR="$(ARTIFACT_DIR)" \
		TMPDIR="$(ARTIFACT_DIR)/test-tmp" \
		SAMPLER_BINARY="$(SAMPLER)" \
		PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) -B -m unittest discover -s tests -v
