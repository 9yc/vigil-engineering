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
CADENCE_SIM = $(ARTIFACT_DIR)/cadence_sim_cli
TELEMETRY_DECODER = $(ARTIFACT_DIR)/telemetry_block_decoder
CONFIG_CALIBRATOR = $(ARTIFACT_DIR)/config_calibrator_cli
PERIODIC_SAMPLER = $(ARTIFACT_DIR)/periodic_cmd_sampler
DVFS_RESIDENCY = $(ARTIFACT_DIR)/dvfs_residency_audit

BINARIES = \
	$(SAMPLER) \
	$(CADENCE_SIM) \
	$(TELEMETRY_DECODER) \
	$(CONFIG_CALIBRATOR) \
	$(PERIODIC_SAMPLER) \
	$(DVFS_RESIDENCY)

all: $(BINARIES)

$(SAMPLER): tools/sample_nodes.cpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) "$<" -o "$@"

$(CADENCE_SIM): tools/cadence_sim_cli.cpp include/cadence_controller.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

$(TELEMETRY_DECODER): tools/telemetry_block_decoder.cpp include/framed_telemetry.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

$(CONFIG_CALIBRATOR): tools/config_calibrator_cli.cpp include/binomial_bounds.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

$(PERIODIC_SAMPLER): tools/periodic_cmd_sampler.cpp include/periodic_sampler.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

$(DVFS_RESIDENCY): tools/dvfs_residency_audit.cpp include/cpufreq_residency.hpp
	mkdir -p "$(ARTIFACT_DIR)"
	$(CXX) $(CXXFLAGS) -Iinclude "$<" -o "$@"

test: all
	mkdir -p "$(ARTIFACT_DIR)/test-tmp"
	ARTIFACT_DIR="$(ARTIFACT_DIR)" \
		TMPDIR="$(ARTIFACT_DIR)/test-tmp" \
		SAMPLER_BINARY="$(SAMPLER)" \
		PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) -B -m unittest discover -s tests -v
