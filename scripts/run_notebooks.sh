#!/bin/bash

# Script/command runs all notebooks in the `notebooks` directory, while clearing metadata etc to
# avoid redundant information being stored in the git repository. This is useful for
# CI pipelines, and locally to ensure that the notebooks are "clean" before committing.

PROJECT_ROOT=$(git rev-parse --show-toplevel)

uv run \
    --with ipykernel \
    --with nbconvert \
    jupyter nbconvert --execute --to notebook --inplace \
    --ClearOutputPreprocessor.enabled=True \
    --ClearMetadataPreprocessor.enabled=True \
    "$PROJECT_ROOT"/notebooks/*.ipynb

# Show message if notebook contents have changed.
# Used in CI to ensure everything is up to date.
ORANGE='\033[0;33m'
GREEN='\033[0;32m'
RESET='\033[0m'

if ! git -C "$PROJECT_ROOT" diff --exit-code -- notebooks/; then
    printf "${ORANGE}Some notebook content has changed.${RESET}\n" >&2
    exit 1
else
    printf "${GREEN}All notebooks are up to date${RESET}\n"
fi
