#!/bin/bash
poetry install --no-interaction >/dev/null
poetry run python bench_tui.py
