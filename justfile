# Tasks of the Vernier repository. `just` lists them.

default:
    @just --list

# Install the tool and its test dependencies into .venv
setup:
    uv sync

# Run the tests that need no network (under two minutes)
test:
    uv run pytest

# Run the tests that read the boards' own repositories: hosts is github, or github,huggingface
test-live hosts="github":
    VERNIER_LIVE={{hosts}} uv run pytest tests/test_live.py -v

# Also run the exact test with the t statistic on GIFT-Eval (about ten minutes)
test-slow:
    VERNIER_SLOW=1 uv run pytest tests/test_boards.py -k exact_t -v

# Print one row per board on the paper's snapshots
reports:
    uv run vernier report --board all --pinned

# Print one row per board as it is now (network)
live:
    uv run vernier report --board all

# Write the paper's snapshot of a board into src/vernier/pins from the board's own repository (network)
pin board:
    uv run vernier export --board {{board}} --pinned
