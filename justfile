default:
    @just --list

# The tests that need no network
test:
    uv run pytest

# The tests that read the boards' own repositories: github, or github,huggingface
test-live hosts="github":
    VERNIER_LIVE={{hosts}} uv run pytest tests/test_live.py -v

# The exact t test on GIFT-Eval (about ten minutes)
test-slow:
    VERNIER_SLOW=1 uv run pytest tests/test_boards.py -k exact_t -v

# Write a board's snapshot into src/vernier/pins from the board's own repository
pin board:
    uv run vernier export --board {{board}} --pinned
