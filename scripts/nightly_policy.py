"""Shared nightly-stack policy for refresh_nightly_stacks.py and
validate_torch_index_stacks.py - one authoritative copy so the refresher can
never resolve an entry the validator rejects in the same workflow run.

(The desktop app additionally stops OFFERING a nightly entry ~45 days after
its wheel date, ahead of PyTorch's ~60-day index purge.)
"""

# Index tags to offer nightlies for. NVIDIA covers the overwhelming
# majority of users; add tags here deliberately (AMD Windows stays out -
# pytorch.org publishes no Windows ROCm wheels, and mps has no dev builds
# on PyPI). Every tag must be one the desktop app's runtime index gate
# accepts (cu*/rocm*/xpu/cpu).
NIGHTLY_TAGS = ("cu132",)

# Refuse to publish a nightly tuple older than this. A nightly index that
# has not produced a coherent triple for a week is a problem a human should
# look at, not something to silently republish.
MAX_AGE_DAYS = 7
