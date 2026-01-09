from __future__ import annotations

# NOTE: Individual plugin modules are NOT imported here to avoid circular imports.
# Import them directly when needed:
#   from covertreex.plugins import metrics
#   from covertreex.plugins import conflict
#   from covertreex.plugins import traversal
#
# See: https://github.com/anthropics/covertreex/issues/XXX

__all__ = ["conflict", "metrics", "traversal"]
