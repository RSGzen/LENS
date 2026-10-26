"""thin UI : a Streamlit viewer over recorded runs.

``data`` is the read-only data layer (no Streamlit import, testable offline);
``app`` is the Streamlit script itself. Run it with::

    uv sync --extra ui
    streamlit run src/lens/ui/app.py
"""

from __future__ import annotations

__all__ = ["data"]
