"""Compatibility shims — imported before any ``coffea.lookup_tools`` import.

coffea's Rochester / double-Crystal-Ball modules are imported eagerly by
``coffea.lookup_tools.__init__`` (the package that also contains
``correctionlib_wrapper``, ``extractor`` and the JEC ``jetmet_tools``).  Those
modules reference a private scipy helper, ``_lazywhere``, that newer scipy
(>= ~1.15) removed, while also trying a newer path
(``scipy._lib.array_api_extra.apply_where``) that older/mid scipy does not have.
On scipy 1.18 *both* are absent, so importing anything from
``coffea.lookup_tools`` raises ``ImportError`` and blocks the correction tools.

This analysis never uses those distributions.  We restore the small
``_lazywhere`` helper so coffea's documented old-style fallback import path
succeeds, without changing the shared conda environment.  Importing this module
is idempotent and a no-op once scipy (or a prior import) provides the symbol.
"""

from __future__ import annotations

import numpy as np


def _install_lazywhere() -> None:
    try:
        import scipy.stats._continuous_distns as _cd
    except Exception:
        return
    if hasattr(_cd, "_lazywhere"):
        return

    def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
        """Vendored copy of scipy's removed ``_lazywhere`` utility."""
        cond = np.asarray(cond)
        arrays = np.broadcast_arrays(*arrays)
        tcode = np.mintypecode([np.asarray(a).dtype.char for a in arrays])
        out = np.full(np.shape(arrays[0]), fill_value=fillvalue, dtype=tcode)
        np.place(out, cond, f(*tuple(np.extract(cond, a) for a in arrays)))
        if f2 is not None:
            np.place(out, ~cond, f2(*tuple(np.extract(~cond, a) for a in arrays)))
        return out

    _cd._lazywhere = _lazywhere


_install_lazywhere()
