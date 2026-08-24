import os

import pytest

# Headless rendering for any test that touches matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _reset_columns_params():
    """Put custom/columns.py back on its defaults before every test.

    ``configure()`` installs into module-level objects, so without this a test
    that sets ``rpc_mode`` (or one whose ``configure()`` call raised) leaks
    its settings into whatever runs next and the suite becomes order
    dependent.  Nothing is installed at import time any more, so this is also
    what puts the defaults in place for the tests that never configure.
    """
    try:
        from custom import columns
    except ImportError:
        return
    columns.configure()
