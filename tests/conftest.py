import os

# Headless rendering for any test that touches matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")
