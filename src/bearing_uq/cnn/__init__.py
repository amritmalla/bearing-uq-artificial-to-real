"""1D-CNN baseline on raw vibration windows (trained in Colab; see notebooks/colab_cnn.ipynb).

The modules in this package are copied into the Colab notebook by scripts/build_cnn_notebook.py, so they
must stay self-contained: they import only numpy, pandas, torch and each other (lines starting with
``from bearing_uq.cnn`` are dropped when the notebook is built).
"""
