# Calibrated Confidence and Selective Automation for Bearing Fault Diagnosis: From Artificial to Real Damage

Code for a study of whether calibrated confidence in bearing fault diagnosis survives the shift from artificial to
real damage, on the Paderborn University bearing dataset.

## Setup

```bash
pip install -r requirements.txt
pytest
```

## Data

The Paderborn bearing data are not included. They are available from the
[KAt-DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download) of Paderborn
University under the CC BY-NC 4.0 licence.

## Usage

`scripts/` has one script per analysis step, and `notebooks/colab_cnn.ipynb` trains the 1D-CNN in Google Colab. Output is
written to `results/`.

## Licence

MIT (see [`LICENSE`](LICENSE)).
