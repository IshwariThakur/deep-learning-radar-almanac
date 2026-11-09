#!/usr/bin/env bash
# One setup for the notebooks and the prototype (tested with Python 3.11 in a new virtual environment).
#   bash install.sh            (uses "python" of the active environment)
set -e
PYTHON=${PYTHON:-python}

$PYTHON -m pip install -r requirements.txt

# keras-hub provides SegFormer. Its package metadata asks for tensorflow-text, which would replace
# TensorFlow 2.20 with another version; SegFormer does not use tensorflow-text, so keras-hub is installed
# without dependencies, and the small packages it really needs are installed separately.
$PYTHON -m pip install --no-deps keras-hub==0.24.0
$PYTHON -m pip install kagglehub==1.0.3 huggingface_hub==2.4.0 regex==2026.11.7 rich==15.0.0

$PYTHON -c "import tensorflow as tf, keras, keras_hub, gradio, ee, rasterio; print('TensorFlow', tf.__version__, '| Keras', keras.__version__, '| keras-hub', keras_hub.__version__, '| Gradio', gradio.__version__)"
