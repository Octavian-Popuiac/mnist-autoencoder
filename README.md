# MNIST Autoencoder — Depth and Regularization

Implementation and empirical evaluation of fully connected autoencoders
on MNIST using TensorFlow and Keras.

This project was developed for the Generative Artificial Intelligence
course at the University of Algarve.

## Objective

Investigate how encoder depth and regularization affect reconstruction
performance, training stability, and latent representations.

## Experimental setup

Three encoder architectures are compared, each with a symmetric decoder:

| Encoder depth | Encoder                  | Decoder                  |
| ------------- | ------------------------ | ------------------------ |
| 1             | 784 → 128 → 2            | 2 → 128 → 784            |
| 2             | 784 → 256 → 128 → 2      | 2 → 128 → 256 → 784      |
| 3             | 784 → 256 → 128 → 64 → 2 | 2 → 64 → 128 → 256 → 784 |

Depth excludes the latent and output layers.

Each architecture is trained with:

- No regularization.
- L1 regularization on latent activations.
- L2 regularization on Dense-layer weights.

Three random seeds (42, 123, and 2026) are used for each configuration,
resulting in 27 training runs.

### Data and training

- Training images: 50,000.
- Validation images: 10,000.
- Test images: 10,000.
- Pixel values normalized to [0, 1].
- Latent dimension: 2.
- Optimizer: Adam, learning rate 0.001.
- Batch size: 128.
- Maximum epochs: 30.
- Early stopping: patience of 5 epochs, monitoring validation MSE
  and restoring the best weights.
- L1 coefficient: 1e-4.
- L2 penalty: (1e-5 / 2) × sum of squared weights.

## Results

The three-layer autoencoder without regularization achieved the lowest
mean validation MSE.

| Configuration               | Mean validation MSE | Standard deviation |
| --------------------------- | ------------------: | -----------------: |
| 1 layer, no regularization  |            0.043256 |           0.000367 |
| 2 layers, no regularization |            0.038057 |           0.000192 |
| 3 layers, no regularization |            0.036497 |           0.000585 |

The selected configuration achieved a mean test MSE of
**0.036351 ± 0.000428** across three seeds.

At the tested strengths, L1 and L2 did not improve mean validation
reconstruction performance. The two-layer regularized configurations
were particularly sensitive to initialization, with seed 2026 showing
stalled learning.

The notebook includes the full comparison, learning curves,
reconstructions, latent-space visualizations, and discussion.

## Repository structure

    .
    ├── README.md
    ├── Lab1_MNIST_Octavian_Popuiac_79911.ipynb
    └── images/
        └── vanilla_autoencoder.png

The `resultados_tp1/` directory is created during execution and contains
saved weights, metrics, training histories, and figures.

## Running the notebook

Install the required packages:

    pip install tensorflow numpy pandas matplotlib jupyter

Start Jupyter:

    jupyter notebook

Open `Lab1_MNIST_Octavian_Popuiac_79911.ipynb` and execute the cells
in order. MNIST is downloaded automatically on the first run.

The saved experiment outputs were produced with TensorFlow 2.21.0.
Results and execution times may vary across environments.

Running all cells retrains the 27 models and updates the generated
result files. The saved notebook outputs can be inspected without
retraining.

## Limitations

The experiment uses three seeds, one coefficient per regularization
type, a two-dimensional latent representation, and a limited training
budget. Depth and parameter count vary together, so their effects
cannot be isolated.

The results do not imply that regularization is generally ineffective
or that the selected architecture is optimal for MNIST.

## References and acknowledgements

The project builds on the course materials `MNIST_AE.ipynb` and
`Lab 1-MNIST-AE-v2.pdf`.

- [Sparse Autoencoder — Andrew Ng, Stanford](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
- [Dive into Deep Learning — Weight Decay](https://classic.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html)

## Author

Octavian Popuiac  
University of Algarve
