# BEM: Better Experimentation Manager for deep learning with Pytorch

This repository provides a framework to experiment with different deep learning models with Pytorch, streamlining training, evaluation, and generation (for generative models). BEM is designed to satisfy the following properties:

- **Flexibility**: Allows easy integration of custom models and methods.
- **Modularity**: Components like data handling, training, evaluation, and logging are modular.
- **Extensibility**: By implementing the required functions, you can support your own generative models.
- **Reproducibility**: Saving experiment parameters and metrics to ensure reproducibility and organized storage.
- **Device Support**: Utilizes available hardware (CPU, CUDA, MPS).

See projects [DLPM](https://github.com/darioShar/DLPM) and [Generative PDMPs](https://github.com/darioShar/PDMP) using this framework.


This repo provides a unique framework to experiment with different deep generative models, making it vastly easier to:
* Train them
* Evaluate them
* Compare them


See our `how_to_use_bem.ipynb` notebook.