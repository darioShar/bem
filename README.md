# BEM: Better Experimentation Manager (for generative models)

This repository provides a unique framework to experiment with different deep generative models, making it easier to:

* Train them
* Evaluate them
* Compare them

See for instance projects [DLPM](https://github.com/darioShar/DLPM) and [Generative PDMPs](https://github.com/darioShar/PDMP) using this framework.

# How to use

## Configuration

The framework requires defining a dictionnary `p` containing the following subdictionnaries:

- `method`: Specifies the generative method (e.g., `ddpm`, `gan`, `vae`, ...).
- `data`: Dataset configurations.
- `model`: Model-specific parameters.
- `training`: Training parameters for the specified method.
- `optim`: Optimizer and learning rate scheduler settings.
- `eval`: Evaluation settings and method-specific evaluation parameters.
- `run`: Parameters for a specific run.


### Configuration file

The framework can use a YAML configuration file to manage experiment parameters, retreiving the associated dictionnary with the `bem.utils_exp.FileHandler.get_param_from_config` function.

An example configuration file (`config.yaml`) might look like:

```yaml
seed: 42 # or null

method: ddpm

data:
  dataset: mnist
  ...

model:
  ddpm:
    num_layers: 10
    hidden_dim: 128
    ...

training:
  batch_size: 256
  num_workers: 2
  ddpm:
    epochs: 100
    grad_clip: 1.0
    ...

optim:
  lr: 0.0001
  schedule: steplr
  ...

eval:
  data_to_generate: 1000
  real_data: 1000
  batch_size: 128
  ddpm:
    timesteps: 1000
    ...

run:
  epochs: 1000
  eval_freq: 250
  checkpoint_freq: 250
  ...

```


## Functions to define

To integrate your own models or methods into the framework, you need to define the following functions:

1. **`init_method_by_parameter(p)`**: Initializes the generative method based on parameters `p`.

   ```python
   def init_method_by_parameter(p):
       # Example for DDPM
       return DDPMMethod(p['model']['ddpm'])
   ```

2. **`init_models_by_parameter(p)`**: Initializes the models, optimizers, and learning schedules.

   ```python
   def init_models_by_parameter(p):
       model = DDPMModel(**p['model']['ddpm'])
       optimizer = optim.AdamW(model.parameters(), lr=p['optim']['lr'])
       learning_schedule = get_scheduler(
           p['optim']['schedule'],
           optimizer=optimizer,
           num_warmup_steps=p['optim']['warmup'],
           num_training_steps=p['optim']['lr_steps']
       )
       return {'default': model}, {'default': optimizer}, {'default': learning_schedule}
   ```

3. **`reset_models()`**: Resets models to their initial state (useful if training encounters issues like NaNs).

   ```python
   def reset_models():
       models, optimizers, learning_schedules = init_models_by_parameter(p)
       return models, optimizers, learning_schedules
   ```

These functions allow the framework to be method-agnostic and flexible.

## Default example with Denoising Diffusion (DDPM)

Here’s how to set up and run a default experiment with DDPM.

1. **Define the Configuration File**

   Create a `config_ddpm.yaml` file with your parameters.

   ```yaml
   method: ddpm
   data:
     dataset: mnist
     data_path: ./data/mnist
   model:
     ddpm:
       num_layers: 10
       hidden_dim: 128
       ...
   training:
     ddpm:
       batch_size: 64
       num_workers: 4
       epochs: 100
       grad_clip: 1.0
   optim:
     lr: 0.0001
     schedule: cosine
     warmup: 5000
   eval:
     data_to_generate: 1000
     real_data: 1000
     batch_size: 128
     ddpm:
       timesteps: 1000
       ...
   ```

2. **Implement Required Functions**

   ```python
   # Assuming you have defined DDPMMethod and DDPMModel elsewhere
   def init_method_by_parameter(p):
       return DDPMMethod(p['model']['ddpm'])

   def init_models_by_parameter(p):
       model = DDPMModel(**p['model']['ddpm'])
       optimizer = optim.AdamW(model.parameters(), lr=p['optim']['lr'])
       learning_schedule = get_scheduler(
           p['optim']['schedule'],
           optimizer=optimizer,
           num_warmup_steps=p['optim']['warmup'],
           num_training_steps=p['training']['ddpm']['epochs']
       )
       return {'default': model}, {'default': optimizer}, {'default': learning_schedule}

   def reset_models():
       models, optimizers, learning_schedules = init_models_by_parameter(p)
       return models, optimizers, learning_schedules
   ```

3. **Run the Experiment**

   ```python
   from BEM.Experiments import Experiment
   from BEM.utils_exp import ExpUtils

   # Load parameters from the config file
   p = ExpUtils.get_param_from_config('./configs', 'config_ddpm.yaml')

   # Initialize the Experiment
   exp = Experiment(
       checkpoint_dir='./checkpoints',
       p=p,
       init_method_by_parameter=init_method_by_parameter,
       init_models_by_parameter=init_models_by_parameter,
       reset_models=reset_models
   )

   # Prepare the experiment
   exp.prepare()

   # Run the experiment
   exp.run()
   ```

4. **Evaluate and Visualize Results**

   ```python
   # Evaluate the model
   exp.manager.evaluate()

   # Display generated images
   exp.manager.display_plots()
   ```

## Logger

The framework uses an abstract `Logger` class (`Logger.py`) to handle logging. You can create a custom logger by inheriting from this class.

**Methods to Implement:**

- `initialize(self, p)`: Initialize the logger with parameters `p`.
- `set_values(self, value_dict)`: Set initial values or states.
- `log(self, data_type, data)`: Log data (e.g., losses, metrics).
- `stop(self)`: Flush and stop the logger.

**Example Implementation:**

```python
class CustomLogger(Logger):
    def initialize(self, p):
        # Initialize logging (e.g., open files, start sessions)
        pass

    def set_values(self, value_dict):
        # Set initial values or states
        pass

    def log(self, data_type, data):
        # Log data (e.g., write to console or file)
        print(f"{data_type}: {data}")

    def stop(self):
        # Clean up resources
        pass
```

**Using the Logger:**

```python
logger = CustomLogger()

exp = Experiment(
    checkpoint_dir='./checkpoints',
    p=p,
    logger=logger,
    init_method_by_parameter=init_method_by_parameter,
    init_models_by_parameter=init_models_by_parameter,
    reset_models=reset_models
)
```

## Scripts

*To Do*: Scripts for automating experiments, training, and evaluation are forthcoming.

# General structure and prescribed behaviour

## Training

Training is handled by the `TrainingManager` class (`TrainingManager.py`). It manages:

- **Epochs and Batches**: Iterates over data for the specified number of epochs.
- **Loss Computation**: Computes training losses using the method's `training_losses` function.
- **Optimization**: Performs gradient descent and updates models.
- **EMA Updates**: Updates Exponential Moving Averages if EMA is used.
- **Callbacks**: Supports batch and epoch callbacks for logging or other purposes.

**Example Training Loop:**

```python
exp.manager.train(
    total_epoch=100,
    eval_freq=10,
    checkpoint_freq=10,
    grad_clip=1.0
)
```

## Evaluation

Evaluation is managed by the `EvaluationManager` class (`EvaluationManager.py`). It computes metrics such as:

- **Wasserstein Distance**
- **Maximum Mean Discrepancy (MMD)**
- **Fréchet Inception Distance (FID)**
- **Precision and Recall Metrics**

Evaluation can be performed periodically during training or after training completes.

**Perform Evaluation:**

```python
# Evaluate the current model
exp.manager.evaluate()

# Evaluate EMA models if used
exp.manager.evaluate(evaluate_emas=True)
```

## Storing/Checkpointing

Experiments can be saved and loaded using the `save` and `load` methods in the `Experiment` class. Checkpoints include:

- Model parameters
- Optimizer states
- Learning schedules
- Evaluation metrics
- Experiment parameters

### Hash Functions for Checkpointing

To ensure reproducibility and organized storage, the framework uses hash functions to uniquely identify experiments and evaluations. This mechanism helps in managing checkpoints effectively.

**How It Works:**

- **Experiment Hash (`exp_hash`)**: Generated based on the experiment's parameters, such as dataset, method, and model configurations.
- **Evaluation Hash (`eval_hash`)**: Generated based on evaluation-specific parameters.
- **File Naming and Directories**: Checkpoint files and directories include these hashes to uniquely identify experiments and evaluations.

**Default Behavior:**

- The `exp_hash` is computed using a SHA-256 hash of a subset of the experiment parameters. By default, it includes:

  ```python
  {
      'data': p['data'],
      p['method']: p[p['method']],
      'model': p['model'][p['method']],
  }
  ```

- The `eval_hash` is similarly computed based on evaluation parameters:

  ```python
  {
      'eval': p['eval'][p['method']]
  }
  ```

- The hashes are truncated to manageable lengths (e.g., first 16 characters).

**Customizing Hash Functions:**

You can customize the hashing behavior by providing your own functions when initializing `ExpUtils`:

```python
from BEM.utils_exp import ExpUtils

def custom_exp_hash(p):
    # Define custom experiment hash parameters
    return {
        'custom_param': p['custom_param'],
        'model': p['model'],
        ...
    }

def custom_eval_hash(p):
    # Define custom evaluation hash parameters
    return {
        'eval_custom': p['eval']['custom_eval_param'],
        ...
    }

exp_utils = ExpUtils(
    p=p,
    exp_hash=custom_exp_hash,
    eval_hash=custom_eval_hash,
    init_method_by_parameter=init_method_by_parameter,
    init_models_by_parameter=init_models_by_parameter,
    reset_models=reset_models
)
```

### Checkpointing Paths and Filenames

The hashes are used to construct filenames and directories for saving and loading:

- **Model Checkpoints**: Saved in a directory specific to the dataset, with filenames including the `exp_hash` and optionally the epoch number.
- **Evaluation Results**: Saved in subdirectories with names including both `exp_hash` and `eval_hash`.
- **Parameter Files**: Saved alongside model checkpoints, also including the `exp_hash`.

**Example Paths:**

- Model Checkpoint:

  ```
  ./checkpoints/mnist/model_<exp_hash>_<epoch>.pt
  ```

- Evaluation Metrics:

  ```
  ./checkpoints/mnist/new_eval_<exp_hash>_<eval_hash>/eval_<exp_hash>_<eval_hash>.pt
  ```

- Parameters:

  ```
  ./checkpoints/mnist/parameters_<exp_hash>.pt
  ```

### Saving an Experiment:

```python
# Save at the current epoch
exp.save(curr_epoch=exp.manager.epochs)
```

This will save the model, parameters, and evaluation metrics with filenames including the hashes and epoch number.

### Loading an Experiment:

```python
# Load from a checkpoint directory
exp.load(checkpoint_dir='./checkpoints', curr_epoch=500)
```

The framework will search for checkpoint files matching the provided parameters and hashes.

### Benefits:

- **Reproducibility**: By hashing the parameters, you ensure that each experiment can be uniquely identified and reproduced.
- **Organization**: Checkpoints are neatly organized in directories and filenames that reflect their configurations.
- **Conflict Avoidance**: Using hashes prevents overwriting files from different experiments that might have similar names.

## Retrieving experiments results: comparison, charts, plots...

The framework provides tools to retrieve and visualize experiment results.

**Display Evaluation Metrics:**

```python
# Display losses
exp.manager.display_evals(key='losses', log_scale=True)
```

**Generate and Visualize Samples:**

```python
# Generate samples and display plots
exp.manager.display_plots(
    ema_mu=0.999,  # Use EMA model with mu=0.999
    title='Generated Samples',
    nb_datapoints=10000,
    plot_original_data=True
)
```

**Retrieve Experiment Results:**

- All results, metrics, and plots are stored in the specified checkpoint directory.
- Use the `EvaluationManager` and `GenerationManager` to load and visualize results.










## Available Datasets

The framework supports both image and synthetic datasets, which are managed through the `datasets` module (`datasets/__init__.py`).

**Image Datasets:**

- **MNIST**: Handwritten digit images.
- **CIFAR-10**: 32x32 color images in 10 classes.
- **CelebA**: Large-scale face attributes dataset.
- **LSUN**: Large-scale scene understanding.
- **ImageNet**: Large-scale image dataset.
- **FFHQ**: Flickr-Faces-HQ dataset.

**Synthetic Distributions (2D Data):**

- **gmm_2**: 2-component Gaussian Mixture Model.
- **gmm_grid**: Grid of Gaussian Mixture Models.
- **swiss_roll**: Swiss Roll distribution.
- **skewed_levy**: Skewed Lévy distribution.
- **sas**: Symmetric Alpha-Stable distribution.
- **sas_grid**: Grid of Symmetric Alpha-Stable distributions.

**Toy Datasets:**

- **rose**
- **fractal_tree**
- **olympic_rings**
- **checkerboard**

Specify the dataset in the `data` section:

```yaml
data:
  dataset: swiss_roll
  nsamples: 10000
  dim: 2
  n_mixture: 8
  std: 0.1
  normalized: True
```

**Dataset Loading and Transformation:**

The `get_dataset(p)` function in `datasets/__init__.py` handles dataset loading based on the configuration parameters `p`. It supports:

- **Custom Data Generation**: For synthetic distributions using the `Generator` class.
- **Image Datasets**: Applies appropriate transformations like resizing, random flipping, and normalization.
- **Data Dequantization**: Supports uniform and Gaussian dequantization.

**Example Usage:**

```python
from datasets import get_dataset

# Assume 'p' is your configuration dictionary
train_dataset, test_dataset = get_dataset(p)
```


### Get Dataset Details

The `datasets` module (`datasets/__init__.py`) is responsible for handling dataset loading, transformations, and providing utilities related to datasets.

### Dataset Loading

The function `get_dataset(p)` loads datasets based on the parameters provided in the configuration `p`. It supports:

- **Image Datasets**: Loads standard datasets like MNIST, CIFAR-10, CelebA, LSUN, etc.
- **Synthetic Distributions**: Generates synthetic datasets using the `Generator` class from `datasets/Data.py`.
- **Toy Datasets**: Loads toy datasets like "rose", "fractal_tree", "olympic_rings", and "checkerboard".

**Example Usage:**

```python
from datasets import get_dataset

# Assume 'p' is your configuration dictionary
train_dataset, test_dataset = get_dataset(p)
```

### Data Transformations

- **Affine Transformation**: Scales image pixel values from [0, 1] to [-1, 1].
  ```python
  def affine_transform(x):
      return 2 * x - 1
  ```
- **Inverse Affine Transformation**: Scales pixel values from [-1, 1] back to [0, 1].
  ```python
  def inverse_affine_transform(x):
      return (x + 1) / 2
  ```

<!-- - **Logit Transformation**: Applies logit transformation to data for certain models. -->

### Utility Functions

- **`is_image_dataset(name)`**: Determines if a given dataset name corresponds to an image dataset.

<!-- 
### Dataset Classes

- **`imagenet64_dataset`**: Custom class for loading downsampled ImageNet datasets.
- **`Crop`**: Custom transformation class for cropping images. -->

### Data Path

All datasets are assumed to be located in the `./data` directory by default.

### Example Dataset Configuration

```yaml
data:
  dataset: cifar10
  image_size: 32
  random_flip: True
```

### Handling New Datasets

To add support for a new dataset:

1. **Extend `get_dataset(p)`**: Add conditions to handle your new dataset based on `p['data']['dataset']`.
2. **Implement Necessary Transformations**: Define any specific transformations required.
3. **Update Available Datasets**: Add the new dataset to the list of supported datasets.







# Misc

- **Flexibility**: The framework is designed to be flexible, allowing easy integration of custom models and methods.
- **Modularity**: Components like data handling, training, evaluation, and logging are modular.
- **Extensibility**: By implementing the required functions, you can extend the framework to support new generative models.
- **Device Support**: Automatically detects and utilizes available hardware (CPU, GPU, MPS).
- **Reproducibility**: Experiment parameters are hashed to ensure reproducibility and organized storage.

**Contributing**: Contributions are welcome! Please submit issues or pull requests for bugs, features, or improvements.

**License**: This project is licensed under the MIT License.



