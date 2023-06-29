<h1 align="center">
<img src="https://images.squarespace-cdn.com/content/v1/6172d4d5e7f28f6b50179b1b/c3b7cc24-51a1-44aa-8b39-b5e94ac01e8d/image+%282%29.png">
<br>Pipeline VSDI
</h1><br>

Preprocessing tools for voltage sensitive dye imaging data

## Structure

- **VSDI_Session** - Contains classes for handling sessions and full datasets.
- **io** - Contains data loaders and savers.
- **cleaning** - Contains functions to clean and handle the data
- **embedding** - Contains functions to perform linear (PCA-ICA) and nonlinear (convolutional variational autoencoders) dimensionality reduction on the vsdi video data.
- **decoding** - Contains functions to perform decoding of the behavioural output from the neural activity

## Installation

### Prerequisites

- Python 3.7 or higher
- [Poetry](https://python-poetry.org/docs/#installation) Python packaging and dependency management tool.

### Steps

#### 1. Clone the repository

```bash
git clone https://github.com/dabadav/pipeline_vsdi.git
cd pipeline_vsdi
```

#### 2. Install the package using Poetry

Ensure you're in the `pipeline_vsdi` directory (or the directory where you cloned your repository) and run:

```bash
poetry install
```

This command installs all the dependencies specified in the `pyproject.toml` file in a new virtual environment. If you want to use the virtual environment for other tasks, you can activate it using:

```bash
poetry shell
```

After this, you should have all the necessary dependencies installed and be able to use the functionality provided by the Pipeline VSDI.

## Usage

After installation, you can import and use the tools in the Python interpreter or your scripts like:

```python
from vsdi import Session
from preprocessing import clean_data
from dim_reduction import reduce_dimension

# Use the tools as per your requirement.
```