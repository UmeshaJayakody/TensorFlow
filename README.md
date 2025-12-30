# TensorFlow Learning

This repository contains educational materials and code examples for learning TensorFlow, a popular open-source machine learning framework developed by Google. It provides hands-on examples covering fundamental machine learning concepts, TensorFlow basics, and core learning algorithms.

## Overview

This project is designed to help beginners understand the fundamentals of machine learning and TensorFlow 2.x. It includes:

- Explanations of different types of machine learning
- TensorFlow tensor operations and concepts
- Core machine learning algorithms with practical implementations
- Data preprocessing and feature engineering
- Model training and evaluation examples

## Prerequisites

Before starting, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd TensorFlow
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The requirements include:
   - numpy: Numerical computing
   - pandas: Data manipulation
   - matplotlib: Data visualization
   - ipython: Enhanced Python shell
   - six: Python 2/3 compatibility
   - tensorflow: Machine learning framework
   - gym: Reinforcement learning environments

## Project Structure

```
TensorFlow/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── text.txt                  # Additional notes
├── Code/                     # Python script examples
│   ├── add_data.py          # Data augmentation for Titanic dataset
│   ├── module_02.py         # TensorFlow tensor operations
│   ├── module_03_1.py       # Basic plotting and linear regression
│   ├── module_03_2.py       # Titanic survival prediction (linear vs neural network)
│   ├── module_03_3.py       # Iris classification (linear vs deep neural network)
│   ├── module_03_4.py       # Hidden Markov Models
│   ├── module_04_1.py       # Neural Networks with Fashion MNIST
│   └── graph_output/        # Generated plots and visualizations
│       ├── module_03/
│       └── module_04/
├── data/                    # Datasets
│   ├── iris/                # Iris flower dataset
│   │   ├── iris_test.csv
│   │   └── iris_training.csv
│   └── titanic/             # Titanic passenger dataset
│       ├── eval.csv
│       ├── generator.py     # Data preprocessing script
│       ├── titanic.csv      # Raw data
│       └── train.csv
├── Note/                    # Jupyter notebooks
│   ├── 01_introduction.ipynb          # Types of machine learning
│   ├── 02_Tensorfloe_introduction.ipynb  # TensorFlow fundamentals
│   ├── 03_Alternative.ipynb           # Alternative explanations of core algorithms
│   ├── 03_Core_Learning_Algorithms.ipynb # Core ML algorithms
│   └── 04_Neural_Networks.ipynb       # Neural Networks tutorial
└── .venv/                   # Virtual environment (created during setup)
```

## Content Overview

### Module 1: Types of Machine Learning
- **Supervised Learning**: Learning with labeled data (e.g., classification, regression)
- **Unsupervised Learning**: Finding patterns in unlabeled data (e.g., clustering)
- **Reinforcement Learning**: Learning through trial and error with rewards

### Module 2: TensorFlow Fundamentals
- Tensor creation and data types
- Tensor rank and shape
- Reshaping and slicing operations
- Special tensor types (zeros, ones, random, etc.)
- Type casting and evaluation

### Module 3: Core Learning Algorithms

#### Linear Regression
- Predicting numeric values using linear relationships
- Line of best fit concepts
- Implementation on Titanic survival prediction

#### Classification
- Binary and multi-class classification
- Logistic regression vs. neural networks
- Feature encoding and preprocessing

#### Clustering
- Grouping similar data points
- Unsupervised learning techniques

#### Hidden Markov Models
- Sequential data modeling
- Time-series prediction
- Probability distributions and state transitions

### Module 4: Neural Networks
- Introduction to Keras API
- Building and training neural networks
- Fashion MNIST classification example
- Model evaluation and predictions
- Visualization of results

## Datasets

### Titanic Dataset
- Passenger survival prediction
- Features: age, sex, class, fare, etc.
- Used for binary classification examples
- Includes data preprocessing and augmentation scripts

### Iris Dataset
- Flower species classification
- Features: sepal/petal dimensions
- Used for multi-class classification examples

## Usage

### Running Python Scripts

1. Navigate to the Code directory:
   ```bash
   cd Code
   ```

2. Run individual modules:
   ```bash
   python module_02.py    # TensorFlow tensor basics
   python module_03_1.py  # Basic plotting
   python module_03_2.py  # Titanic classification
   python module_03_3.py  # Iris classification
   python module_03_4.py  # Hidden Markov Models
   python module_04_1.py  # Neural Networks with Fashion MNIST
   ```

3. For data augmentation:
   ```bash
   python add_data.py     # Add synthetic data to Titanic dataset
   ```

### Jupyter Notebooks

1. Start Jupyter notebook server:
   ```bash
   jupyter notebook
   ```

2. Open notebooks in the Note/ directory:
   - `01_introduction.ipynb`: Learn about ML types
   - `02_Tensorfloe_introduction.ipynb`: Interactive TensorFlow tutorial
   - `03_Alternative.ipynb`: Simplified explanations of core algorithms
   - `03_Core_Learning_Algorithms.ipynb`: Core algorithms walkthrough
   - `04_Neural_Networks.ipynb`: Neural Networks with Fashion MNIST

### Data Processing

1. Navigate to data directory:
   ```bash
   cd data/titanic
   ```

2. Run data preprocessing:
   ```bash
   python generator.py    # Process raw Titanic data into train/eval splits
   ```

## Key Examples

### Titanic Survival Prediction
- **Linear Classifier**: Logistic regression implementation
- **Neural Network**: Multi-layer perceptron with dropout
- **Comparison**: Performance metrics and visualization
- **EDA**: Exploratory data analysis plots

### Iris Species Classification
- **Linear Classifier**: Basic classification model
- **Deep Neural Network**: Multi-layer architecture
- **Evaluation**: Accuracy comparison and predictions
- **Interactive**: User input prediction feature

### Fashion MNIST Classification
- **Neural Network**: Multi-layer perceptron for image classification
- **Dataset**: 10 clothing categories with 28x28 grayscale images
- **Training**: Model compilation, fitting, and evaluation
- **Predictions**: Visualizing predictions vs actual labels

## Generated Outputs

Running the scripts will create:
- `graph_output/module_03/plot_*.png`: Various data visualizations
- `graph_output/module_04/sample_image.png`: Fashion MNIST sample images
- `graph_output/module_04/sample_image_test.png`: Test predictions visualization
- Model training logs and performance metrics
- Prediction examples and comparisons

## Learning Path

1. **Start Here**: Read `01_introduction.ipynb` for ML concepts
2. **TensorFlow Basics**: Study `02_Tensorfloe_introduction.ipynb` and `module_02.py`
3. **Hands-on Practice**: Run `module_03_1.py` for basic plotting
4. **Classification Examples**: Execute `module_03_2.py` and `module_03_3.py`
5. **Advanced Algorithms**: Try `module_03_4.py` for Hidden Markov Models
6. **Neural Networks**: Explore `04_Neural_Networks.ipynb` and `module_04_1.py`
7. **Alternative Explanations**: Review `03_Alternative.ipynb` for simplified concepts

## Contributing

Feel free to contribute by:
- Adding more modules or examples
- Improving explanations and documentation
- Providing additional code examples
- Fixing bugs or enhancing features
- Creating new datasets or preprocessing scripts

## Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Google Colab](https://colab.research.google.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## License

This project is for educational purposes. Please refer to TensorFlow's license for any code usage.

## Acknowledgments

This project builds upon official TensorFlow tutorials and documentation, adapted for educational purposes with additional examples and explanations.

## Referances

- TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial - freeCodeCamp.org