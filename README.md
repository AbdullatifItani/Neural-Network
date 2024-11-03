# Car Price Prediction using Deep Learning

This project implements a deep learning solution for predicting car prices based on various features such as kilometers driven, fuel type, transmission, and more. The model uses a feed-forward neural network implemented in PyTorch.

## Dataset

The dataset contains information about used cars including:
- Car name
- Price (in Lakhs)
- Kilometers driven
- Fuel type
- Transmission type
- Ownership details
- Manufacturing year
- Engine capacity
- Number of seats

## Project Structure

```
car-price-prediction/
│
├── data_utils.py        # Data preprocessing utilities
├── model.py            # Neural network model implementation
├── requirements.txt    # Project dependencies
└── demo.ipynb         # Jupyter notebook with usage example
```

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Use the model in your own code:
```python
from data_utils import CarDataProcessor
from model import CarPricePredictor, ModelTrainer
import pandas as pd

# Load and preprocess data
data_processor = CarDataProcessor()
df = pd.read_csv('your_data.csv')
X_train, X_test, y_train, y_test, features = data_processor.prepare_data(df)

# Create and train model
model = CarPricePredictor(input_size=len(features))
trainer = ModelTrainer(model)
train_losses, test_losses = trainer.train(X_train, y_train, X_test, y_test)

# Evaluate model
mse, r2, predictions = trainer.evaluate(X_test, y_test)
```

For a detailed example of usage, please refer to the `demo.ipynb` notebook.

## Model Architecture

The neural network consists of:
- Input layer: Features dimension
- Hidden layer 1: 64 neurons with ReLU activation
- Dropout layer (0.2)
- Hidden layer 2: 32 neurons with ReLU activation
- Dropout layer (0.2)
- Output layer: 1 neuron (price prediction)

The model is trained using:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100

## License

This project is licensed under the MIT License - see the LICENSE file for details.
