import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class ModelTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        # Initialize the optimizer with weight_decay for L2 regularization
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Adjust the weight_decay as necessary
        
    def train(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test_tensor)
                test_loss = self.criterion(test_outputs, y_test_tensor).item()
            
            train_losses.append(train_loss/len(X_train))
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(X_train):.4f}, Test Loss: {test_loss:.4f}')
        
        return train_losses, test_losses
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance using MSE and R2 score
        """
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            predictions = self.model(X_test_tensor).numpy()
            
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return mse, mae, r2, predictions