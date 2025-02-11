import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import numpy as np


class RiskPredictionNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 1e-3):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input tensor.
        hidden_dim : int
            Dimension of hidden layer.
        lr : float
            Learning rate for optimizer.
        """
        super(RiskPredictionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output risk probability
        )

        # Loss function & optimizer
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1),
            each value represents the risk
            of event.
        """
        return self.fc(x)

    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    num_epochs: int=50, batch_size: int=32) -> None:
        """
        Train the risk model on given data.
        
        Parameters
        ----------
        X_train : torch.Tensor
            Input tensor of shape (n_samples, input_dim).
        y_train : torch.Tensor
            Target tensor of shape (n_samples,).
            Each value is either 0 or 1.
            0 - not event-related, 1 - event-related.
        num_epochs : int
            Number of epochs for training.
        batch_size : int
            Batch size for training
        """
        # Convert data to PyTorch tensors
        y_train = y_train.unsqueeze(1).float()

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                risk_pred = self.forward(X_batch)
                loss = self.criterion(risk_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def predict_risk(
            self,
            X_test: torch.Tensor, threshold: float=0.5
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts risk probability and applies a threshold to classify events.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test data of shape (n_samples, input_dim).
        threshold : float
            Threshold value to classify risk probabilities.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (risk_prob, risk_labels).
            risk_prob : np.ndarray
                Probability of event risk for each sample.
            risk_labels : np.ndarray
                Binary labels based on the threshold.
        """
        # Convert test data to tensor
        X_test = X_test.to(dtype=torch.float32)

        # Forward pass to predict risk probabilities
        with torch.no_grad():
            risk_prob = self.forward(X_test).numpy().flatten()

        # Apply threshold to convert probabilities to binary labels
        risk_labels = (risk_prob > threshold).astype(int)

        return risk_prob, risk_labels