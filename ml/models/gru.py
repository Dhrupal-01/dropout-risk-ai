"""
PyTorch GRU Early-Warning Model for OULAD Benchmark
Sequence architecture over weekly clickstream vectors + static features.
Includes sklearn-compatible estimator with internal validation split and early stopping.
"""

import copy
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class PyTorchGRUEarlyWarning(nn.Module):
        """
        PyTorch GRU architecture:
        - Processes weekly sequence (B, W, 1) through GRU layer
        - Extracts final hidden state h_T (B, hidden_dim)
        - Concatenates with static feature vector (B, D_static)
        - Passes through MLP classification head with dropout
        """

        def __init__(self, seq_len: int, static_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
            super().__init__()
            self.seq_len = seq_len
            self.static_dim = static_dim
            self.hidden_dim = hidden_dim

            self.gru = nn.GRU(
                input_size=1,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=1,
            )
            combined_dim = hidden_dim + static_dim
            self.head = nn.Sequential(
                nn.Linear(combined_dim, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
            )

        def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
            # x_seq: (B, W, 1)
            # x_static: (B, D_static)
            out, _ = self.gru(x_seq)
            # Take final time-step hidden state: (B, hidden_dim)
            h_last = out[:, -1, :]
            combined = torch.cat([h_last, x_static], dim=1)
            logits = self.head(combined)
            return logits.squeeze(-1)


class PyTorchGRUEstimator(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible estimator wrapping PyTorchGRUEarlyWarning.
    Fits exclusively on training presentations and carves an internal validation holdout
    from training data for early stopping.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 0.003,
        batch_size: int = 256,
        max_epochs: int = 40,
        patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        self.model_: Optional[nn.Module] = None
        self.classes_ = np.array([0, 1])
        self.scaler_static_ = StandardScaler()
        self.scaler_seq_ = StandardScaler()
        self.seq_cols_: List[str] = []
        self.static_cols_: List[str] = []

    def _split_features(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            if not self.seq_cols_ or not self.static_cols_:
                self.seq_cols_ = [c for c in X.columns if c.startswith("clicks_week_")]
                # Sort numerically by week number
                self.seq_cols_ = sorted(self.seq_cols_, key=lambda c: int(c.split("_")[-1]))
                self.static_cols_ = [c for c in X.columns if c not in self.seq_cols_]
            x_seq = X[self.seq_cols_].values.astype(np.float32)
            x_static = X[self.static_cols_].values.astype(np.float32)
        else:
            # Fallback if raw numpy array: treat first N cols as seq if defined, or 0
            n_seq = len(self.seq_cols_)
            x_seq = X[:, :n_seq].astype(np.float32)
            x_static = X[:, n_seq:].astype(np.float32)

        return x_seq, x_static

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PyTorchGRUEstimator. Install with: pip install -e '.[research]'")

        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y = np.asarray(y, dtype=np.float32)
        x_seq_raw, x_static_raw = self._split_features(X)

        seq_len = x_seq_raw.shape[1]
        static_dim = x_static_raw.shape[1]

        # Standardize features
        # Reshape sequence to (N * W, 1) for fitting scaler, then reshape back
        n_samples = len(x_seq_raw)
        x_seq_scaled = self.scaler_seq_.fit_transform(x_seq_raw.reshape(-1, 1)).reshape(n_samples, seq_len, 1)
        x_static_scaled = self.scaler_static_.fit_transform(x_static_raw)

        # Carve internal validation holdout strictly from training presentations (20% stratified holdout)
        idx_train, idx_val = train_test_split(
            np.arange(n_samples),
            test_size=0.20,
            random_state=self.random_state,
            stratify=y,
        )

        t_x_seq_tr = torch.tensor(x_seq_scaled[idx_train], dtype=torch.float32)
        t_x_static_tr = torch.tensor(x_static_scaled[idx_train], dtype=torch.float32)
        t_y_tr = torch.tensor(y[idx_train], dtype=torch.float32)

        t_x_seq_val = torch.tensor(x_seq_scaled[idx_val], dtype=torch.float32)
        t_x_static_val = torch.tensor(x_static_scaled[idx_val], dtype=torch.float32)
        t_y_val = torch.tensor(y[idx_val], dtype=torch.float32)

        train_dataset = TensorDataset(t_x_seq_tr, t_x_static_tr, t_y_tr)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = PyTorchGRUEarlyWarning(
            seq_len=seq_len,
            static_dim=static_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.2,
        )

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model_.state_dict())
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            for b_seq, b_static, b_y in train_loader:
                optimizer.zero_grad()
                logits = self.model_(b_seq, b_static)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

            # Validation evaluation
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(t_x_seq_val, t_x_static_val)
                val_loss = float(criterion(val_logits, t_y_val).item())

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model_.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        self.model_.load_state_dict(best_weights)
        self.model_.eval()
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not HAS_TORCH or self.model_ is None:
            raise RuntimeError("Estimator has not been fitted or PyTorch is unavailable.")

        x_seq_raw, x_static_raw = self._split_features(X)
        n_samples = len(x_seq_raw)
        seq_len = x_seq_raw.shape[1]

        x_seq_scaled = self.scaler_seq_.transform(x_seq_raw.reshape(-1, 1)).reshape(n_samples, seq_len, 1)
        x_static_scaled = self.scaler_static_.transform(x_static_raw)

        t_x_seq = torch.tensor(x_seq_scaled, dtype=torch.float32)
        t_x_static = torch.tensor(x_static_scaled, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(t_x_seq, t_x_static)
            probs = torch.sigmoid(logits).cpu().numpy()

        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
