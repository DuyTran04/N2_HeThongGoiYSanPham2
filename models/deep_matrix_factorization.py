import torch
import torch.nn as nn
import numpy as np

class DeepMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, factors=[64, 32, 16, 8]):
        super(DeepMatrixFactorization, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, factors[0])
        self.item_embedding = nn.Embedding(n_items, factors[0])
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(factors[0], factors[1]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[1]),
            nn.Dropout(0.2),
            nn.Linear(factors[1], factors[2]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[2]),
            nn.Dropout(0.2),
            nn.Linear(factors[2], factors[3])
        )
        
        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(factors[0], factors[1]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[1]),
            nn.Dropout(0.2),
            nn.Linear(factors[1], factors[2]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[2]),
            nn.Dropout(0.2),
            nn.Linear(factors[2], factors[3])
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        
        # Pass through towers
        user_vector = self.user_tower(user_embedded)
        item_vector = self.item_tower(item_embedded)
        
        # Normalize embeddings
        user_vector = nn.functional.normalize(user_vector, p=2, dim=1)
        item_vector = nn.functional.normalize(item_vector, p=2, dim=1)
        
        # Compute prediction
        prediction = torch.sum(user_vector * item_vector, dim=1)
        return torch.sigmoid(prediction)

class DMFTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch in train_loader:
            user_ids, item_ids, ratings = batch
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            # Accuracy
            predicted_labels = (predictions >= 0.5).float()
            correct_predictions += (predicted_labels == ratings).sum().item()
            total_samples += ratings.size(0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader), correct_predictions / total_samples

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids, item_ids, ratings = batch
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                predicted_labels = (predictions >= 0.5).float()
                correct_predictions += (predicted_labels == ratings).sum().item()
                total_samples += ratings.size(0)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader), correct_predictions / total_samples

class RecommenderDataset(torch.utils.data.Dataset):
    def __init__(self, df, rating_range=5.0):
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['product_id'].values)
        self.ratings = torch.FloatTensor(df['rating'].values) / rating_range
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]