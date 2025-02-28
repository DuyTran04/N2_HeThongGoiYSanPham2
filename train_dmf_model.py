import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models.deep_matrix_factorization import DeepMatrixFactorization
from torch.utils.data import DataLoader, Dataset

# Class dataset cho huấn luyện
class RecommenderDataset(torch.utils.data.Dataset):
    def __init__(self, df, rating_range=5.0):
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['product_id'].values)
        self.ratings = torch.FloatTensor(df['rating'].values) / rating_range
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Class DMFTrainer để huấn luyện mô hình
class DMFTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.BCELoss()
        
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

def main():
    # Tải dữ liệu từ amazon.csv
    print("Loading data...")
    try:
        data = pd.read_csv('data/amazon.csv')
    except:
        print("Error loading amazon.csv. Please make sure the file exists in the data directory.")
        return
    
    # Tiền xử lý dữ liệu
    print("Preprocessing data...")
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating', 'user_id', 'product_id'])
    
    # Encode user_id và product_id
    user_id_encoder = LabelEncoder()
    product_id_encoder = LabelEncoder()
    
    data['user_id_encoded'] = user_id_encoder.fit_transform(data['user_id'])
    data['product_id_encoded'] = product_id_encoder.fit_transform(data['product_id'])
    
    # Chia dữ liệu thành train và validation
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Tạo dataset và dataloader
    train_dataset = RecommenderDataset(
        pd.DataFrame({
            'user_id': train_data['user_id_encoded'],
            'product_id': train_data['product_id_encoded'],
            'rating': train_data['rating']
        })
    )
    
    val_dataset = RecommenderDataset(
        pd.DataFrame({
            'user_id': val_data['user_id_encoded'],
            'product_id': val_data['product_id_encoded'],
            'rating': val_data['rating']
        })
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    
    # Khởi tạo mô hình
    print("Initializing model...")
    n_users = len(data['user_id_encoded'].unique())
    n_items = len(data['product_id_encoded'].unique())
    
    model = DeepMatrixFactorization(n_users, n_items)
    trainer = DMFTrainer(model)
    
    # Huấn luyện mô hình
    print("Training model...")
    num_epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Lưu mô hình tốt nhất dựa trên validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Lưu mô hình
            save_info = {
                'model_state_dict': model.state_dict(),
                'n_users': n_users,
                'n_items': n_items,
                'user_encoder': user_id_encoder,
                'product_encoder': product_id_encoder
            }
            
            with open('data/dmf_model.pkl', 'wb') as f:
                pickle.dump(save_info, f)
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()