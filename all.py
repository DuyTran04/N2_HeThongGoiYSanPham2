# -*- coding: utf-8 -*-
"""
Amazon Product Recommender System
Using Deep Matrix Factorization and Popularity Model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
from tqdm import tqdm
import pickle
import random
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class DataProcessor:
    """
    Class for data preprocessing and cleaning
    """
    @staticmethod
    def clean_amazon_data(df):
        """
        Clean and preprocess the Amazon dataset
        """
        print("Cleaning Amazon data...")
        # Create a copy to avoid affecting the original data
        df_clean = df.copy()
        
        # Process ID columns
        id_columns = ['product_id', 'user_id', 'review_id']
        for col in id_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
        
        # Process text columns
        text_columns = ['product_name', 'category', 'about_product', 'user_name',
                      'review_title', 'review_content', 'img_link', 'product_link']
        for col in text_columns:
            if col in df_clean.columns:
                # Remove extra whitespace
                df_clean[col] = df_clean[col].str.strip()
                # Fill null values with 'Unknown'
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Process numeric columns
        numeric_columns = ['discounted_price', 'actual_price', 'discount_percentage',
                          'rating', 'rating_count']
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convert to numeric type
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Ensure valid values
        if all(col in df_clean.columns for col in numeric_columns):
            df_clean = df_clean[
                (df_clean["discounted_price"] >= 0) &
                (df_clean["actual_price"] >= 0) &
                (df_clean["discount_percentage"].between(0, 100)) &
                (df_clean["rating"].between(0, 5)) &
                (df_clean["rating_count"] >= 0)
            ]
        
        # Convert rating to numeric (again, to be sure)
        df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
        
        return df_clean
    
    @staticmethod
    def encode_categorical_columns(df):
        """
        Encode categorical columns using LabelEncoder
        """
        df_encoded = df.copy()
        label_encoders = {}
        
        # Columns to encode
        categorical_columns = ["product_id", "user_id", "category", "product_name"]
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le  # Save encoders for later use
        
        return df_encoded, label_encoders
    
    @staticmethod
    def prepare_for_popularity_model(df):
        """
        Prepare data for the popularity model
        """
        print("Preparing data for popularity model...")
        df_pop = df.copy()
        
        # Create purchase_count_estimated if it doesn't exist
        if 'purchase_count_estimated' not in df_pop.columns:
            df_pop['purchase_count_estimated'] = df_pop['rating_count'] * 0.5
            
        # Replace zero values with 1 (for log transformation)
        df_pop['purchase_count_estimated'] = df_pop['purchase_count_estimated'].replace(0, 1)
        
        # Apply log transformation
        df_pop['log_purchase_count'] = np.log(df_pop['purchase_count_estimated'])
        
        # Create event type classification
        df_pop['eventType'] = df_pop.apply(DataProcessor._classify_event, axis=1)
        
        # Define event strength based on event type
        event_type_strength = {
            'event_type_1': 1.0,
            'event_type_2': 2.0,
            'event_type_3': 3.0,
            'event_type_4': 4.0
        }
        
        df_pop['eventStrength'] = df_pop['eventType'].apply(lambda x: event_type_strength.get(x, 1.0))
        
        return df_pop
    
    @staticmethod
    def _classify_event(row):
        """
        Classify events based on rating, rating_count, and purchase_count_estimated
        """
        if 'rating' not in row or 'rating_count' not in row or 'purchase_count_estimated' not in row:
            return 'event_type_1'
            
        if row['rating'] >= 4.5 and row['rating_count'] > 5000 and row['purchase_count_estimated'] > 1000:
            return 'event_type_4'  # Hot product
        elif row['rating'] >= 4.0 and row['rating_count'] > 1000 and row['purchase_count_estimated'] > 500:
            return 'event_type_3'  # Popular product
        elif row['rating'] >= 3.0 and row['rating_count'] > 100 and row['purchase_count_estimated'] > 100:
            return 'event_type_2'  # Average product
        else:
            return 'event_type_1'  # Less popular product

    @staticmethod
    def rating_bin(rating):
        """Helper function to bin ratings into categories"""
        if rating >= 4.5:
            return 3  # Very high
        elif rating >= 4.0:
            return 2  # High
        elif rating >= 3.0:
            return 1  # Medium
        else:
            return 0.5  # Low
    
    @staticmethod
    def create_popularity_dataframe(df):
        """
        Create a popularity DataFrame with all necessary metrics
        """
        # Create popularity_df with basic metrics
        popularity_df = df.groupby('product_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        # Flatten column names
        popularity_df.columns = ['product_id', 'rating_count', 'avg_rating']
        
        # Add rating_bin
        popularity_df['rating_bin'] = popularity_df['avg_rating'].apply(DataProcessor.rating_bin)
        
        # Calculate purchase_count if it exists, otherwise use rating_count
        if 'purchase_count_estimated' in df.columns:
            purchase_counts = df.groupby('product_id')['purchase_count_estimated'].sum().reset_index()
            popularity_df = popularity_df.merge(purchase_counts, on='product_id', how='left')
        else:
            # If purchase_count_estimated doesn't exist, use rating_count instead
            popularity_df['purchase_count_estimated'] = popularity_df['rating_count']
        
        # Calculate final_event_strength
        popularity_df['final_event_strength'] = (
            (popularity_df['avg_rating'] * 25) +
            (popularity_df['rating_bin'] * 2) +
            (np.log1p(popularity_df['rating_count']) * 1.5) +
            (np.log1p(popularity_df['purchase_count_estimated']) * 0.5)
        )
        
        # Ensure product_id is int
        popularity_df['product_id'] = popularity_df['product_id'].astype(int)
        
        return popularity_df

# Deep Matrix Factorization Model
class DeepMatrixFactorization(nn.Module):
    """
    Deep Matrix Factorization model using PyTorch
    """
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
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        """Forward pass through the network"""
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
    """Trainer class for Deep Matrix Factorization"""
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc="Training"):
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
        """Evaluate the model"""
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
    """Dataset class for recommender systems"""
    def __init__(self, df, rating_range=5.0):
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['product_id'].values)
        self.ratings = torch.FloatTensor(df['rating'].values) / rating_range

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class PopularityRecommender:
    """Popularity-based recommender system"""
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, threshold=3.0):
        """Initialize the recommender system"""
        self.popularity_df = popularity_df.copy()
        
        # Ensure product_id is int
        self.popularity_df['product_id'] = self.popularity_df['product_id'].astype(int)
        
        # Filter products with average rating > threshold
        self.popularity_df = self.popularity_df[self.popularity_df['avg_rating'] > threshold]
        
        # Normalize final_event_strength
        if 'final_event_strength' in self.popularity_df.columns:
            scaler = MinMaxScaler()
            self.popularity_df['final_event_strength'] = scaler.fit_transform(
                self.popularity_df[['final_event_strength']]
            )

    def get_model_name(self):
        """Return model name"""
        return self.MODEL_NAME

    def recommend_items(self, user_id=None, items_to_ignore=[], topn=10, verbose=False):
        """Provide recommendations"""
        recommendations_df = self.popularity_df[
            ~self.popularity_df['product_id'].isin(items_to_ignore)
        ].sort_values('final_event_strength', ascending=False).head(topn)
        
        return recommendations_df

def train_deep_mf_model(data, num_epochs=20, batch_size=128, save_path="dmf_model.pkl"):
    """
    Train the Deep Matrix Factorization model
    """
    print("\n=== Training Deep Matrix Factorization Model ===")
    
    # Split data into train and test sets
    Train, Test = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train size: {len(Train)}, Test size: {len(Test)}")
    
    # Create datasets and dataloaders
    train_dataset = RecommenderDataset(Train)
    test_dataset = RecommenderDataset(Test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    # Initialize model
    n_users = len(data['user_id'].unique())
    n_items = len(data['product_id'].unique())
    model = DeepMatrixFactorization(n_users, n_items)
    trainer = DMFTrainer(model)
    
    # Training loop
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_loss, test_acc = trainer.evaluate(test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('dmf_training_results.png')
    plt.close()
    
    # Save model and training info
    save_info = {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'n_users': n_users,
        'n_items': n_items
    }
    
    # Save encoder information if available
    if 'label_encoders' in globals():
        save_info['user_encoder'] = label_encoders.get('user_id')
        save_info['product_encoder'] = label_encoders.get('product_id')
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_info, f)
    print(f"\nModel saved to {save_path}")
    
    return model, save_info

def train_popularity_model(data, threshold=3.0, save_path="popularity_model.pkl"):
    """
    Train the Popularity Model
    """
    print("\n=== Training Popularity Model ===")
    
    # Create popularity DataFrame
    popularity_df = DataProcessor.create_popularity_dataframe(data)
    
    # Initialize and train model
    model = PopularityRecommender(popularity_df, threshold=threshold)
    
    # Save model
    save_info = {
        'popularity_df': model.popularity_df,
        'threshold': threshold
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_info, f)
    print(f"\nModel saved to {save_path}")
    
    return model, save_info

def get_recommendations_dmf(model, product_id, n_recommendations=5, data=None, full_data=None):
    """
    Get recommendations using Deep Matrix Factorization
    """
    if data is None or full_data is None:
        raise ValueError("data and full_data must be provided")
    
    model.eval()
    with torch.no_grad():
        # Get all product embeddings
        all_products = torch.arange(len(data['product_id'].unique()))
        target_product = torch.tensor([product_id])
        
        # Get predictions
        similarities = model(
            target_product.repeat(len(all_products)),
            all_products
        ).numpy()
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-n_recommendations-1:][::-1]
        # Remove the input product if it's in the recommendations
        top_indices = top_indices[top_indices != product_id][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            # Get product details from the original dataframe
            if 'label_encoders' in globals() and 'product_id' in label_encoders:
                original_product_id = label_encoders['product_id'].inverse_transform([idx])[0]
                recommended_product_details = full_data[full_data['product_id'] == original_product_id].iloc[0]
            else:
                recommended_product_details = full_data[full_data['product_id'] == idx].iloc[0]
            
            recommendation_info = {
                'product_id': idx,
                'product_name': recommended_product_details.get('product_name', 'Unknown'),
                'category': recommended_product_details.get('category', 'Unknown'),
                'rating': recommended_product_details.get('rating', 0),
                'similarity': similarities[idx]
            }
            recommendations.append(recommendation_info)
    
    return recommendations

def get_recommendations_popularity(model, n_recommendations=5):
    """
    Get recommendations using the Popularity Model
    """
    return model.recommend_items(topn=n_recommendations)

def main():
    """
    Main function to run the recommendation system
    """
    print("=== Amazon Product Recommendation System ===")
    
    # Check if data files exist
    if not os.path.exists("amazon.csv") and not os.path.exists("clean_amazon.csv"):
        print("Error: Required data files not found.")
        return
    
    # Load data
    if os.path.exists("clean_amazon.csv"):
        print("Loading clean_amazon.csv...")
        data = pd.read_csv("clean_amazon.csv")
    else:
        print("Loading and cleaning amazon.csv...")
        data = pd.read_csv("amazon.csv")
        data = DataProcessor.clean_amazon_data(data)
    
    # Keep a copy of the original data
    full_data = data.copy()
    
    # Encode categorical columns
    data, label_encoders = DataProcessor.encode_categorical_columns(data)
    
    # Prepare data for models
    data = DataProcessor.prepare_for_popularity_model(data)
    
    # Train Deep Matrix Factorization Model
    dmf_model, _ = train_deep_mf_model(data, num_epochs=20)
    
    # Train Popularity Model
    pop_model, _ = train_popularity_model(data)
    
    # Get and display recommendations
    print("\n=== Deep Matrix Factorization Recommendations ===")
    # Use the first product ID as an example
    sample_product_id = 0
    dmf_recommendations = get_recommendations_dmf(dmf_model, sample_product_id, 
                                               data=data, full_data=full_data)
    for i, rec in enumerate(dmf_recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Product ID: {rec['product_id']}")
        print(f"Product Name: {rec['product_name']}")
        print(f"Category: {rec['category']}")
        print(f"Rating: {rec['rating']}")
        print(f"Similarity Score: {rec['similarity']:.4f}")
    
    print("\n=== Popularity Model Recommendations ===")
    pop_recommendations = get_recommendations_popularity(pop_model, 5)
    print(pop_recommendations[['product_id', 'final_event_strength']])
    
    print("\nRecommendation system completed successfully!")

if __name__ == "__main__":
    main()
