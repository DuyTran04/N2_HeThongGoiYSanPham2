import pandas as pd
import pickle
import os

def train_popularity_model():
    """
    Tạo và lưu popularity model từ clean_amazon.csv
    """
    # Kiểm tra file đầu vào
    if not os.path.exists('data/clean_amazon.csv'):
        print("Error: data/clean_amazon.csv not found")
        return
    
    print("Loading data for popularity model...")
    data = pd.read_csv('data/clean_amazon.csv')
    
    # Chuyển đổi columns thành numeric
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data['rating_count'] = pd.to_numeric(data['rating_count'], errors='coerce')
    
    # Tạo purchase_count_estimated nếu chưa có
    if 'purchase_count_estimated' not in data.columns:
        data['purchase_count_estimated'] = data['rating_count'] * 0.5
    else:
        data['purchase_count_estimated'] = pd.to_numeric(data['purchase_count_estimated'], errors='coerce')
    
    # Fill NA values
    data['rating'] = data['rating'].fillna(0)
    data['rating_count'] = data['rating_count'].fillna(0)
    data['purchase_count_estimated'] = data['purchase_count_estimated'].fillna(0)
    
    # Tính popularity score
    data['popularity_score'] = (
        (data['rating'] * 25) +
        (data['rating_count'] * 1.5) +
        (data['purchase_count_estimated'] * 0.5)
    )
    
    # Sắp xếp theo popularity_score
    popularity_model = data.sort_values('popularity_score', ascending=False)
    
    # Lưu popularity model
    popularity_info = {
        'model': popularity_model,
        'score_formula': {
            'rating_weight': 25,
            'rating_count_weight': 1.5,
            'purchase_count_weight': 0.5
        }
    }
    
    with open('data/popularity_model.pkl', 'wb') as f:
        pickle.dump(popularity_info, f)
    
    print(f"Popularity model saved! Top 5 products:")
    top_5 = popularity_model.head(5)[['product_id', 'product_name', 'popularity_score']]
    print(top_5)
    
    return popularity_model

if __name__ == "__main__":
    train_popularity_model()