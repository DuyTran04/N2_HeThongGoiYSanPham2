import argparse
import os

def train_all_models():
    """
    Huấn luyện tất cả các mô hình cần thiết
    """
    print("Starting training of all recommendation models...")
    
    # Kiểm tra thư mục data
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")
    
    # Kiểm tra file dữ liệu
    required_files = ['data/amazon.csv', 'data/clean_amazon.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Warning: The following required files are missing: {missing_files}")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Huấn luyện Popularity Model
    print("\n=== Training Popularity Model ===")
    from train_popularity_model import train_popularity_model
    train_popularity_model()
    
    # Huấn luyện DMF Model
    print("\n=== Training Deep Matrix Factorization Model ===")
    from train_dmf_model import main as train_dmf
    train_dmf()
    
    print("\nAll models have been trained!")
    
    # Kiểm tra các file đã được tạo
    expected_outputs = ['data/popularity_model.pkl', 'data/dmf_model.pkl']
    missing_outputs = [f for f in expected_outputs if not os.path.exists(f)]
    
    if missing_outputs:
        print(f"Warning: The following output files were not created: {missing_outputs}")
    else:
        print("All model files were successfully created.")

if __name__ == "__main__":
    train_all_models()