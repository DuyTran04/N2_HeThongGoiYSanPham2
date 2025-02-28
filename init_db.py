import pandas as pd
import sqlite3
import re
import os

def init_database(csv_path='data/amazon.csv', clean_csv_path='data/clean_amazon.csv', db_path='ecommerce.db'):
    """
    Khởi tạo database từ CSV file
    """
    print(f"Initializing database from {csv_path}...")
    
    # Kiểm tra nếu file DB đã tồn tại
    if os.path.exists(db_path):
        print(f"Database file {db_path} already exists. Skipping initialization.")
        return
    
    # Ưu tiên đọc từ clean_amazon.csv nếu có
    if os.path.exists(clean_csv_path):
        print(f"Loading data from {clean_csv_path}...")
        df = pd.read_csv(clean_csv_path)
        
        # Đảm bảo có các cột cần thiết
        if 'purchase_count_estimated' not in df.columns:
            # Chuyển rating_count thành số
            df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
            df['purchase_count_estimated'] = df['rating_count'] * 0.5
    else:
        # Đọc dữ liệu từ file CSV gốc
        print(f"Clean data not found. Loading from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Chuyển rating_count thành số và tạo purchase_count_estimated
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
        df['purchase_count_estimated'] = df['rating_count'] * 0.5
    
    # Đảm bảo các cột số đều là kiểu số
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0)
    df['purchase_count_estimated'] = pd.to_numeric(df['purchase_count_estimated'], errors='coerce').fillna(0)
    
    # Khởi tạo danh sách chứa dữ liệu review
    review_data = []
    
    # Tách dữ liệu vào bảng products và reviews
    for idx, row in df.iterrows():
        product_id = row["product_id"]
        
        # Chuyển đổi các giá trị thành chuỗi và tách theo dấu phẩy
        if "review_id" in df.columns:
            review_ids = str(row["review_id"]).split(",") if pd.notna(row["review_id"]) else []
            user_ids = str(row["user_id"]).split(",") if pd.notna(row["user_id"]) else []
            user_names = str(row["user_name"]).split(",") if pd.notna(row["user_name"]) else []
            review_titles = str(row["review_title"]).split(",") if pd.notna(row["review_title"]) else []
            review_contents = str(row["review_content"]).split(",") if pd.notna(row["review_content"]) else []
            
            # Xác định số lượng review (dựa vào số lượng nhỏ nhất)
            n = min(len(review_ids), len(user_ids), len(user_names), len(review_titles), len(review_contents))
            
            for i in range(n):
                review_data.append({
                    "review_id": review_ids[i].strip(),
                    "product_id": product_id,
                    "user_id": user_ids[i].strip(),
                    "user_name": user_names[i].strip(),
                    "review_title": review_titles[i].strip(),
                    "review_content": review_contents[i].strip()
                })
    
    # Chuyển review_data thành DataFrame
    if review_data:
        df_reviews = pd.DataFrame(review_data)
        df_reviews = df_reviews.drop_duplicates()
    else:
        df_reviews = pd.DataFrame(columns=["review_id", "product_id", "user_id", "user_name", "review_title", "review_content"])
    
    # Lọc ra bảng sản phẩm (loại bỏ các cột liên quan đến review nếu cần)
    review_columns = ["review_id", "review_title", "review_content"]
    product_columns = [col for col in df.columns if col not in review_columns]
    df_products = df[product_columns].drop_duplicates()
    
    # Thêm từ khóa tìm kiếm
    df_products["search_keywords"] = df_products.apply(generate_search_keywords, axis=1)
    
    # Tính final popularity score
    df_products['final_popularity_score'] = (
        (df_products['rating'] * 25) +
        (df_products['rating_count'] * 1.5) +
        (df_products['purchase_count_estimated'] * 0.5)
    )
    
    print(f"Total records after removing duplicates:")
    print(f"Products: {len(df_products)}")
    print(f"Reviews: {len(df_reviews)}")
    
    # Tạo database và các bảng trong SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tạo bảng products
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            discounted_price REAL,
            actual_price REAL,
            discount_percentage REAL,
            rating REAL,
            rating_count INTEGER,
            about_product TEXT,
            img_link TEXT,
            product_link TEXT,
            user_id TEXT,
            search_keywords TEXT,
            purchase_count_estimated REAL,
            final_popularity_score REAL
        )
    ''')
    
    # Tạo bảng reviews nếu có dữ liệu review
    if not df_reviews.empty:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                product_id TEXT,
                user_id TEXT,
                user_name TEXT,
                review_title TEXT,
                review_content TEXT,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        ''')
    
    conn.commit()
    
    # Lưu dữ liệu vào SQLite
    df_products.to_sql("products", conn, if_exists="replace", index=False)
    if not df_reviews.empty:
        df_reviews.to_sql("reviews", conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized successfully in {db_path}!")

def generate_search_keywords(row):
    """Hàm tạo từ khóa tìm kiếm từ tên sản phẩm, danh mục, mô tả."""
    columns = ['product_name', 'category', 'about_product']
    text_columns = [col for col in columns if col in row.index]
    text = " ".join([str(row[col]) for col in text_columns if pd.notna(row[col])])
    
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = " ".join(set(text.split()))  # Loại bỏ từ trùng lặp
    return text

if __name__ == "__main__":
    init_database()