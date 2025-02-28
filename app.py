from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import os
import pickle
import sqlite3
from models.deep_matrix_factorization import DeepMatrixFactorization
from models.hybrid_recommender import HybridRecommender
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from error_handlers import register_error_handlers
from init_db import init_database

app = Flask(__name__)
app.secret_key = "ecommerce_recommendation_system"  # Cần thiết cho session

# Đăng ký error handlers
register_error_handlers(app)

# Khởi tạo database nếu chưa tồn tại
db_path = 'ecommerce.db'
if not os.path.exists(db_path):
    init_database(csv_path='data/amazon.csv', db_path=db_path)

# Tải dữ liệu từ database
def load_data_from_db():
    conn = sqlite3.connect(db_path)
    data = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return data

# Tải dữ liệu từ CSV (fallback nếu không có DB)
def load_data_from_csv():
    try:
        # Tải dữ liệu chính cho popularity model
        data = pd.read_csv('data/clean_amazon.csv')
        return data
    except:
        # Fallback nếu không tìm thấy clean_amazon.csv
        data = pd.read_csv('data/amazon.csv')
        return data

# Tải dữ liệu keywords cho cá nhân hóa
def load_keywords_data():
    try:
        keywords_data = pd.read_csv('data/amazon_keywords.csv')
        return keywords_data
    except:
        app.logger.warning("Keywords data file not found. Personalization will be limited.")
        return None

# Tải dữ liệu và mô hình khi khởi động ứng dụng
try:
    data = load_data_from_db()
    app.logger.info("Data loaded from database successfully.")
except Exception as e:
    app.logger.error(f"Error loading data from database: {e}")
    data = load_data_from_csv()
    app.logger.info("Data loaded from CSV successfully.")

# Tải keywords data
keywords_data = load_keywords_data()

# Khởi tạo hybrid recommender với cả hai mô hình
hybrid_recommender = HybridRecommender(
    data=data,
    keywords_data=keywords_data,
    dmf_model_path='data/dmf_model.pkl',
    popularity_model_path='data/popularity_model.pkl',
    dmf_weight=0.7,
    pop_weight=0.3
)

# Route chính - trang chủ
@app.route('/')
def home():
    # Lấy tất cả các category và sắp xếp theo bảng chữ cái
    categories = sorted(data['category'].unique())
    categories.insert(0, "All")  # Thêm tùy chọn "All" cho tất cả danh mục
    return render_template('index.html', categories=categories)

# API để lấy sản phẩm theo category
@app.route('/filter_products', methods=['POST'])
def filter_products():
    selected_category = request.form.get('category')
    
    # Lọc sản phẩm theo category hoặc lấy tất cả nếu chọn "All"
    if selected_category and selected_category != "All":
        filtered_data = data[data['category'] == selected_category]
    else:
        filtered_data = data
    
    products = sorted(filtered_data['product_name'].values)
    
    return jsonify({'products': products})

# API để lấy chi tiết sản phẩm
@app.route('/product_details', methods=['POST'])
def product_details():
    selected_product_name = request.form.get('product')
    filtered_data = data[data['product_name'] == selected_product_name]
    
    if len(filtered_data) > 0:
        # Lấy row đầu tiên thành dictionary
        product_details = filtered_data.iloc[0].to_dict()
        
        # Lưu product_id vào session để sử dụng sau
        session['selected_product_id'] = product_details['product_id']
        session['selected_category'] = product_details['category']
        
        return jsonify({
            'product_id': product_details['product_id'],
            'product_name': product_details['product_name'],
            'category': product_details['category'],
            'rating': product_details['rating'],
            'rating_count': product_details['rating_count'],
            'img_link': product_details['img_link'],
            'product_link': product_details['product_link']
        })
    else:
        return jsonify({'error': 'Product not found'})

# API để lấy khuyến nghị
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    # Lấy product_id từ session hoặc form
    product_id = session.get('selected_product_id')
    category = session.get('selected_category')
    search_keywords = request.form.get('search_keywords')
    
    if not product_id and not search_keywords:
        return jsonify({'error': 'No product ID or search keywords provided'})
    
    try:
        # Sử dụng hybrid recommender để lấy khuyến nghị
        recommendations = hybrid_recommender.get_hybrid_recommendations(
            product_id=product_id,
            category=category,
            search_keywords=search_keywords,
            n_recommendations=5
        )
        
        if not recommendations:
            return jsonify({'error': 'No recommendations found'})
            
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        app.logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)})

# Thêm route để tìm kiếm sản phẩm
@app.route('/search', methods=['POST'])
def search():
   search_keywords = request.form.get('search_keywords', '')
   
   if search_keywords:
       session['search_keywords'] = search_keywords
       
       try:
           # Lấy recommendations dựa trên từ khóa
           recommendations = hybrid_recommender.get_personalized_recommendations(
               search_keywords=search_keywords,
               n_recommendations=5
           )
           
           return jsonify({'recommendations': recommendations})
       except Exception as e:
           app.logger.error(f"Error searching with keywords: {e}")
           return jsonify({'error': str(e)})
   else:
       return jsonify({'error': 'No search keywords provided'})

# Thêm route để kiểm tra trạng thái hệ thống
@app.route('/health')
def health_check():
   return jsonify({
       'status': 'ok',
       'data_size': len(data),
       'categories': len(data['category'].unique()),
       'products': len(data['product_id'].unique()),
       'keywords_data': 'loaded' if keywords_data is not None else 'not loaded',
       'dmf_model': 'loaded' if hybrid_recommender.dmf_model is not None else 'not loaded',
       'popularity_model': 'loaded' if hybrid_recommender.popularity_model is not None else 'not loaded'
   })

if __name__ == '__main__':
   app.run(debug=True)