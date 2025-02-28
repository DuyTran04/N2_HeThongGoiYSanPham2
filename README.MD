# Hệ thống gợi ý thương mại điện tử (E-commerce Recommendation System)

Hệ thống gợi ý sản phẩm sử dụng kết hợp hai mô hình: Deep Matrix Factorization và Popularity Model, được phát triển với Flask framework.

## Tổng quan

Hệ thống này sử dụng hai phương pháp gợi ý chính:

1. **Deep Matrix Factorization (DMF)**: Phương pháp này sử dụng mạng neural để hiểu các mẫu phức tạp trong tương tác người dùng-sản phẩm, cung cấp gợi ý cá nhân hóa sâu.

2. **Popularity Model**: Phương pháp này gợi ý các sản phẩm phổ biến dựa trên chỉ số đánh giá, số lượng đánh giá và số lượng mua ước tính.

Ngoài ra, hệ thống cũng hỗ trợ tìm kiếm và gợi ý dựa trên từ khóa sử dụng dữ liệu từ `amazon_keywords.csv`.

## Cấu trúc dự án

```
ecommerce-recommendation/
├── static/                      # Thư mục chứa tài nguyên tĩnh
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/                   # Thư mục chứa các template HTML
│   ├── base.html
│   ├── index.html
│   └── error.html
│
├── models/                      # Thư mục chứa các mô hình ML
│   ├── __init__.py
│   ├── deep_matrix_factorization.py
│   └── hybrid_recommender.py
│
├── data/                        # Thư mục chứa dữ liệu
│   ├── amazon.csv               # Dữ liệu sản phẩm gốc
│   ├── clean_amazon.csv         # Dữ liệu đã được làm sạch (cho popularity model)
│   ├── amazon_keywords.csv      # Dữ liệu từ khóa cho cá nhân hóa
│   ├── dmf_model.pkl            # File mô hình DMF đã được huấn luyện
│   └── popularity_model.pkl     # File mô hình Popularity đã được huấn luyện
│
├── app.py                       # File chính của ứng dụng Flask
├── init_db.py                   # Script khởi tạo cơ sở dữ liệu
├── error_handlers.py            # Xử lý lỗi cho ứng dụng
├── train_dmf_model.py           # Script huấn luyện mô hình DMF
├── train_popularity_model.py    # Script tạo mô hình Popularity
├── train_all_models.py          # Script huấn luyện tất cả các mô hình
├── wsgi.py                      # Entry point cho WSGI servers
├── ecommerce.db                 # Database SQLite
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Các tính năng chính

- **Gợi ý sản phẩm hybrid**: Kết hợp DMF và Popularity model
- **Tìm kiếm sản phẩm**: Dựa trên từ khóa để tìm sản phẩm liên quan
- **Lọc theo danh mục**: Xem và lọc sản phẩm theo danh mục
- **Hiển thị nguồn gợi ý**: Hiển thị rõ sản phẩm được gợi ý từ mô hình nào (dmf/popularity)
- **Trang chi tiết sản phẩm**: Hiển thị đầy đủ thông tin sản phẩm

## Cài đặt và Chạy

### Yêu cầu

- Python 3.8+ 
- PyTorch
- Flask
- Pandas, NumPy, scikit-learn
- SQLite3

### Cài đặt

1. Clone repository:
```bash
git clone https://github.com/DuyTran04/N2_HeThongGoiYSanPham2.git
cd N2_HeThongGoiYSanPham2
```

2. Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

3. Huấn luyện các mô hình:
```bash
python train_all_models.py
```

4. Khởi động ứng dụng:
```bash
python app.py
```

5. Truy cập ứng dụng từ trình duyệt tại địa chỉ:
```
http://localhost:5000
```

### Docker

Nếu bạn muốn chạy ứng dụng bằng Docker:

```bash
docker-compose up --build
```

## Cách sử dụng

1. **Lấy gợi ý dựa trên sản phẩm**:
   - Chọn một danh mục từ dropdown đầu tiên
   - Chọn một sản phẩm cụ thể từ dropdown thứ hai
   - Nhấn nút "Get Recommendations" để xem các gợi ý sản phẩm

2. **Tìm kiếm sản phẩm**:
   - Nhập từ khóa tìm kiếm vào ô tìm kiếm
   - Nhấn nút "Search" để xem các gợi ý liên quan đến từ khóa

3. **Xem chi tiết sản phẩm**:
   - Nhấp vào "View Product" trên thẻ sản phẩm để xem chi tiết đầy đủ

## Huấn luyện mô hình

### Deep Matrix Factorization

Để huấn luyện lại mô hình DMF:

```bash
python train_dmf_model.py
```

### Popularity Model

Để tạo lại mô hình Popularity:

```bash
python train_popularity_model.py
```

### Huấn luyện tất cả các mô hình

```bash
python train_all_models.py
```

## Nguồn dữ liệu

- **amazon.csv**: Dữ liệu gốc từ web dataset
- **clean_amazon.csv**: Dữ liệu đã được làm sạch cho Popularity model
- **amazon_keywords.csv**: Dữ liệu từ khóa cho cá nhân hóa

## Tiến độ
- Hiện tại được kết nối với flask framework với 2 model nhưng chưa được mở rộng về phần user , và chưa hiểu các thuật toán.
