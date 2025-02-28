FROM python:3.9-slim

WORKDIR /app

# Cài đặt các dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code dự án
COPY . .

# Đảm bảo thư mục data tồn tại
RUN mkdir -p data

# Khởi tạo database (nếu cần)
RUN python init_db.py

# Mở port
EXPOSE 5000

# Chạy ứng dụng với Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]