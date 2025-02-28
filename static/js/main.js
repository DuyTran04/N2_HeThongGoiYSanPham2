$(document).ready(function() {
    // Xử lý thay đổi danh mục
    $('#category-select').change(function() {
        const category = $(this).val();
        if (category) {
            // Reset các phần khác
            $('#product-select').html('<option value="">Select a product</option>').prop('disabled', true);
            $('#product-details-container').hide();
            $('#get-recommendations-btn').prop('disabled', true);
            $('#recommendations-container').hide();
            
            // Hiển thị loading indicator
            $('#loading-indicator').show();
            
            // Lấy danh sách sản phẩm theo danh mục
            $.ajax({
                url: '/filter_products',
                method: 'POST',
                data: { category: category },
                success: function(response) {
                    const productSelect = $('#product-select');
                    productSelect.html('<option value="">Select a product</option>');
                    
                    response.products.forEach(function(product) {
                        productSelect.append(`<option value="${product}">${product}</option>`);
                    });
                    
                    productSelect.prop('disabled', false);
                    $('#loading-indicator').hide();
                },
                error: function() {
                    alert('Error loading products. Please try again.');
                    $('#loading-indicator').hide();
                }
            });
        }
    });
    
    // Xử lý thay đổi sản phẩm
    $('#product-select').change(function() {
        const product = $(this).val();
        if (product) {
            // Hiển thị loading indicator
            $('#loading-indicator').show();
            
            // Lấy thông tin sản phẩm
            $.ajax({
                url: '/product_details',
                method: 'POST',
                data: { product: product },
                success: function(response) {
                    if (response.error) {
                        alert(response.error);
                        $('#loading-indicator').hide();
                        return;
                    }
                    
                    $('#product-id').text(response.product_id);
                    $('#product-name').text(response.product_name);
                    $('#product-category').text(response.category);
                    $('#product-rating').text(`${response.rating} (${response.rating_count} ratings)`);
                    $('#product-image').attr('src', response.img_link);
                    $('#product-link-href').attr('href', response.product_link);
                    
                    // Lưu product_id để sử dụng khi lấy khuyến nghị
                    $('#get-recommendations-btn').data('product-id', response.product_id).prop('disabled', false);
                    
                    $('#product-details-container').show();
                    $('#recommendations-container').hide();
                    $('#loading-indicator').hide();
                },
                error: function() {
                    alert('Error loading product details. Please try again.');
                    $('#loading-indicator').hide();
                }
            });
        } else {
            // Ẩn phần chi tiết sản phẩm nếu không có sản phẩm được chọn
            $('#product-details-container').hide();
            $('#get-recommendations-btn').prop('disabled', true);
        }
    });
    
    // Xử lý nút lấy khuyến nghị
    $('#get-recommendations-btn').click(function() {
        const productId = $(this).data('product-id');
        
        // Hiển thị loading indicator
        $('#loading-indicator').show();
        
        $.ajax({
            url: '/get_recommendations',
            method: 'POST',
            data: { product_id: productId },
            success: function(response) {
                if (response.error) {
                    alert(response.error);
                    $('#loading-indicator').hide();
                    return;
                }
                
                displayRecommendations(response.recommendations);
            },
            error: function() {
                alert('Error getting recommendations. Please try again.');
                $('#loading-indicator').hide();
            }
        });
    });

    // Xử lý tìm kiếm
    $('#search-button').click(function() {
        const searchKeywords = $('#search-input').val().trim();
        
        if (searchKeywords) {
            // Hiển thị loading indicator
            $('#loading-indicator').show();
            
            // Gửi request tìm kiếm
            $.ajax({
                url: '/search',
                method: 'POST',
                data: { search_keywords: searchKeywords },
                success: function(response) {
                    if (response.error) {
                        alert(response.error);
                        $('#loading-indicator').hide();
                        return;
                    }
                    
                    displayRecommendations(response.recommendations);
                },
                error: function() {
                    alert('Error searching products. Please try again.');
                    $('#loading-indicator').hide();
                }
            });
        }
    });

    // Xử lý khi nhấn Enter trong ô tìm kiếm
    $('#search-input').keypress(function(e) {
        if (e.which === 13) { // Phím Enter
            $('#search-button').click();
            return false; // Ngăn form submit mặc định
        }
    });

    // Hàm hiển thị recommendations
    function displayRecommendations(recommendations) {
        const recommendationsList = $('#recommendations-list');
        recommendationsList.empty();
        
        if (!recommendations || recommendations.length === 0) {
            recommendationsList.html('<div class="col-12"><p class="alert alert-warning">No recommendations found.</p></div>');
            $('#recommendations-container').show();
            $('#loading-indicator').hide();
            return;
        }
        
        recommendations.forEach(function(rec, index) {
            const sourceLabel = rec.source ? `<span class="badge bg-info">${rec.source}</span>` : '';
            
            const card = `
                <div class="col-md-6 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-primary text-white">
                            <h5 class="card-title mb-0">Recommendation ${index + 1} ${sourceLabel}</h5>
                        </div>
                        <div class="text-center pt-3">
                            <img src="${rec.img_link}" class="card-img-top recommendation-img" alt="Product Image">
                        </div>
                        <div class="card-body">
                            <h6 class="card-subtitle mb-2 text-muted">${rec.product_id}</h6>
                            <h5 class="card-title">${rec.product_name}</h5>
                            <p class="card-text"><strong>Category:</strong> ${rec.category}</p>
                            <p class="card-text"><strong>Rating:</strong> ${rec.rating} (${rec.rating_count} ratings)</p>
                            <p class="card-text"><strong>Similarity Score:</strong> ${(rec.similarity * 100).toFixed(2)}%</p>
                        </div>
                        <div class="card-footer">
                            <a href="${rec.product_link}" target="_blank" class="btn btn-primary w-100">View Product</a>
                        </div>
                    </div>
                </div>
            `;
            recommendationsList.append(card);
        });
        
        $('#recommendations-container').show();
        $('#loading-indicator').hide();
        
        // Scroll to recommendations
        $('html, body').animate({
            scrollTop: $('#recommendations-container').offset().top - 70
        }, 500);
    }

    // Xử lý nút clear
    $('#clear-button').click(function() {
        $('#search-input').val('');
        $('#category-select').val('').trigger('change');
        $('#product-select').html('<option value="">Select a product</option>').prop('disabled', true);
        $('#product-details-container').hide();
        $('#recommendations-container').hide();
    });

    // Kiểm tra trạng thái hệ thống khi trang tải xong
    $.ajax({
        url: '/health',
        method: 'GET',
        success: function(response) {
            console.log('System status:', response);
            if (response.status !== 'ok') {
                alert('Warning: System may not be fully operational. Some features might be limited.');
            }
        },
        error: function() {
            console.error('Error checking system health');
        }
    });
});