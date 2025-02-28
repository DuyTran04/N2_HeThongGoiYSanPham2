from flask import render_template, jsonify, request

def register_error_handlers(app):
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', error_code=404, error_message="Page not found"), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('error.html', error_code=500, error_message="Internal server error"), 500
        
    @app.errorhandler(Exception)
    def handle_exception(e):
        # Trả về lỗi dạng JSON nếu là AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(error=str(e)), 500
        # Ngược lại hiển thị trang lỗi
        return render_template('error.html', error_code=500, error_message=str(e)), 500