import os
from datetime import datetime
from mltu.configs import BaseModelConfigs
class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/03_handwriting_recognition", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))  ##Đặt đường dẫn
        self.vocab = ""  ##Biến lưu trữ từ vựng ký tự
        self.height = 32 ##Xác định chiều cao hình ảnh viết tay đầu vào để nhận dạng
        self.width  = 128 ##Xác định chiều rộng
        self.max_text_length =0 ##Độ dài dự kiến tối đa, sau có thể thay đổi
        self.batch_size = 16 ##Số lượng ảnh được xử lý cùng nhau trong quá trình đào tạo trong 1 lần lặp
        self.learning_rate = 0.0005 #Tham số kiểm soát kích thước bước mà trình tối ưu hóa thực hiện khi cập nhập trọng số của mô hình trong quá trình đào tạo
        self.train_epochs = 10 ##Số lượng tối đa  toàn bộ tập dữ liệu huấn luyện sẽ được chuyển qua mô hình trong quá trình huấn luyện
        self.train_workers = 20 ##Số lượng sẽ được sử dụng để tải và xử lý trước dữ liệu đào tạo song song
        