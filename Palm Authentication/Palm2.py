import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

class PalmAuthSystem:
    def __init__(self, data_dir="palm_data"):
        self.data_dir = data_dir
        self.model = None
        self.mean_img = None
        self.std_img = None
        
        # Tạo thư mục dữ liệu nếu chưa tồn tại
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def capture_palm(self, user_id, num_samples=5):
        """Thu thập ảnh lòng bàn tay cho người dùng"""
        user_dir = os.path.join(self.data_dir, user_id)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera!")
            return False
        
        samples_collected = 0
        while samples_collected < num_samples:
            print(f"Đặt lòng bàn tay của bạn vào khung hình. Chuẩn bị chụp mẫu {samples_collected+1}/{num_samples}")
            
            # Đếm ngược
            for i in range(3, 0, -1):
                print(f"{i}...")
                ret, frame = cap.read()
                if not ret:
                    print("Không thể chụp ảnh!")
                    continue
                
                # Hiển thị khung hình với hướng dẫn
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Chuẩn bị: {i}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Camera', display_frame)
                
                # Thoát nếu nhấn 'q'
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            
            # Chụp ảnh
            ret, frame = cap.read()
            if ret:
                # Chuyển sang ảnh xám
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Lưu ảnh
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(user_dir, f"sample_{samples_collected}_{timestamp}.jpg")
                cv2.imwrite(filename, gray)
                
                # Hiển thị ảnh đã chụp
                cv2.putText(gray, "Đã chụp!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Ảnh đã chụp', gray)
                cv2.waitKey(1000)
                cv2.destroyWindow('Ảnh đã chụp')
                
                samples_collected += 1
                print(f"Đã lưu mẫu {samples_collected}/{num_samples}")
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def preprocess_image(self, image_path, target_size=(128, 128)):
        """Tiền xử lý ảnh"""
        # Đọc ảnh
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Thay đổi kích thước
        img = cv2.resize(img, target_size)
        
        # Cân bằng histogram để cải thiện độ tương phản
        img = cv2.equalizeHist(img)
        
        # Lọc nhiễu
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        return img
    
    def extract_features(self, img):
        """Trích xuất đặc trưng từ ảnh"""
        # Phát hiện cạnh
        edges = cv2.Canny(img, 50, 150)
        
        # HOG (Histogram of Oriented Gradients)
        winSize = (128, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog_features = hog.compute(img)
        
        # Kết hợp đặc trưng
        img_flat = img.flatten() / 255.0  # Chuẩn hóa pixel về [0,1]
        edges_flat = edges.flatten() / 255.0
        
        # Kết hợp các đặc trưng
        features = np.concatenate([img_flat, edges_flat, hog_features.flatten()])
        
        return features
    
    def load_data(self):
        """Tải và chuẩn bị dữ liệu cho huấn luyện"""
        X = []
        y = []
        
        # Duyệt qua tất cả thư mục người dùng
        for user_id in os.listdir(self.data_dir):
            user_dir = os.path.join(self.data_dir, user_id)
            if os.path.isdir(user_dir):
                for img_file in os.listdir(user_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(user_dir, img_file)
                        
                        # Tiền xử lý ảnh
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            # Trích xuất đặc trưng
                            features = self.extract_features(img)
                            
                            # Thêm vào tập dữ liệu
                            X.append(features)
                            y.append(user_id)
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Huấn luyện mô hình nhận dạng lòng bàn tay"""
        # Tải dữ liệu
        X, y = self.load_data()
        
        if len(X) == 0:
            print("Không có dữ liệu để huấn luyện!")
            return False
        
        # Chuẩn hóa dữ liệu
        self.mean_feat = np.mean(X, axis=0)
        self.std_feat = np.std(X, axis=0)
        X_norm = (X - self.mean_feat) / (self.std_feat + 1e-10)
        
        # Chia tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42)
        
        # Khởi tạo và huấn luyện mô hình KNN
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X_train, y_train)
        
        # Đánh giá mô hình
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2%}")
        
        # Lưu mô hình
        with open('palm_auth_model.pkl', 'wb') as f:
            pickle.dump((self.model, self.mean_feat, self.std_feat), f)
        
        return True
    
    def load_model(self):
        """Tải mô hình đã huấn luyện"""
        try:
            with open('palm_auth_model.pkl', 'rb') as f:
                self.model, self.mean_feat, self.std_feat = pickle.load(f)
            return True
        except:
            print("Không thể tải mô hình!")
            return False
    
    def authenticate(self):
        """Xác thực người dùng bằng lòng bàn tay"""
        if self.model is None:
            if not self.load_model():
                print("Chưa có mô hình nào được huấn luyện!")
                return None
        
        # Chụp ảnh lòng bàn tay
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera!")
            return None
        
        print("Đặt lòng bàn tay của bạn vào khung hình để xác thực...")
        
        # Đếm ngược
        for i in range(3, 0, -1):
            print(f"{i}...")
            ret, frame = cap.read()
            if not ret:
                continue
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Chuẩn bị: {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Camera', display_frame)
            cv2.waitKey(1000)
        
        # Chụp ảnh
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        
        if not ret:
            print("Không thể chụp ảnh!")
            return None
        
        # Tiền xử lý ảnh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (128, 128))
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Trích xuất đặc trưng
        features = self.extract_features(img)
        
        # Chuẩn hóa đặc trưng
        features_norm = (features - self.mean_feat) / (self.std_feat + 1e-10)
        
        # Dự đoán và tính điểm tin cậy
        distances, indices = self.model.kneighbors([features_norm], n_neighbors=3)
        neighbors = self.model.classes_[indices][0]
        
        # Tính số lần xuất hiện của mỗi lớp trong k láng giềng gần nhất
        unique_classes, counts = np.unique(neighbors, return_counts=True)
        
        # Lấy lớp có số lần xuất hiện nhiều nhất
        max_count_idx = np.argmax(counts)
        predicted_user = unique_classes[max_count_idx]
        confidence = counts[max_count_idx] / sum(counts)
        
        # Kiểm tra ngưỡng tin cậy
        if confidence >= 0.67:  # Ít nhất 2/3 láng giềng cùng lớp
            return predicted_user, confidence
        else:
            return None, confidence

def main():
    auth_system = PalmAuthSystem()
    
    while True:
        print("\n===== HỆ THỐNG XÁC THỰC LÒNG BÀN TAY =====")
        print("1. Đăng ký người dùng mới")
        print("2. Huấn luyện mô hình")
        print("3. Xác thực người dùng")
        print("4. Thoát")
        
        choice = input("Nhập lựa chọn: ")
        
        if choice == '1':
            user_id = input("Nhập ID người dùng: ")
            num_samples = int(input("Số lượng mẫu (mặc định: 5): ") or "5")
            auth_system.capture_palm(user_id, num_samples)
            
        elif choice == '2':
            print("Bắt đầu huấn luyện mô hình...")
            if auth_system.train_model():
                print("Huấn luyện mô hình thành công!")
            
        elif choice == '3':
            print("Đang xác thực...")
            result = auth_system.authenticate()
            
            if result is not None:
                user_id, confidence = result
                print(f"Xác thực thành công! Người dùng: {user_id} (Độ tin cậy: {confidence:.2%})")
            else:
                print("Xác thực thất bại hoặc người dùng không được nhận dạng!")
            
        elif choice == '4':
            break
            
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()

# Cải tiến có thể thêm vào

# Nâng cao xử lý ảnh:

# Phát hiện vùng lòng bàn tay tự động
# Chuẩn hóa góc quay và vị trí
# Thêm các bộ lọc nâng cao


# Cải thiện trích xuất đặc trưng:

# Sử dụng các đặc trưng riêng của lòng bàn tay (đường chính, đường phụ)
# Áp dụng các thuật toán trích xuất đặc điểm như SIFT, SURF
# Sử dụng mạng neural CNN để trích xuất đặc trưng


# Mô hình nâng cao:

# Thử nghiệm với SVM, Random Forest hoặc mạng neural
# Áp dụng Transfer Learning với các mô hình đã huấn luyện sẵn
# Cập nhật mô hình liên tục


# Bảo mật:

# Thêm phát hiện "liveness" để chống giả mạo
# Mã hóa dữ liệu sinh trắc học
# Thêm xác thực nhiều yếu tố