import cv2
import numpy as np
import pytesseract
from imutils.perspective import four_point_transform
from datetime import datetime
import re
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import os

class CCCDVerifier:
    def __init__(self):
        # Khởi tạo mô hình phát hiện giả mạo
        self.fraud_detection_model = self.build_fraud_detection_model()
        
        # Load mô hình đã huấn luyện nếu có
        if os.path.exists('cccd_fraud_model.h5'):
            self.fraud_detection_model.load_weights('cccd_fraud_model.h5')
        
        # Thiết lập Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Thay đổi đường dẫn phù hợp
        
        # Biểu thức chính quy cho các trường thông tin
        self.id_pattern = r'\b\d{12}\b'  # Mẫu cho số CCCD 12 số
        self.name_pattern = r'Họ và tên:?\s*([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\s]+)'
        self.dob_pattern = r'Ngày sinh:?\s*(\d{2}/\d{2}/\d{4})'
        self.gender_pattern = r'Giới tính:?\s*(Nam|Nữ)'
        self.nationality_pattern = r'Quốc tịch:?\s*(Việt Nam)'
        self.address_pattern = r'Nơi thường trú:?\s*(.+)'
        self.expiry_pattern = r'Có giá trị đến:?\s*(\d{2}/\d{2}/\d{4})'
        
    def build_fraud_detection_model(self):
        """Xây dựng mô hình phát hiện giả mạo dựa trên Transfer Learning"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Đóng băng các lớp của mô hình cơ sở
        for layer in base_model.layers:
            layer.trainable = False
        
        # Thêm các lớp mới
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def preprocess_image(self, image):
        """Tiền xử lý ảnh CCCD"""
        # Chuyển ảnh sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Làm mờ ảnh để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Áp dụng ngưỡng thích ứng để nhị phân hóa ảnh
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tìm contours trong ảnh
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sắp xếp contours theo diện tích, từ lớn đến nhỏ
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Biến để lưu đường viền của CCCD
        card_contour = None
        
        # Duyệt qua các contours
        for contour in contours:
            # Tính gần đúng đường viền
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Nếu đường viền có 4 điểm, đó có thể là CCCD
            if len(approx) == 4:
                card_contour = approx
                break
        
        # Nếu không tìm thấy đường viền hợp lệ
        if card_contour is None:
            return None
        
        # Áp dụng phép biến đổi perspection để lấy góc nhìn vuông góc
        warped = four_point_transform(image, card_contour.reshape(4, 2))
        
        return warped
    
    def detect_fraud(self, image):
        """Phát hiện CCCD giả mạo sử dụng mô hình deep learning"""
        # Tiền xử lý ảnh cho mô hình
        img = cv2.resize(image, (224, 224))
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)
        
        # Dự đoán
        score = self.fraud_detection_model.predict(img)[0][0]
        
        # Điểm số > 0.5 là giả mạo
        is_fraudulent = score > 0.5
        
        return is_fraudulent, score
    
    def extract_information(self, image):
        """Trích xuất thông tin từ ảnh CCCD đã được xử lý"""
        # Chuyển sang ảnh xám cho OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Cải thiện độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Thực hiện OCR với ngôn ngữ Tiếng Việt
        text = pytesseract.image_to_string(enhanced, lang='vie')
        
        # Trích xuất thông tin bằng regex
        cccd_num = re.search(self.id_pattern, text)
        name = re.search(self.name_pattern, text)
        dob = re.search(self.dob_pattern, text)
        gender = re.search(self.gender_pattern, text)
        nationality = re.search(self.nationality_pattern, text)
        address = re.search(self.address_pattern, text)
        expiry = re.search(self.expiry_pattern, text)
        
        # Tạo từ điển kết quả
        result = {
            'cccd_number': cccd_num.group(0) if cccd_num else None,
            'name': name.group(1) if name else None,
            'date_of_birth': dob.group(1) if dob else None,
            'gender': gender.group(1) if gender else None,
            'nationality': nationality.group(1) if nationality else None,
            'address': address.group(1) if address else None,
            'expiry_date': expiry.group(1) if expiry else None,
        }
        
        return result
    
    def validate_information(self, info):
        """Kiểm tra tính hợp lệ của thông tin CCCD"""
        validation_results = {}
        
        # Kiểm tra số CCCD
        if info['cccd_number']:
            if len(info['cccd_number']) == 12 and info['cccd_number'].isdigit():
                validation_results['cccd_number'] = True
            else:
                validation_results['cccd_number'] = False
        else:
            validation_results['cccd_number'] = False
        
        # Kiểm tra ngày sinh
        if info['date_of_birth']:
            try:
                dob = datetime.strptime(info['date_of_birth'], '%d/%m/%Y')
                current_year = datetime.now().year
                age = current_year - dob.year
                if 16 <= age <= 120:  # Giả sử độ tuổi hợp lệ cho CCCD
                    validation_results['date_of_birth'] = True
                else:
                    validation_results['date_of_birth'] = False
            except:
                validation_results['date_of_birth'] = False
        else:
            validation_results['date_of_birth'] = False
        
        # Kiểm tra ngày hết hạn
        if info['expiry_date']:
            try:
                expiry = datetime.strptime(info['expiry_date'], '%d/%m/%Y')
                if expiry > datetime.now():
                    validation_results['expiry_date'] = True
                else:
                    validation_results['expiry_date'] = False
            except:
                validation_results['expiry_date'] = False
        else:
            validation_results['expiry_date'] = False
        
        # Các trường còn lại chỉ kiểm tra có tồn tại hay không
        for field in ['name', 'gender', 'nationality', 'address']:
            validation_results[field] = info[field] is not None
        
        # Tính toán kết quả tổng thể
        overall_valid = all(validation_results.values())
        
        return overall_valid, validation_results
    
    def verify_cccd(self, image_path):
        """Quy trình xác thực CCCD hoàn chỉnh"""
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'message': 'Không thể đọc ảnh',
                'data': None
            }
        
        # Tiền xử lý ảnh
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return {
                'success': False,
                'message': 'Không thể tìm thấy CCCD trong ảnh',
                'data': None
            }
        
        # Phát hiện giả mạo
        is_fraudulent, fraud_score = self.detect_fraud(processed_image)
        if is_fraudulent:
            return {
                'success': False,
                'message': f'Nghi ngờ CCCD giả mạo (Điểm số: {fraud_score:.2f})',
                'data': None
            }
        
        # Trích xuất thông tin
        information = self.extract_information(processed_image)
        
        # Xác thực thông tin
        is_valid, validation_details = self.validate_information(information)
        
        # Tạo kết quả
        result = {
            'success': is_valid,
            'message': 'CCCD hợp lệ' if is_valid else 'CCCD không hợp lệ hoặc thiếu thông tin',
            'data': {
                'information': information,
                'validation_details': validation_details,
                'fraud_score': fraud_score
            }
        }
        
        return result
    
    def train_fraud_detection_model(self, real_images_dir, fake_images_dir, epochs=10, batch_size=32):
        """Huấn luyện mô hình phát hiện giả mạo với dữ liệu mới"""
        # Chuẩn bị dữ liệu
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2
        )
        
        # Generator cho tập huấn luyện
        train_generator = datagen.flow_from_directory(
            directory=os.path.dirname(real_images_dir),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        
        # Generator cho tập kiểm tra
        validation_generator = datagen.flow_from_directory(
            directory=os.path.dirname(real_images_dir),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        # Huấn luyện mô hình
        history = self.fraud_detection_model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs
        )
        
        # Lưu mô hình
        self.fraud_detection_model.save_weights('cccd_fraud_model.h5')
        
        return history

# Sử dụng
if __name__ == "__main__":
    verifier = CCCDVerifier()
    
    # Xác thực CCCD từ ảnh
    result = verifier.verify_cccd('path_to_cccd_image.jpg')
    
    if result['success']:
        print("CCCD hợp lệ!")
        print("Thông tin:")
        for key, value in result['data']['information'].items():
            print(f"{key}: {value}")
    else:
        print(f"Lỗi: {result['message']}")