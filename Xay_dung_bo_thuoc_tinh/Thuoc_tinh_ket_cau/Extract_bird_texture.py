import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_image(image_path):
    """Tải hình ảnh từ đường dẫn"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_haralick_features(image):
    """Trích xuất đặc trưng Haralick từ GLCM
    
    Trả về 13 đặc trưng Haralick cho mỗi khoảng cách và hướng, 
    sau đó lấy giá trị trung bình cho tất cả các hướng.
    """
    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Giảm số mức xám để tăng tốc độ tính toán và giảm nhiễu
    gray = (gray // 32).astype(np.uint8)
    
    # Tính toán GLCM với các khoảng cách [1, 2, 3] và các hướng [0, 45, 90, 135]
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 độ
    
    # Tính GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                      symmetric=True, normed=True)
    
    # Trích xuất các thuộc tính từ GLCM
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    
    # Tính toán các thuộc tính cho mỗi khoảng cách và lấy trung bình qua các hướng
    for prop in props:
        for d in range(len(distances)):
            feat = graycoprops(glcm, prop)[d].mean()  # Trung bình qua các hướng
            features.append(feat)
    
    # Tạo dict với tên đặc trưng
    feature_names = [f'haralick_{prop}_dist{d+1}' for d in range(len(distances)) for prop in props]
    return dict(zip(feature_names, features))

def extract_gabor_features(image, num_scales=4, num_orientations=8):
    """Trích xuất đặc trưng từ bộ lọc Gabor
    
    Sử dụng bộ lọc Gabor với số tỷ lệ và số hướng định trước
    """
    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Chuẩn hóa kích thước ảnh (tùy chọn)
    gray = cv2.resize(gray, (256, 256))
    
    # Chuẩn hóa giá trị điểm ảnh
    gray = gray / 255.0
    
    # Các tham số cho bộ lọc Gabor
    scales = np.logspace(-1, 0.8, num_scales)
    orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
    
    features = {}
    
    for i, sigma in enumerate(scales):
        for j, theta in enumerate(orientations):
            # Tạo kernel Gabor
            frequency = 1.0 / (sigma * 4)  # Liên hệ giữa tần số và tỷ lệ
            
            # Tạo bộ lọc Gabor thực
            gabor_real = cv2.getGaborKernel(
                (31, 31), sigma, theta, frequency, 0.5, 0, ktype=cv2.CV_32F
            )
            
            # Áp dụng bộ lọc
            filtered_img = cv2.filter2D(gray, cv2.CV_64F, gabor_real)
            
            # Tính toán các thống kê từ ảnh đã lọc
            mean = np.mean(filtered_img)
            var = np.var(filtered_img)
            energy = np.sum(filtered_img**2)
            
            # Lưu các đặc trưng
            features[f'gabor_mean_scale{i+1}_orient{j+1}'] = mean
            features[f'gabor_var_scale{i+1}_orient{j+1}'] = var
            features[f'gabor_energy_scale{i+1}_orient{j+1}'] = energy
    
    return features

def extract_all_texture_features(image_path):
    """Trích xuất tất cả các đặc trưng kết cấu từ một ảnh"""
    try:
        # Tải ảnh
        img = load_image(image_path)
        
        # Trích xuất đặc trưng Haralick
        haralick_features = extract_haralick_features(img)
        
        # Trích xuất đặc trưng Gabor
        gabor_features = extract_gabor_features(img)
        
        # Kết hợp các đặc trưng
        features = {**haralick_features, **gabor_features}
        
        # Thêm tên file
        features['image_file'] = os.path.basename(image_path)
        
        return features
    except Exception as e:
        print(f"Lỗi khi xử lý {image_path}: {e}")
        return {'image_file': os.path.basename(image_path)}

def main():
    # Xác định đường dẫn tương đối đến thư mục Data dựa trên vị trí file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))  # Lên 2 cấp thư mục
    image_dir = os.path.join(project_dir, 'Data')
    output_file = 'bird_texture_features.csv'
    
    print(f"Đường dẫn đến thư mục ảnh: {image_dir}")
    
    # Lấy tất cả các file ảnh
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục {image_dir}")
    
    # Trích xuất đặc trưng cho tất cả các ảnh
    all_features = []
    for img_path in tqdm(image_files, desc="Đang trích xuất đặc trưng kết cấu"):
        features = extract_all_texture_features(img_path)
        all_features.append(features)
    
    # Tạo DataFrame
    df = pd.DataFrame(all_features)
    
    # Lưu kết quả
    df.to_csv(output_file, index=False)
    print(f"Đã lưu đặc trưng kết cấu vào {output_file}")
    print(f"Số lượng đặc trưng đã trích xuất: {df.shape[1] - 1}")  # Trừ cột 'image_file'

if __name__ == "__main__":
    main()