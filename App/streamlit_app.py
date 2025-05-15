import os
import cv2
import numpy as np
import sqlite3
import streamlit as st
from skimage import feature, color
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import cosine, euclidean
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import io
import joblib

class BirdFeatureExtractor:
    def __init__(self):
        pass
    
    def remove_background(self, img):
        # Chuyển sang xám và nhị phân hóa
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img  # Không tìm thấy contour, trả về ảnh gốc
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        # Áp dụng mask lên ảnh RGB
        img_fg = cv2.bitwise_and(img, img, mask=mask)
        return img_fg
    
    def extract_features(self, image_path):
        """Trích xuất tất cả các đặc trưng từ ảnh."""
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Chuyển đổi sang RGB để hiển thị và xử lý
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_fg = self.remove_background(img_rgb)
        # Tạo bản sao để xử lý
        img_for_processing = img_fg.copy()
        
        # Trích xuất các đặc trưng
        color_features = self.extract_color_features(img_for_processing)
        shape_features = self.extract_shape_features(img_for_processing)
        texture_features = self.extract_texture_features(img_for_processing)
        
        # Kết hợp tất cả các đặc trưng thành một vector đặc trưng
        all_features = np.concatenate([color_features, shape_features, texture_features])
        
        return all_features
    
    def extract_color_features(self, img):
        """Trích xuất đặc trưng màu sắc: Biểu đồ HSV và màu chủ đạo."""
        # Chuyển sang không gian màu HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 1. Biểu đồ màu HSV
        h_bins = 8
        s_bins = 8
        v_bins = 8
        
        h_hist = cv2.calcHist([hsv_img], [0], None, [h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv_img], [1], None, [s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [v_bins], [0, 256])
        
        # Chuẩn hóa
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # 2. Màu chủ đạo (Dominant Colors)
        # Chuyển đổi ảnh sang dạng một mảng các pixel
        pixels = hsv_img.reshape((-1, 3))
        
        # Lượng tử hóa màu để giảm số lượng màu
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5  # số lượng màu chủ đạo muốn trích xuất
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Đếm số lượng pixel cho mỗi trung tâm màu
        counts = Counter(labels.flatten())
        
        # Tính tỷ lệ của mỗi màu chủ đạo
        total_pixels = float(len(labels))
        dominant_color_features = np.zeros(k * 4)  # H, S, V và tỷ lệ cho mỗi màu
        
        for i in range(k):
            j = 0
            dominant_color_features[i*4 + j] = centers[i][0] / 180.0  # H normalized
            j += 1
            dominant_color_features[i*4 + j] = centers[i][1] / 255.0  # S normalized 
            j += 1
            dominant_color_features[i*4 + j] = centers[i][2] / 255.0  # V normalized
            j += 1
            dominant_color_features[i*4 + j] = counts.get(i, 0) / total_pixels  # Tỷ lệ
        
        # Kết hợp các đặc trưng màu
        color_features = np.concatenate([h_hist, s_hist, v_hist, dominant_color_features])
        
        return color_features
    
    def extract_shape_features(self, img):
        """Trích xuất đặc trưng hình dạng: Tỷ lệ hình học, Mô-men Hu và Fourier Descriptors."""
        # Chuyển ảnh sang ảnh xám để xử lý hình dạng
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Nhị phân hóa ảnh
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tìm các contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Nếu không tìm thấy contour, trả về vector đặc trưng với các giá trị 0
            return np.zeros(30)  # Điều chỉnh kích thước theo nhu cầu
        
        # Lấy contour lớn nhất (giả sử đó là chim)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 1. Tỷ lệ hình học (Geometric Ratios)
        # Tính bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h
        
        # Tính diện tích và chu vi
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Tính độ tròn (circularity)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Tính độ đặc (solidity)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 2. Mô-men Hu (Hu Moments)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Áp dụng log transform để giảm phạm vi giá trị
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 3. Fourier Descriptors
        # Chuyển contour sang một mảng các điểm
        contour_points = largest_contour.reshape(-1, 2)
        
        # Nếu có quá ít điểm, thêm các điểm nội suy
        if len(contour_points) < 10:
            # Tạo một contour mới với các điểm nội suy
            contour_points = np.vstack([contour_points, contour_points[:1]])  # Đóng contour
            
        # Lấy tọa độ x và y
        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]
        
        # Tạo các tọa độ phức
        complex_coords = x_coords + 1j * y_coords
        
        # Tính FFT
        fourier_result = np.fft.fft(complex_coords)
        
        # Lấy biên độ của các hệ số Fourier và chuẩn hóa
        fourier_descriptors = np.abs(fourier_result[1:11]) / np.abs(fourier_result[0]) if np.abs(fourier_result[0]) > 0 else np.zeros(10)
        
        # Kết hợp các đặc trưng hình dạng
        shape_features = np.concatenate([
            [aspect_ratio, circularity, solidity],
            hu_moments,
            fourier_descriptors
        ])
        
        return shape_features
    
    def extract_texture_features(self, img):
        """Trích xuất đặc trưng kết cấu: Haralick (GLCM) và Gabor."""
        # Chuyển ảnh sang ảnh xám để xử lý kết cấu
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Đặc trưng Haralick (từ GLCM)
        # Tạo GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 degrees
        glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        # Tính các đặc trưng từ GLCM
        haralick_features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in properties:
            feature = graycoprops(glcm, prop).flatten()
            haralick_features.append(feature)
        
        haralick_features = np.concatenate(haralick_features)
        
        # 2. Bộ lọc Gabor (Gabor Filters)
        # Định nghĩa các tham số của bộ lọc Gabor
        num_theta = 4  # số hướng
        theta = np.linspace(0, np.pi, num_theta)
        num_sigma = 2  # số tỷ lệ
        sigma = [1, 3]
        num_frequency = 2  # số tần số
        frequency = [0.1, 0.4]
        
        gabor_features = []
        
        for t in theta:
            for s in sigma:
                for f in frequency:
                    # Tạo kernel Gabor
                    kernel = cv2.getGaborKernel((21, 21), s, t, f, 0.5, 0, ktype=cv2.CV_32F)
                    
                    # Áp dụng bộ lọc
                    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    
                    # Tính mean và std của ảnh sau khi lọc
                    mean = np.mean(filtered_img)
                    std = np.std(filtered_img)
                    
                    gabor_features.extend([mean, std])
        
        # Kết hợp các đặc trưng kết cấu
        texture_features = np.concatenate([haralick_features, gabor_features])
        
        return texture_features

class BirdImageDatabase:
    def __init__(self, db_path='bird_features.db'):
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """Khởi tạo cơ sở dữ liệu SQLite."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Tạo bảng lưu trữ đường dẫn ảnh và đặc trưng
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bird_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            features BLOB NOT NULL
        )
        ''')
        
        self.conn.commit()
    
    def store_features(self, image_path, features):
        """Lưu đặc trưng của ảnh vào cơ sở dữ liệu."""
        cursor = self.conn.cursor()
        
        # Kiểm tra xem ảnh đã tồn tại trong cơ sở dữ liệu chưa
        cursor.execute("SELECT id FROM bird_images WHERE image_path = ?", (image_path,))
        result = cursor.fetchone()
        
        # Chuyển đổi features thành dạng binary để lưu trữ
        features_blob = features.tobytes()
        
        if result:
            # Cập nhật nếu ảnh đã tồn tại
            cursor.execute("UPDATE bird_images SET features = ? WHERE id = ?", (features_blob, result[0]))
        else:
            # Thêm mới nếu ảnh chưa tồn tại
            cursor.execute("INSERT INTO bird_images (image_path, features) VALUES (?, ?)", (image_path, features_blob))
        
        self.conn.commit()
    
    def load_all_features(self):
        """Tải tất cả các đặc trưng và đường dẫn ảnh từ cơ sở dữ liệu."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT image_path, features FROM bird_images")
        results = cursor.fetchall()
        
        image_paths = []
        features_list = []
        
        for image_path, features_blob in results:
            # Chuyển đổi blob thành numpy array
            features = np.frombuffer(features_blob, dtype=np.float64)
            
            image_paths.append(image_path)
            features_list.append(features)
        
        return image_paths, features_list
    
    def close(self):
        """Đóng kết nối cơ sở dữ liệu."""
        if self.conn:
            self.conn.close()
            self.conn = None

class BirdImageRetrieval:
    def __init__(self, db_path='bird_features.db'):
        self.feature_extractor = BirdFeatureExtractor()
        self.db = BirdImageDatabase(db_path)
        self.image_paths = None
        self.image_features = None
        self.load_database()
    
    def load_database(self):
        """Tải dữ liệu từ cơ sở dữ liệu."""
        self.image_paths, self.image_features = self.db.load_all_features()
        
        # Chuyển đổi danh sách đặc trưng thành mảng numpy cho xử lý nhanh hơn
        if self.image_features:
            # Kiểm tra xem tất cả các vector có cùng kích thước không
            feature_dims = [len(feat) for feat in self.image_features]
            
            if len(set(feature_dims)) > 1:
                print(f"Cảnh báo: Các vector đặc trưng có kích thước khác nhau: {set(feature_dims)}")
                
                # Lọc ra các cặp ảnh và đặc trưng có cùng kích thước phổ biến nhất
                most_common_dim = max(set(feature_dims), key=feature_dims.count)
                valid_indices = [i for i, dim in enumerate(feature_dims) if dim == most_common_dim]
                
                self.image_paths = [self.image_paths[i] for i in valid_indices]
                self.image_features = [self.image_features[i] for i in valid_indices]
                
                print(f"Đã lọc và giữ lại {len(valid_indices)} ảnh có kích thước vector {most_common_dim}")
            
            # Xác định kích thước của vector đặc trưng
            if self.image_features:
                feature_dim = len(self.image_features[0])
                self.image_features = np.array(self.image_features)
    
    def process_all_images(self, image_dir):
        """Xử lý tất cả các ảnh trong thư mục và lưu đặc trưng vào cơ sở dữ liệu."""
        # Liệt kê tất cả các tệp trong thư mục
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_images = len(image_files)
        processed_count = 0
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            try:
                # Trích xuất đặc trưng
                features = self.feature_extractor.extract_features(image_path)
                
                # Lưu vào cơ sở dữ liệu
                self.db.store_features(image_path, features)
                
                processed_count += 1
                print(f"Đã xử lý {processed_count}/{total_images}: {image_file}")
            
            except Exception as e:
                print(f"Lỗi khi xử lý {image_file}: {str(e)}")
        
        # Tải lại dữ liệu sau khi xử lý
        self.load_database()
        
        return processed_count
    
    def retrieve_similar_images(self, query_image_path, top_k=3, use_weight=True):
        """Tìm k ảnh tương tự nhất với ảnh truy vấn."""
        if self.image_features is None or len(self.image_features) == 0:
            return [], []
        
        # Trích xuất đặc trưng từ ảnh truy vấn
        query_features = self.feature_extractor.extract_features(query_image_path)
        
        # Đảm bảo query_features có cùng kích thước với image_features
        if len(query_features) != self.image_features.shape[1]:
            print(f"Kích thước của query_features ({len(query_features)}) không khớp với image_features ({self.image_features.shape[1]})")
            # Cắt bớt hoặc mở rộng vector nếu cần thiết
            if len(query_features) > self.image_features.shape[1]:
                query_features = query_features[:self.image_features.shape[1]]
            else:
                # Chỉ dùng cho debug, không nên xảy ra trong thực tế
                temp = np.zeros(self.image_features.shape[1])
                temp[:len(query_features)] = query_features
                query_features = temp
        
        # Tính trọng số và chuẩn hóa các đặc trưng
        if use_weight:
            # Xác định chỉ số cho từng loại đặc trưng
            feature_dim = len(query_features)
            
            # Điều chỉnh các chỉ số dựa trên kích thước thực tế
            color_end = min(int(feature_dim * 0.7), feature_dim)  # Khoảng 40% đầu là đặc trưng màu
            shape_end = min(int(feature_dim * 0.3), feature_dim)  # Khoảng 30% tiếp theo là đặc trưng hình dạng
            
            # Tạo vector trọng số
            weights = np.ones(feature_dim)
            
            # Đặt trọng số theo loại đặc trưng - điều chỉnh theo nhu cầu
            weights[:color_end] *= 80.0        # Tăng mạnh độ quan trọng của màu sắc
            weights[color_end:shape_end] *= 10.0# Tăng vừa phải độ quan trọng của hình dạng
            weights[shape_end:] *= 0.2        # Giảm độ quan trọng của kết cấu
            
            # Chuẩn hóa để đảm bảo công bằng giữa các đặc trưng
            # Đầu tiên chuẩn hóa vector đặc trưng
            query_features = query_features / (np.linalg.norm(query_features) + 1e-10)
            db_features_normalized = self.image_features / (np.linalg.norm(self.image_features, axis=1, keepdims=True) + 1e-10)
            
            # Sau đó áp dụng trọng số
            query_features = query_features * weights
            weighted_db_features = db_features_normalized * weights
        else:
            # Nếu không dùng trọng số, vẫn nên chuẩn hóa
            query_features = query_features / (np.linalg.norm(query_features) + 1e-10)
            weighted_db_features = self.image_features / (np.linalg.norm(self.image_features, axis=1, keepdims=True) + 1e-10)
        
        # Kết hợp các phương pháp đo khoảng cách
        distances = []
        for features in weighted_db_features:
            # Tính khoảng cách cosine (càng thấp càng tốt)
            cosine_dist = cosine(query_features, features)
            
            # Tính khoảng cách Euclidean đã chuẩn hóa (càng thấp càng tốt)
            euclidean_dist = euclidean(query_features, features) / np.sqrt(len(query_features))
            
            # Kết hợp các khoảng cách (trọng số có thể điều chỉnh)
            combined_dist = 0.6 * cosine_dist + 0.4 * euclidean_dist
            distances.append(combined_dist)
        
        # Chuyển đổi khoảng cách thành điểm số độ tương đồng
        similarity_scores = [1 / (1 + d) for d in distances]
        
        
        # Sắp xếp các chỉ số theo độ tương đồng giảm dần
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        # Lấy top_k ảnh tương tự nhất
        top_k = min(top_k, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        top_paths = [self.image_paths[i] for i in top_indices]
        top_scores = [similarity_scores[i] for i in top_indices]
        
        return top_paths, top_scores
    
    def close(self):
        """Đóng kết nối cơ sở dữ liệu."""
        self.db.close()


# Ứng dụng Streamlit
def main():
    st.set_page_config(page_title="Hệ thống tìm kiếm ảnh chim", layout="wide")
    
    st.title("Hệ thống tìm kiếm ảnh chim dựa trên nội dung")
    
    # Tab cho các chức năng khác nhau
    tab1, tab2 = st.tabs(["Tìm kiếm ảnh", "Quản lý cơ sở dữ liệu"])
    
    # Tab tìm kiếm ảnh
    with tab1:
        st.header("Tải lên ảnh để tìm kiếm")
        
        uploaded_file = st.file_uploader("Chọn ảnh chim", type=["jpg", "jpeg", "png"])
        
        use_weight = st.checkbox("Sử dụng trọng số cho các đặc trưng", value=True)
        
        if uploaded_file is not None:
            # Hiển thị ảnh đã tải lên
            st.image(uploaded_file, caption="Ảnh tải lên", width=300)
            
            # Lưu ảnh tạm thời để xử lý
            temp_image_path = "temp_query_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Nút tìm kiếm
            if st.button("Tìm kiếm ảnh tương tự"):
                with st.spinner("Đang tìm kiếm ảnh tương tự..."):
                    # Khởi tạo hệ thống tìm kiếm
                    retrieval_system = BirdImageRetrieval()
                    
                    # Lấy 3 ảnh tương tự nhất
                    try:
                        similar_image_paths, similarity_scores = retrieval_system.retrieve_similar_images(
                            temp_image_path, top_k=3, use_weight=use_weight
                        )
                        retrieval_system.close()
                        
                        if similar_image_paths:
                            st.success("Đã tìm thấy các ảnh tương tự!")
                            
                            # Hiển thị các ảnh tương tự
                            st.subheader("Kết quả tìm kiếm")
                            
                            # Tạo 3 cột để hiển thị 3 ảnh
                            cols = st.columns(3)
                            
                            for i, (img_path, score) in enumerate(zip(similar_image_paths, similarity_scores)):
                                with cols[i]:
                                    try:
                                        img = Image.open(img_path)
                                        img = img.convert("RGB")
                                        st.image(img, caption=f"#{i+1}: {os.path.basename(img_path)}", width=250)
                                        st.write(f"Độ tương đồng: {score:.4f}")
                                    except Exception as e:
                                        st.error(f"Không thể hiển thị ảnh: {str(e)}")
                        else:
                            st.warning("Không tìm thấy ảnh tương tự nào trong cơ sở dữ liệu.")
                    
                    except Exception as e:
                        st.error(f"Lỗi khi tìm kiếm: {str(e)}")
    
    # Tab quản lý cơ sở dữ liệu
    with tab2:
        st.header("Quản lý cơ sở dữ liệu ảnh")
        
        # Nhập đường dẫn đến thư mục ảnh
        image_dir = st.text_input(
            "Đường dẫn đến thư mục chứa ảnh chim", 
            placeholder="Ví dụ: /path/to/bird_images"
        )
        
        # Nút xử lý tất cả ảnh
        if st.button("Xử lý tất cả ảnh trong thư mục"):
            if image_dir and os.path.isdir(image_dir):
                with st.spinner("Đang xử lý ảnh và trích xuất đặc trưng..."):
                    try:
                        retrieval_system = BirdImageRetrieval()
                        processed_count = retrieval_system.process_all_images(image_dir)
                        retrieval_system.close()
                        
                        st.success(f"Đã xử lý thành công {processed_count} ảnh chim!")
                    except Exception as e:
                        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            else:
                st.error("Vui lòng nhập đường dẫn hợp lệ đến thư mục ảnh.")
        
        # Hiển thị thông tin về cơ sở dữ liệu
        if st.button("Hiển thị thông tin cơ sở dữ liệu"):
            try:
                retrieval_system = BirdImageRetrieval()
                if retrieval_system.image_paths:
                    st.info(f"Số lượng ảnh trong cơ sở dữ liệu: {len(retrieval_system.image_paths)}")
                    
                    # Hiển thị danh sách ảnh
                    if st.checkbox("Hiển thị danh sách ảnh"):
                        st.write("Danh sách ảnh trong cơ sở dữ liệu:")
                        for i, path in enumerate(retrieval_system.image_paths[:20]):  # Chỉ hiển thị 20 ảnh đầu tiên
                            st.write(f"{i+1}. {path}")
                        
                        if len(retrieval_system.image_paths) > 20:
                            st.write(f"... và {len(retrieval_system.image_paths) - 20} ảnh khác.")
                else:
                    st.warning("Cơ sở dữ liệu đang trống.")
                
                retrieval_system.close()
            except Exception as e:
                st.error(f"Lỗi khi truy vấn cơ sở dữ liệu: {str(e)}")


if __name__ == "__main__":
    main()