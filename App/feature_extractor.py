# feature_extractor.py
import cv2
import numpy as np
from skimage import feature, color # Sử dụng skimage.feature và skimage.color
from skimage.feature import graycomatrix, graycoprops
from collections import Counter

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
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_fg = self.remove_background(img_rgb)
        img_for_processing = img_fg.copy()

        color_features = self.extract_color_features(img_for_processing) # Dự kiến 44 đặc trưng
        shape_features = self.extract_shape_features(img_for_processing) # Dự kiến 20 đặc trưng
        texture_features = self.extract_texture_features(img_for_processing) # Dự kiến 52 đặc trưng

        # Kết hợp tất cả các đặc trưng thành một vector đặc trưng (tổng cộng 44+20+52 = 116 đặc trưng)
        all_features = np.concatenate([color_features, shape_features, texture_features])
        return all_features

    def extract_color_features(self, img):
        """Trích xuất đặc trưng màu sắc: Biểu đồ HSV và màu chủ đạo."""
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        h_bins, s_bins, v_bins = 8, 8, 8
        h_hist = cv2.calcHist([hsv_img], [0], None, [h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv_img], [1], None, [s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [v_bins], [0, 256])

        h_hist = cv2.normalize(h_hist, h_hist).flatten() # 8
        s_hist = cv2.normalize(s_hist, s_hist).flatten() # 8
        v_hist = cv2.normalize(v_hist, v_hist).flatten() # 8

        pixels = hsv_img.reshape((-1, 3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        counts = Counter(labels.flatten())
        total_pixels = float(len(labels))
        dominant_color_features = np.zeros(k * 4) # 5 * 4 = 20

        for i in range(k):
            dominant_color_features[i*4 + 0] = centers[i][0] / 180.0  # H normalized
            dominant_color_features[i*4 + 1] = centers[i][1] / 255.0  # S normalized
            dominant_color_features[i*4 + 2] = centers[i][2] / 255.0  # V normalized
            dominant_color_features[i*4 + 3] = counts.get(i, 0) / total_pixels  # Tỷ lệ

        color_features = np.concatenate([h_hist, s_hist, v_hist, dominant_color_features]) # 8+8+8+20 = 44
        return color_features

    def extract_shape_features(self, img):
        """Trích xuất đặc trưng hình dạng: Tỷ lệ hình học, Mô-men Hu và Fourier Descriptors."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(20) # Phù hợp với số lượng đặc trưng hình dạng (3+7+10=20)

        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        geometric_ratios = [aspect_ratio, circularity, solidity] # 3 đặc trưng

        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10) # 7 đặc trưng

        contour_points = largest_contour.reshape(-1, 2)
        fourier_descriptors = np.zeros(10) # 10 đặc trưng
        if len(contour_points) >= 5: # Cần đủ điểm cho FFT
            complex_coords = contour_points[:, 0] + 1j * contour_points[:, 1]
            fourier_result = np.fft.fft(complex_coords)
            if len(fourier_result) > 10 and np.abs(fourier_result[0]) > 1e-9: # Cần ít nhất 11 hệ số
                 fourier_descriptors = np.abs(fourier_result[1:11]) / np.abs(fourier_result[0])


        shape_features = np.concatenate([
            geometric_ratios,
            hu_moments,
            fourier_descriptors
        ]) # 3 + 7 + 10 = 20 đặc trưng
        return shape_features

    def extract_texture_features(self, img):
        """Trích xuất đặc trưng kết cấu: Haralick (GLCM) và Gabor."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 4 angles
        glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)

        haralick_features_list = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'] # 5 properties
        for prop in properties:
            feature_val = graycoprops(glcm, prop).flatten() # Mỗi prop cho ra 4 giá trị (1 per angle)
            haralick_features_list.append(feature_val)
        haralick_features = np.concatenate(haralick_features_list) # 5 * 4 = 20 đặc trưng

        num_theta, num_sigma, num_frequency = 4, 2, 2
        theta_vals = np.linspace(0, np.pi, num_theta, endpoint=False) # Sử dụng endpoint=False để tránh trùng lặp 0 và pi
        sigma_vals = [1, 3]
        frequency_vals = [0.1, 0.4]
        gabor_features_list = []

        for t in theta_vals:
            for s in sigma_vals:
                for f in frequency_vals:
                    kernel = cv2.getGaborKernel((21, 21), s, t, 1/f, 0.5, 0, ktype=cv2.CV_32F) # Gabor wavelength is 1/frequency
                    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    gabor_features_list.extend([np.mean(filtered_img), np.std(filtered_img)])
        gabor_features = np.array(gabor_features_list) # 4*2*2*2 = 32 đặc trưng

        texture_features = np.concatenate([haralick_features, gabor_features]) # 20 + 32 = 52 đặc trưng
        return texture_features