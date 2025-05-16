import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import csv
from rembg import remove
from PIL import Image
import io

# ===== 1. Đọc ảnh có Unicode bằng imdecode
def imread_unicode(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"[LỖI imread_unicode] {path}: {e}")
        return None

# ===== 2. Xóa nền bằng rembg (cải tiến cho ảnh chim)
def remove_background(image_cv2):
    try:
        # Chuyển ảnh từ OpenCV BGR -> PIL RGB
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Chuyển sang bytes để rembg xử lý tốt hơn
        with io.BytesIO() as input_io:
            pil_image.save(input_io, format="PNG")
            input_bytes = input_io.getvalue()

        # Gọi rembg để xóa nền
        output_bytes = remove(input_bytes)

        # Đọc ảnh đầu ra từ bytes và giữ kênh alpha
        with io.BytesIO(output_bytes) as output_io:
            result_image = Image.open(output_io).convert("RGBA")

        # Chuyển về định dạng OpenCV BGRA
        image_no_bg = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGBA2BGRA)
        return image_no_bg

    except Exception as e:
        print(f"[LỖI remove_background]: {e}")
        return image_cv2  # fallback nếu lỗi

# ===== 3. Biểu đồ màu HSV
def extract_hsv_histogram(image, h_bins=18, s_bins=8, v_bins=8):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist /= hist.sum() + 1e-6  # tránh chia cho 0
    return hist

# ===== 4. Màu chủ đạo bằng K-means
def extract_dominant_colors(image, k=3):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    percentages = counts / counts.sum()

    features = []
    for i in range(k):
        features.extend(colors[i])        # R, G, B
        features.append(percentages[i])   # Tỉ lệ %
    return np.array(features)

# ===== 5. Xử lý 1 ảnh
def process_image(image_path, hsv_bins=(18, 8, 8), kmeans_k=3):
    image = imread_unicode(image_path)
    if image is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path}")
        return None

    image = cv2.resize(image, (600, 400))

    # === Xóa nền
    image = remove_background(image)

    # === Nếu ảnh có alpha (BGRA), chuyển về BGR
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # === Trích xuất đặc trưng HSV và KMeans
    hsv_feat = extract_hsv_histogram(image, *hsv_bins)
    kmeans_feat = extract_dominant_colors(image, k=kmeans_k)

    return np.concatenate([hsv_feat, kmeans_feat])

# ===== 6. Xử lý toàn bộ thư mục
def extract_features_from_folder(folder_path, output_csv):
    header = [f"hsv_{i}" for i in range(18 + 8 + 8)]  # HSV = 34 chiều
    header += [f"kmeans_r{i//4+1}" if i % 4 == 0 else
               f"kmeans_g{i//4+1}" if i % 4 == 1 else
               f"kmeans_b{i//4+1}" if i % 4 == 2 else
               f"kmeans_p{i//4+1}" for i in range(3 * 4)]  # K-means = 12 chiều
    header = ['filename'] + header

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for file in os.listdir(folder_path):
            filename = os.fsdecode(file)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder_path, filename)
                features = process_image(path)
                if features is not None:
                    writer.writerow([filename] + features.tolist())
                    print(f"✔ Đã xử lý: {filename}")
                else:
                    print(f"✘ Lỗi xử lý: {filename}")

# ===== 7. Đường dẫn tương đối
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "..", "..", "data")  # Thư mục ảnh
    output_csv = os.path.join(base_dir, "features.csv")       # File output

    extract_features_from_folder(folder_path, output_csv)
