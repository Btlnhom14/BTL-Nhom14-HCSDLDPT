# bird_search_web.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import euclidean_distances
from Xay_dung_bo_thuoc_tinh.Thuoc_tinh_mau_sac.Extract_features_and_save import extract_color_features
from Xay_dung_bo_thuoc_tinh.Thuoc_tinh_hinh_dang.bird_shape_features_extraction import extract_shape_features
from Xay_dung_bo_thuoc_tinh.Thuoc_tinh_ket_cau.Extract_bird_texture import extract_texture_features

# --- Load features ---
@st.cache_data
def load_features():
    shape = pd.read_csv("./Xay_dung_bo_thuoc_tinh/Thuoc_tinh_hinh_dang/bird_shape_features.csv")
    texture = pd.read_csv("./Xay_dung_bo_thuoc_tinh/Thuoc_tinh_ket_cau/bird_texture_features.csv")
    color = pd.read_csv("./Xay_dung_bo_thuoc_tinh/Thuoc_tinh_mau_sac/features.csv")

    shape = shape.set_index("file_name")
    texture = texture.set_index("image_file")
    color = color.set_index("filename")

    common_index = shape.index.intersection(texture.index).intersection(color.index)
    combined = pd.concat([shape.loc[common_index],
                          texture.loc[common_index],
                          color.loc[common_index]], axis=1)
    return combined

# --- Trích đặc trưng thật sự cho ảnh truy vấn ---
def extract_query_features(img_path):
    shape_vec = extract_shape_features(img_path)
    texture_vec = extract_texture_features(img_path)
    color_vec = extract_color_features(img_path)
    return np.hstack([shape_vec, texture_vec, color_vec]).reshape(1, -1)

# --- Load data ---
st.title("🔍 Hệ thống tìm kiếm ảnh chim")
feature_db = load_features()

# --- Upload ảnh ---
uploaded = st.file_uploader("Tải lên ảnh chim truy vấn", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Ảnh truy vấn", use_column_width=True)

    with open("query.png", "wb") as f:
        f.write(uploaded.getbuffer())

    # --- Extract feature for query ---
    query_vec = extract_query_features("query.png")

    # --- Compare with dataset ---
    db_vectors = feature_db.values
    distances = euclidean_distances(query_vec, db_vectors)[0]

    feature_db["distance"] = distances
    top3 = feature_db.sort_values("distance").head(3)

    st.subheader("🔗 Top 3 ảnh giống nhất:")
    for fname in top3.index:
        img_path = os.path.join("./data", fname)
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption=fname, width=300)
        else:
            st.warning(f"Không tìm thấy ảnh: {fname}")
