import streamlit as st
import cv2
import numpy as np
import astroalign as aa
import scipy.ndimage as ndimage
import tempfile
import os
import rawpy
from PIL import Image

# Set page config
st.set_page_config(page_title="StackerPy Web", layout="wide")
st.title("Astro-Stacker & Processor")

# --- Core Functions (from your notebook) ---
def multi_scale_wavelet_sharpen(img, s1=0.0, s2=0.0, s3=0.0):
    if img is None: return None
    img_float = img.astype(np.float32)
    g1 = ndimage.gaussian_filter(img_float, sigma=1.0)
    layer1_detail = img_float - g1
    g2 = ndimage.gaussian_filter(img_float, sigma=2.5)
    layer2_detail = g1 - g2
    g3 = ndimage.gaussian_filter(img_float, sigma=5.0)
    layer3_detail = g2 - g3
    sharpened = img_float + (layer1_detail * s1) + (layer2_detail * s2) + (layer3_detail * s3)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def align_frames(frames, top_percentage=20, target_type='Planetary'):
    num_keep = max(1, int(len(frames) * (top_percentage / 100)))
    best_frames = frames[:num_keep]
    reference_frame = best_frames[0]
    ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    
    aligned_frames = [reference_frame]
    progress_bar = st.progress(0)
    
    for i, img in enumerate(best_frames[1:]):
        if target_type == 'Planetary':
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
            try:
                _, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
                aligned = cv2.warpAffine(img, warp_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_frames.append(aligned)
            except:
                pass
        elif target_type == 'Deep Sky':
            try:
                aligned_img, _ = aa.register(img, reference_frame)
                aligned_frames.append(aligned_img)
            except:
                pass
        progress_bar.progress((i + 1) / (num_keep - 1))
    return aligned_frames

def stack_frames(aligned_frames, method='median'):
    stack = np.stack(aligned_frames, axis=0).astype(np.float32)
    integrated = np.zeros_like(aligned_frames[0], dtype=np.float32)
    height = stack.shape[1]
    chunk_size = max(1, height // 100) 
    
    for y in range(0, height, chunk_size):
        y_end = min(y + chunk_size, height)
        if method == 'median':
            integrated[y:y_end] = np.median(stack[:, y:y_end], axis=0)
        elif method == 'mean':
            integrated[y:y_end] = np.mean(stack[:, y:y_end], axis=0)
            
    return np.clip(integrated, 0, 255).astype(np.uint8)

# --- State Management ---
if 'stacked_image' not in st.session_state:
    st.session_state.stacked_image = None
if 'frames' not in st.session_state:
    st.session_state.frames = []

# --- UI Layout ---
tab1, tab2 = st.tabs(["1. Input & Extract", "2. Align, Stack & Sharpen"])

with tab1:
    st.header("Upload Video or Images")
    upload_type = st.radio("Input Type:", ['Video File', 'Multiple Images'])
    
    if upload_type == 'Video File':
        uploaded_video = st.file_uploader("Upload Astronomy Video (AVI, MP4)", type=['avi', 'mp4', 'mov'])
        if uploaded_video is not None and st.button("Extract Frames"):
            with st.spinner("Reading video frames..."):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                
                cap = cv2.VideoCapture(tfile.name)
                frames, scores = [], []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    frames.append(frame)
                    scores.append(sharpness)
                cap.release()
                
                sorted_indices = np.argsort(scores)[::-1]
                st.session_state.frames = [frames[i] for i in sorted_indices]
                st.success(f"Extracted and scored {len(frames)} frames!")
                st.image(cv2.cvtColor(st.session_state.frames[0], cv2.COLOR_BGR2RGB), caption="Best Frame Preview", width=500)

    else:
        uploaded_images = st.file_uploader("Upload Image Sequence", type=['png', 'jpg', 'tif', 'dng'], accept_multiple_files=True)
        if uploaded_images and st.button("Process Images"):
            with st.spinner("Scoring images..."):
                frames, scores = [], []
                for file in uploaded_images:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if frame is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                        frames.append(frame)
                        scores.append(sharpness)
                
                sorted_indices = np.argsort(scores)[::-1]
                st.session_state.frames = [frames[i] for i in sorted_indices]
                st.success(f"Loaded and scored {len(frames)} images!")

with tab2:
    st.header("Stacking & Processing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        target_type = st.radio("Target:", ['Planetary', 'Deep Sky'])
    with col2:
        stack_method = st.selectbox("Stack Method:", ['median', 'mean'])
    with col3:
        quality = st.slider("Keep Top % of Frames:", 1, 100, 20)

    if st.button("Align & Stack Frames") and len(st.session_state.frames) > 0:
        with st.spinner(f"Aligning top {quality}% of frames..."):
            aligned = align_frames(st.session_state.frames, top_percentage=quality, target_type=target_type)
        with st.spinner("Stacking frames..."):
            st.session_state.stacked_image = stack_frames(aligned, method=stack_method)
            st.success("Stacking Complete!")

    if st.session_state.stacked_image is not None:
        st.divider()
        st.subheader("Wavelet Sharpening")
        s1 = st.slider("L1 (Fine Details):", 0.0, 5.0, 0.0, 0.1)
        s2 = st.slider("L2 (Medium Details):", 0.0, 5.0, 0.0, 0.1)
        s3 = st.slider("L3 (Gross Details):", 0.0, 5.0, 0.0, 0.1)
        
        final_img = multi_scale_wavelet_sharpen(st.session_state.stacked_image, s1, s2, s3)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(st.session_state.stacked_image, cv2.COLOR_BGR2RGB), caption="Raw Stack")
        with c2:
            st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="Wavelet Sharpened")
            
        # Download Button
        is_success, buffer = cv2.imencode(".png", final_img)
        io_buf = buffer.tobytes()
        st.download_button(label="Download Final Image", data=io_buf, file_name="astro_output.png", mime="image/png")