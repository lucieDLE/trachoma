import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import nrrd
import numpy as np
import os
import SimpleITK as sitk
# st.set_page_config(layout="wide")
# import cv2
import matplotlib.colors as mcolors
import pdb
# start from 23470
@st.cache_data

def compute_eye_bbx(seg, label=1, pad=0):
    shape = seg.shape

    # Find non-zero (or non-label) indices
    ij = np.argwhere(np.squeeze(seg) != 0)

    if ij.size == 0:
        return np.array([0, 0, 0, 0], dtype=int)

    # Compute bounding box
    xmin = np.clip(np.min(ij[:, 1]), 0, shape[1])
    ymin = np.clip(np.min(ij[:, 0]), 0, shape[0])
    xmax = np.clip(np.max(ij[:, 1]), 0, shape[1])
    ymax = np.clip(np.max(ij[:, 0]), 0, shape[0])

    return np.array([xmin, ymin, xmax, ymax], dtype=int)

def load_csv(path):
    df = pd.read_csv(path)
    return df

# df = df.iloc[]
# mnt = '/Volumes/EGower/'
csv_path = "segmentation_cleaned_opening.csv"
df = pd.read_csv(csv_path)

# Initialize current image index
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Navigation controls
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("⏪ Previous") and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()

with col2:
    if st.button("Next ⏩") and st.session_state.current_index < len(df) - 1:
        st.session_state.current_index += 1
        st.rerun()

# Display current position
st.write(f"Image {st.session_state.current_index + 1} of {len(df)}")

# Progress bar
progress = (st.session_state.current_index + 1) / len(df)
st.progress(progress)

# Load current image
row = df.iloc[st.session_state.current_index]
seg_path = os.path.join(row['segmentation path'])

try:
    # img, * = nrrd.read(seg_path)
    seg = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())
    bbx_eye = compute_eye_bbx(seg, pad=0.05)
    seg_cropped = seg[bbx_eye[1]:bbx_eye[3],bbx_eye[0]:bbx_eye[2] ]

    img = seg_cropped  # Fixed: using seg as img since that's what was loaded
    h1, w1 = img.shape[:2]

    # Setup figure
    fig, ax = plt.subplots()
    img_display = img[::3, ::3]
    ax.imshow(img_display, cmap='gray')
    ax.set_title(f"Frame ID: {row.get('image_path', 'N/A')} | File: {os.path.basename(seg_path)}")
    H, W = img_display.shape[:2]
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Error reading image file `{seg_path}`: {e}")
    st.write("Skipping to next image...")
    if st.session_state.current_index < len(df) - 1:
        st.session_state.current_index += 1
        st.rerun()
    else:
        st.stop()

# Initialize session state
if "bad_frames" not in st.session_state:
    st.session_state.bad_frames = set()

if "frame_comments" not in st.session_state:
    st.session_state.frame_comments = {}

# Get current frame ID
image_path = row.get('image_path', st.session_state.current_index)

# Tagging interface
flagged = st.checkbox("Mark this frame as bad", key=f"flag_{st.session_state.current_index}")
if flagged:
    st.session_state.bad_frames.add(image_path)
else:
    st.session_state.bad_frames.discard(image_path)

# Comment input
comment = st.text_area(
    "Add a comment for this frame:", 
    value=st.session_state.frame_comments.get(image_path, ""),
    help="Enter any observations or notes about this frame",
    key=f"comment_{st.session_state.current_index}"
)

# Update comment in session state
if comment:
    st.session_state.frame_comments[image_path] = comment
elif image_path in st.session_state.frame_comments:
    # Remove empty comments
    del st.session_state.frame_comments[image_path]

st.write(f"Total bad frames flagged: {len(st.session_state.bad_frames)}")
if st.session_state.frame_comments:
    st.write(f"Frames with comments: {len(st.session_state.frame_comments)}")

# Export flagged frames to CSV
if st.button("Export flagged frames"):
    # Create a list to store all frame data
    export_data = []
    
    # Add flagged bad frames
    # bad_df = df[df["image path"].isin(st.session_state.bad_frames)].copy()
    bad_df = df.iloc[list(st.session_state.bad_frames)]
    for idx, row in bad_df.iterrows():
        frame_data = row.to_dict()
        frame_data['flagged_as_bad'] = True
        frame_data['comment'] = st.session_state.frame_comments.get(idx, "")
        export_data.append(frame_data)
    
    # Add frames with comments (even if not flagged as bad)
    for idx, comment in st.session_state.frame_comments.items():
        if idx not in st.session_state.bad_frames:
            # Find the frame in the original dataframe
            frame_row = df.iloc[idx]
            if not frame_row.empty:
                frame_data = frame_row.to_dict()
                frame_data['flagged_as_bad'] = False
                frame_data['comment'] = comment
                export_data.append(frame_data)
    
    if export_data:
        # Create dataframe from export data
        export_df = pd.DataFrame(export_data)
        
        # Try to load existing file and append, or create new one
        output_file = "flagged_bad_frames_cat_v2.csv"
        try:
            existing_df = pd.read_csv(output_file)
            # Remove duplicates based on image_path to avoid re-adding same frames
            existing_df = existing_df[~existing_df['image path'].isin(export_df['image path'])]
            combined_df = pd.concat([existing_df, export_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = export_df
        
        combined_df.to_csv(output_file, index=False)
        st.success(f"Exported {len(export_data)} frames to `{output_file}`")
        
        # Show summary
        bad_count = sum(1 for item in export_data if item['flagged_as_bad'])
        comment_count = sum(1 for item in export_data if item['comment'])
        st.write(f"- {bad_count} frames flagged as bad")
        st.write(f"- {comment_count} frames with comments")
    else:
        st.warning("No frames to export (no flagged frames or comments)")

# Optional: Jump to specific image
with st.expander("Jump to specific image"):
    target_index = st.number_input(
        "Go to image number:", 
        min_value=1, 
        max_value=len(df), 
        value=st.session_state.current_index + 1
    ) - 1
    
    if st.button("Go to image"):
        if 0 <= target_index < len(df):
            st.session_state.current_index = target_index
            st.rerun()

# Optional: Show current session summary
if st.expander("Session Summary"):
    if st.session_state.bad_frames:
        st.write("**Flagged frames:**", list(st.session_state.bad_frames))
    if st.session_state.frame_comments:
        st.write("**Comments:**")
        for fid, comment in st.session_state.frame_comments.items():
            st.write(f"- Frame {fid}: {comment}")