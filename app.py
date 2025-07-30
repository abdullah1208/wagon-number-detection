
##################################################################################

import streamlit as st
import os
import tempfile
from processing import VideoProcessor
import cv2

def main():
    st.title("Automated Train Wagon Detection and Number Reading System")
    
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        if st.button("Start Processing"):
            processor = VideoProcessor()
            progress_bar = st.progress(0)
            video_placeholder = st.empty()
            results = []

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            processed_frames = 0
            for processed_frame, frame_results in processor.process_video(video_path):
                processed_frames += 1
                results = frame_results
                video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                progress = processed_frames / (total_frames // 3)
                progress_bar.progress(min(progress, 1.0))

            st.success("Processing Complete!")
            st.subheader(f"Overall Train Direction: {processor.train_direction}")
            st.subheader("Analysis Results")

            # Table Header
            header_cols = st.columns([2, 2, 2, 1.5, 2.5], gap="large")
            header_cols[0].markdown("**Wagon Image**")
            header_cols[1].markdown("**Number Image**")
            header_cols[2].markdown("**Wagon Number**")
            header_cols[3].markdown("**Direction**")
            header_cols[4].markdown("**Timestamp**")
            
            st.markdown("---")

            # Table Rows
            for result in results:
                cols = st.columns([2, 2, 2, 1.5, 2.5], gap="medium")
                
                # Wagon Image
                with cols[0]:
                    if os.path.exists(result['wagon_image']):
                        st.image(result['wagon_image'], width=130)
                    else:
                        st.error("Missing")
                
                # Number Image
                with cols[1]:
                    if result['number_image'] != "missing" and os.path.exists(result['number_image']):
                        st.image(result['number_image'], width=130)
                    else:
                        st.error("Missing")
                
                # Wagon Number
                cols[2].markdown(f"<div style='font-size: 18px; margin-top: 20px;'>{result['wagon_number']}</div>", 
                               unsafe_allow_html=True)
                
                # Direction
                direction = processor.direction_info.get(result['track_id'], processor.train_direction)
                color = "#4CAF50" if direction == "Right" else "#2196F3"
                cols[3].markdown(
                    f"<div style='color:{color}; font-weight:bold; font-size: 18px; margin-top: 20px;'>{direction}</div>", 
                    unsafe_allow_html=True
                )
                
                # Timestamp
                cols[4].markdown(f"<div style='font-size: 14px; margin-top: 20px;'>{result['timestamp']}</div>", 
                               unsafe_allow_html=True)
                
                # Row separator
                st.markdown("<div style='margin-bottom: 40px; border-bottom: 1px solid #eee;'></div>", 
                          unsafe_allow_html=True)

            # Cleanup
            os.unlink(video_path)
            progress_bar.empty()

if __name__ == "__main__":
    main()

