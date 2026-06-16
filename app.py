import streamlit as st
import cv2
from PIL import Image
from detector import detect_persons
from utils.helpers import read_image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

logo = Image.open("assets\\app_icon.png")
logo1 = Image.open("assets/app_icon.png")

st.set_page_config(
    page_title="CrowdVision AI",
    page_icon=logo,
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

div.stButton > button{
    width:100%;
    height:140px;
    border-radius:20px;
    font-size:24px;
    font-weight:bold;
    border:2px solid #00D4FF;
}

div.stButton > button:hover{
    transform:scale(1.03);
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.home-btn button{
    width:100px !important;
    height:40px !important;

    border-radius:18px !important;

    border:2px solid #00D4FF !important;

    font-size:18px !important;
    font-weight:bold !important;

    white-space:pre-line !important;
    word-break:keep-all !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ONLY reduce detect button size safely */
div.stButton > button {
    height: 55px !important;
    font-size: 16px !important;
    font-weight: 600 !important;

    border-radius: 12px !important;

    border: 2px solid #00D4FF !important;
    background: #1c1f26 !important;
    color: white !important;

    transition: all 0.2s ease;
}

/* hover stays same logic */
div.stButton > button:hover {
    transform: scale(1.03);
    border-color: #00ffcc !important;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "mode" not in st.session_state:
    st.session_state.mode = "home"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.image(logo1, width=250)

    st.markdown("<h2 style='text-align:center'>CrowdVision AI</h2>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <style>

    /* make sidebar buttons compact */
    section[data-testid="stSidebar"] button {
        width: 100% !important;
        height: 40px !important;
        font-size: 14px !important;
        font-weight: 600 !important;

        border-radius: 8px !important;
        margin-bottom: 6px !important;

        border: 1px solid #00D4FF !important;
        background: #1c1f26 !important;
        color: white !important;

        text-align: left !important;
        padding-left: 12px !important;
    }

    section[data-testid="stSidebar"] button:hover {
        transform: scale(1.02);
        border-color: #00ffcc !important;
    }

    </style>
    """, unsafe_allow_html=True)

    if st.button("🏠 Home"):
        st.session_state.mode = "home"
        st.rerun()

    if st.button("🎥 Live Webcam"):
        st.session_state.mode = "webcam"
        st.rerun()

    if st.button("🖼 Image Detector"):
        st.session_state.mode = "image"
        st.rerun()
# ==================================================
# HOME PAGE
# ==================================================

if st.session_state.mode == "home":
    if st.session_state.mode == "home":
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.image(
            logo1,
            width=1000
        )

        st.markdown("""
        <div style="
            width:300px;
            margin:auto;
            text-align:center;
            font-size:36px;
            font-weight:bold;
            color:white;
            margin-top:-10px;
        ">
            CrowdVision AI
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            width:300px;
            margin:auto;
            text-align:center;
            color:#9e9e9e;
            font-size:16px;
            margin-bottom:30px;
        ">
            AI Powered Multi-Person Detection System
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3 style="
            text-align:center;
            color:white;
            margin-bottom:25px;
        ">
            Select Detection Mode
        </h3>
        """, unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button(
                "🎥 Live Webcam",
                use_container_width=True
            ):
                st.session_state.mode = "webcam"
                st.rerun()

        with btn2:
            if st.button(
                "🖼 Image Detector",
                use_container_width=True
            ):
                st.session_state.mode = "image"
                st.rerun()

# ==================================================
# IMAGE DETECTOR PAGE
# ==================================================

elif st.session_state.mode == "image":

    st.markdown("""
    <h1 style='text-align:center'>
    🖼 Image Detection
    </h1>
    """, unsafe_allow_html=True)

    st.info("""
    Steps:

    1️⃣ Upload an image

    2️⃣ Click Detect Persons

    3️⃣ View Detection Result

    4️⃣ Download Processed Image
    """)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png","webp"]
    )

    if uploaded_file:

        image = read_image(uploaded_file)

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            image_rgb,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🚀 Detect Persons"):

            with st.spinner("Detecting Persons..."):

                result_image, person_count = detect_persons(image)


            st.success(
                f"Detected Persons: {person_count}"
                
            )

            st.success(
                f"Status : Completed"
            )

            st.success(
                f"Model: YOLOv8"
            )

            result_rgb = cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                )

            st.image(
                    result_rgb,
                    caption="Detection Result",
                    use_container_width=True
                )
            

            _, buffer = cv2.imencode(
                ".jpg",
                result_image
            )

            st.download_button(
                "⬇ Download Result",
                data=buffer.tobytes(),
                file_name="crowdvision_result.jpg",
                mime="image/jpeg"
            )

# ==================================================
# LIVE WEBCAM PAGE
# ==================================================

elif st.session_state.mode == "webcam":
    st.markdown("<h2 style='text-align:center'>Live Webcam Detection</h2>", unsafe_allow_html=True)

    start = st.button("Start Camera")

    frame_placeholder = st.empty()
    st.success(f"Status: {'LIVE' if start else 'OFFLINE'}")
    st.success(f"Model: YOLOv8")

    if start:
        camera = cv2.VideoCapture(0)

        while start:
            ret, frame = camera.read()
            if not ret:
                st.error("Camera not accessible")
                break

            result_frame, count = detect_persons(frame)


            cv2.putText(
                result_frame,
                f"Persons: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    
            frame_placeholder.image(result_frame, channels="BGR", use_container_width=True)

        camera.release()

