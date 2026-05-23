import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from twilio.rest import Client
import av
import cv2
import time

# ==========================================
# TWILIO CONFIGURATION
# ==========================================
TWILIO_ACCOUNT_SID = "YOUR_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_PHONE_NUMBER = "YOUR_TWILIO_PHONE_NUMBER"
YOUR_PHONE_NUMBER = "YOUR_PHONE_NUMBER"

# Send SMS Function
def send_sms(message):

    try:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN
        )

        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=YOUR_PHONE_NUMBER
        )

        print("SMS Sent Successfully")

    except Exception as e:
        print("SMS Error:", e)

# ==========================================
# LOAD YOLO MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Better accuracy than yolov8n
    return YOLO("yolov8s.pt")

model = load_model()

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("🎥 Live Object Detection & Tracking")

st.sidebar.header("⚙️ Settings")

show_boxes = st.sidebar.checkbox(
    "Show Bounding Boxes",
    True
)

show_labels = st.sidebar.checkbox(
    "Show Labels",
    True
)

show_fps = st.sidebar.checkbox(
    "Show FPS",
    True
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.30,
    step=0.05
)

# ==========================================
# TWILIO SETTINGS
# ==========================================
enable_sms = st.sidebar.checkbox(
    "Enable Twilio SMS Alert",
    False
)

detect_object = st.sidebar.text_input(
    "Send Alert If Object Detected",
    "person"
)

# ==========================================
# SHOW DETECTABLE OBJECTS
# ==========================================
with st.expander("📦 Detectable Objects"):
    st.write(model.names)

# ==========================================
# VIDEO PROCESSOR
# ==========================================
class VideoProcessor(VideoTransformerBase):

    def __init__(self):
        self.prev_time = time.time()
        self.last_alert_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

        # Convert frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        # ==========================================
        # YOLO TRACKING
        # ==========================================
        results = model.track(
            img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            iou=0.5,
            verbose=False
        )

        annotated = img.copy()

        # ==========================================
        # DRAW RESULTS
        # ==========================================
        if results and len(results) > 0:

            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                ids = (
                    result.boxes.id.cpu().numpy()
                    if result.boxes.id is not None
                    else None
                )

                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = map(int, box)

                    class_id = int(classes[i])
                    label = model.names[class_id]

                    # Tracking ID
                    track_id = (
                        int(ids[i])
                        if ids is not None
                        else -1
                    )

                    # Green color
                    color = (0, 255, 0)

                    # ==========================================
                    # DRAW BOUNDING BOX
                    # ==========================================
                    if show_boxes:

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                    # ==========================================
                    # DRAW LABEL + ID
                    # ==========================================
                    if show_labels:

                        if track_id >= 0:
                            text = f"{label} ID:{track_id}"
                        else:
                            text = label

                        cv2.putText(
                            annotated,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                    # ==========================================
                    # TWILIO SMS ALERT
                    # ==========================================
                    if enable_sms:

                        if label.lower() == detect_object.lower():

                            current_time = time.time()

                            # Prevent spam messages
                            if current_time - self.last_alert_time > 20:

                                send_sms(
                                    f"⚠️ ALERT: {label} detected!"
                                )

                                self.last_alert_time = current_time

        # ==========================================
        # FPS COUNTER
        # ==========================================
        if show_fps:

            curr_time = time.time()

            fps = 1 / (curr_time - self.prev_time)

            self.prev_time = curr_time

            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# ==========================================
# START WEBCAM STREAM
# ==========================================
webrtc_streamer(
    key="yolo-object-tracking",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
