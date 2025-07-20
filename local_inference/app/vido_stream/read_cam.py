import logging
import cv2

class Video:

    def __init__(self, num_cam: int = 0):
        self.logger = logging.getLogger(__name__)
        self.cap = cv2.VideoCapture(num_cam)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def get_frames(self):
        
        while True:
            ret, frame = self.cap.read()

            if not ret:
                self.logger.error("""
                    The video camera was not detected
                    """)
                break
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def get_label(self):
        ...