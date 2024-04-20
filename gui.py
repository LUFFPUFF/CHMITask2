import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyautogui
import dlib
pyautogui.FAILSAFE = False

class EyeDetector(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.camera = None
        self.running = False

    def start_camera(self):
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
        self.camera = cv2.VideoCapture(cv2.CAP_ANY)
        if not self.camera.isOpened():
            error_message = "Error: Could not open camera."
            self.frame_ready.emit(np.zeros((480, 640, 3), dtype=np.uint8))
            QtWidgets.QMessageBox.critical(None, "Camera Error", error_message)
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.running = True

        while self.running:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = self.detect_eyes(frame)
                self.frame_ready.emit(frame)
            else:
                break

        self.camera.release()

    def detect_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye = []
            right_eye = []
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye.append((x, y))
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye.append((x, y))

            # Определение направления взгляда
            left_eye_center = np.mean(left_eye, axis=0).astype(int)
            right_eye_center = np.mean(right_eye, axis=0).astype(int)

            if left_eye_center[0] < right_eye_center[0]:
                pyautogui.moveRel(-10, 0)  # Двигаем курсор влево
            elif left_eye_center[0] > right_eye_center[0]:
                pyautogui.moveRel(10, 0)  # Двигаем курсор вправо

            # Визуализация (опционально)
            for eye in [left_eye, right_eye]:
                for i, (x, y) in enumerate(eye):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                cv2.polylines(frame, [np.array(eye)], True, (0, 255, 0), 1)

        return frame

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.eye_detector = EyeDetector()
        self.setWindowTitle("Eye Detection GUI")
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.start_button = QtWidgets.QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera_thread)
        self.stop_button = QtWidgets.QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera_thread)
        self.stop_button.hide()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

    def start_camera_thread(self):
        self.camera_thread = QtCore.QThread()
        self.eye_detector.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.eye_detector.start_camera)
        self.camera_thread.start()
        self.eye_detector.frame_ready.connect(self.update_frame)
        self.start_button.hide()
        self.stop_button.show()

    def stop_camera_thread(self):
        self.eye_detector.running = False
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.eye_detector.camera.release()
            self.camera_thread.quit()
            self.camera_thread.wait()
        self.stop_button.hide()
        self.start_button.show()

    def closeEvent(self, event):
        self.stop_camera_thread()
        event.accept()

    def update_frame(self, frame):
        try:
            if self.isVisible():
                scaled_frame = cv2.resize(frame, (self.video_label.width(), self.video_label.height()))
                image = QtGui.QImage(scaled_frame.data, scaled_frame.shape[1], scaled_frame.shape[0],
                                     QtGui.QImage.Format_BGR888)
                pixmap = QtGui.QPixmap.fromImage(image)
                self.video_label.setPixmap(pixmap)
            else:
                self.stop_camera_thread()
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping camera thread.")
            self.stop_camera_thread()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
