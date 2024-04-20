import cv2
import numpy as np
import pyautogui
import tkinter as tk
from functools import partial


# Создание класса для детектора лиц и глаз
class FaceEyeDetector:
    def __init__(self):
        # Загрузка предварительно обученных классификаторов для лиц и глаз
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Если лицо обнаружено
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Детекция глаз внутри области лица
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:  # Если оба глаза обнаружены
                    eye_centers = []
                    for (ex, ey, ew, eh) in eyes:
                        eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                        eye_centers.append(eye_center)

                    return (x, y, w, h), eye_centers

        return None, None


# Функция для управления курсором
def control_cursor(detector, start_button):
    if not start_button['state'] == 'disabled':
        global is_tracking
        is_tracking = True
        while is_tracking:
            ret, frame = cap.read()
            if not ret:
                break

            # Получение координат лиц и глаз
            face_coords, eye_centers = detector.detect(frame)

            # Если лицо обнаружено, рисуем прямоугольник вокруг него
            if face_coords is not None:
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Если глаза обнаружены, рисуем прямоугольники вокруг них
                if eye_centers is not None:
                    for (ex, ey) in eye_centers:
                        cv2.rectangle(frame, (ex - 5, ey - 5), (ex + 5, ey + 5), (0, 255, 0), 2)

                    # Вычисление среднего положения глаз
                    eye_center = np.mean(eye_centers, axis=0, dtype=np.int32)
                    screen_center = (frame.shape[1] // 2, frame.shape[0] // 2)

                    # Определение направления взгляда
                    horizontal_movement = 0
                    vertical_movement = 0
                    if eye_center[0] < screen_center[0] - 20:
                        horizontal_movement = -10  # Двигаем курсор влево на 10 пикселей
                    elif eye_center[0] > screen_center[0] + 20:
                        horizontal_movement = 10  # Двигаем курсор вправо на 10 пикселей
                    if eye_center[1] < screen_center[1] - 20:
                        vertical_movement = -10  # Двигаем курсор вверх на 10 пикселей
                    elif eye_center[1] > screen_center[1] + 20:
                        vertical_movement = 10  # Двигаем курсор вниз на 10 пикселей

                    pyautogui.moveRel(horizontal_movement, vertical_movement)

            # Отображение интерфейса с камерой
            cv2.imshow('Video', frame)

            # Ожидание нажатия клавиши 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Функция для остановки отслеживания
def stop_tracking():
    global is_tracking
    is_tracking = False


def start_camera():
    start_button['state'] = 'disabled'
    stop_button['state'] = 'normal'
    control_cursor(FaceEyeDetector(), start_button)
    start_button['state'] = 'normal'
    stop_button['state'] = 'disabled'


# Создание окна Tkinter
root = tk.Tk()
root.title("Eye Cursor Control")

# Создание кнопки "Старт"
start_button = tk.Button(root, text="Старт")
start_button.config(command=partial(control_cursor, FaceEyeDetector(), start_button))
start_button.pack()

# Создание кнопки "Стоп"
stop_button = tk.Button(root, text="Стоп", command=stop_tracking, state='disabled')
stop_button.pack()

# Создание кнопки "Запуск камеры"
camera_button = tk.Button(root, text="Запуск камеры", command=start_camera)
camera_button.pack()

# Запуск камеры
cap = cv2.VideoCapture(0)

root.mainloop()
