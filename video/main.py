import cv2
import mediapipe as mp
import numpy as np
import time


class GestureRecognizer:
    def __init__(self):
        # Инициализация MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Жесты, которые мы будем распознавать
        self.gestures = {
            'open_hand': self.is_open_hand,
            'fist': self.is_fist,
            'victory': self.is_victory,
            'thumbs_up': self.is_thumbs_up,
            'pointing': self.is_pointing
        }

    def get_landmark_coordinates(self, hand_landmarks, image_shape):
        """Преобразование нормализованных координат в пиксельные"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * image_shape[1])
            y = int(landmark.y * image_shape[0])
            landmarks.append((x, y))
        return landmarks

    def calculate_distance(self, point1, point2):
        """Расчет расстояния между двумя точками"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_open_hand(self, landmarks):
        """Распознавание открытой ладони"""
        try:
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 6, 10, 14, 18]

            for tip, pip in zip(finger_tips, finger_pips):
                if landmarks[tip][1] < landmarks[pip][1]:
                    return False
            return True
        except:
            return False

    def is_fist(self, landmarks):
        """Распознавание кулака"""
        try:
            finger_tips = [8, 12, 16, 20]
            finger_mcps = [5, 9, 13, 17]

            for tip, mcp in zip(finger_tips, finger_mcps):
                if landmarks[tip][1] > landmarks[mcp][1]:
                    return False
            return True
        except:
            return False

    def is_victory(self, landmarks):
        """Распознавание жеста 'V' (победа)"""
        try:
            index_up = landmarks[8][1] < landmarks[6][1]
            middle_up = landmarks[12][1] < landmarks[10][1]
            ring_down = landmarks[16][1] > landmarks[14][1]
            pinky_down = landmarks[20][1] > landmarks[18][1]

            return index_up and middle_up and ring_down and pinky_down
        except:
            return False

    def is_thumbs_up(self, landmarks):
        """Распознавание большого пальца вверх"""
        try:
            thumb_up = landmarks[4][1] < landmarks[3][1]

            other_fingers_down = all(
                landmarks[tip][1] > landmarks[mcp][1]
                for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]
            )

            return thumb_up and other_fingers_down
        except:
            return False

    def is_pointing(self, landmarks):
        """Распознавание указательного жеста"""
        try:
            index_up = landmarks[8][1] < landmarks[6][1]

            other_fingers_down = all(
                landmarks[tip][1] > landmarks[mcp][1]
                for tip, mcp in [(12, 9), (16, 13), (20, 17)]
            )

            return index_up and other_fingers_down
        except:
            return False

    def recognize_gesture(self, landmarks):
        """Основная функция распознавания жеста"""
        for gesture_name, gesture_func in self.gestures.items():
            if gesture_func(landmarks):
                return gesture_name
        return "unknown"

    def process_frame(self, frame):
        """Обработка одного кадра"""
        try:
            # Конвертируем BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Обрабатываем кадр с MediaPipe
            results = self.hands.process(rgb_frame)

            gesture = "No hand detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Получаем координаты landmarks
                    landmarks = self.get_landmark_coordinates(hand_landmarks, frame.shape)

                    # Распознаем жест
                    gesture = self.recognize_gesture(landmarks)

                    # Рисуем landmarks на руке
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Отображаем результат
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame, gesture
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return frame, "error"


def test_camera():
    """Тестирование доступности камеры"""
    print("Тестирование камеры...")

    # Пробуем разные индексы камер
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Камера {i} доступна. Разрешение: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return i
            cap.release()

    print("Ни одна камера не доступна!")
    return -1


def main():
    # Тестируем камеру
    camera_index = test_camera()

    if camera_index == -1:
        print("Попробуйте следующие решения:")
        print("1. Проверьте, подключена ли веб-камера")
        print("2. Закройте другие программы, использующие камеру")
        print("3. Попробуйте использовать другую камеру")
        return

    # Инициализация распознавателя жестов
    recognizer = GestureRecognizer()

    # Захват видео с веб-камеры
    cap = cv2.VideoCapture(camera_index)

    # Устанавливаем разрешение
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Даем камере время на инициализацию
    time.sleep(2)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру!")
        return

    print("Запуск распознавания жестов. Нажмите 'q' для выхода.")
    print("Покажите руку в камеру для распознавания жестов.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр с камеры!")
            break

        # Обрабатываем кадр
        processed_frame, gesture = recognizer.process_frame(frame)

        # Отображаем FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Отображаем результат
        cv2.imshow('Gesture Recognition', processed_frame)

        # Выводим распознанный жест в консоль (не слишком часто)
        if gesture != "No hand detected" and frame_count % 10 == 0:
            print(f"Распознан жест: {gesture}")

        # Выход по нажатию 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Пауза по пробелу
            cv2.waitKey(0)

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    print("Программа завершена.")


# Альтернативная версия с использованием статического изображения для тестирования
def test_with_image():
    """Тестирование на статическом изображении"""
    print("Тестирование на статическом изображении...")

    # Создаем черное изображение для теста
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST MODE - No camera required", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    recognizer = GestureRecognizer()

    while True:
        # Создаем копию изображения для обработки
        frame = test_image.copy()

        # Обрабатываем кадр
        processed_frame, gesture = recognizer.process_frame(frame)

        # Отображаем результат
        cv2.imshow('Gesture Recognition (Test Mode)', processed_frame)
        print(f"Жест: {gesture} (режим тестирования)")

        # Выход по нажатию 'q'
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Обновление каждую секунду
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Выберите режим:")
    print("1 - Режим с камерой (по умолчанию)")
    print("2 - Тестовый режим (без камеры)")

    choice = input("Введите номер режима (1 или 2): ").strip()

    if choice == "2":
        test_with_image()
    else:
        main()