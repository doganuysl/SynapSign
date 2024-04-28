import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Modelin yüklenmesi
model = load_model('model_dnn.keras')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: 'What a Shame!'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            min_x = min(x_)
            min_y = min(y_)
            max_x = max(x_)
            max_y = max(y_)

            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max_x * W) - 10
            y2 = int(max_y * H) - 10

            # Modelin beklendiği özellik sayısına göre data_aux'u düzenle
            # Örnek olarak, eğer model 84 özellik bekliyorsa ve data_aux'ta 42 özellik varsa:
            data_aux.extend([0] * (84 - len(data_aux)))  # Eksik özellikleri sıfırlarla tamamla veya uygun bir yöntemle doldur

            prediction = model.predict(np.asarray([data_aux]))

            predicted_character = labels_dict[np.argmax(prediction)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
