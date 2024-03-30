import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

prev_time = 0
cur_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cur_time = time.time()
    fps = 1/(cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)