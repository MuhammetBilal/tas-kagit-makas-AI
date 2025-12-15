import math

import cv2
import random
from ultralytics import YOLO

model = YOLO("best.pt")

class_names = model.names

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


player_move = None
computer_move = None
result_text = "Oynamak icin 'SPACE'e bas"


moves = ["Rock", "Paper", "Scissors"]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model.predict(img, stream=True, verbose=False)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cls = int(box.cls[0])
            current_detection = class_names[cls]

            conf = math.ceil((box.conf[0] * 100)) / 100
            label = f'{current_detection} {conf}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            player_move = current_detection

    key = cv2.waitKey(1)

    if key == 32:
        if player_move in moves:

            computer_move = random.choice(moves)

            if player_move == computer_move:
                result_text = "BERABERE!"
            elif (player_move == "Rock" and computer_move == "Scissors") or \
                    (player_move == "Paper" and computer_move == "Rock") or \
                    (player_move == "Scissors" and computer_move == "Paper"):
                result_text = "KAZANDIN! :)"
            else:
                result_text = "KAYBETTIN! :("
        else:
            result_text = "Hamle Algilanmadi!"
            computer_move = "???"

    cv2.rectangle(img, (0, 0), (1280, 150), (0, 0, 0), cv2.FILLED)

    cv2.putText(img, f"Sen: {player_move}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.putText(img, f"PC: {computer_move}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (100, 100, 255), 3)
    cv2.putText(img, result_text, (500, 80), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)


    cv2.imshow("Tas Kagit Makas AI", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()