import cv2
import time
import winsound
from ultralytics import YOLO
from datetime import datetime
import os

MODEL = YOLO("yolov8n.pt")

ZONE = (200, 100, 450, 400)
CONF = 0.5
ALERT_TIME = 2

RESTRICTED_CLASSES = {"cell phone", "laptop"}

os.makedirs("violations", exist_ok=True)

log_file = "events.log"

def log_event(msg):
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")

def intersects(x1,y1,x2,y2,zone):
    zx1, zy1, zx2, zy2 = zone
    return not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)

def main():
    cap = cv2.VideoCapture(0)

    zone_start_time = None
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = MODEL(frame, conf=CONF)[0]

        detected_in_zone = False
        person_detected = False

        for box in results.boxes:
            cls = int(box.cls[0])
            name = MODEL.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name in RESTRICTED_CLASSES:
                if intersects(x1,y1,x2,y2,ZONE):
                    detected_in_zone = True

            if name == "person":
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if ZONE[0] <= cx <= ZONE[2] and ZONE[1] <= cy <= ZONE[3]:
                    person_detected = True

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, name, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

        # 🔊 Restricted object alert logic (GLOBAL TIMER)
        if detected_in_zone:
            if zone_start_time is None:
                zone_start_time = time.time()

            elapsed = time.time() - zone_start_time

            if elapsed >= ALERT_TIME and not alert_triggered:
                winsound.Beep(1500, 800)
                alert_triggered = True
        else:
            zone_start_time = None
            alert_triggered = False

        # 📸 Person capture logic (no tracking)
        if person_detected:
            filename = f"violations/person_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            log_event("Person detected in restricted zone")

        # Draw zone
        cv2.rectangle(frame, (ZONE[0],ZONE[1]), (ZONE[2],ZONE[3]), (255,0,0),2)

        cv2.imshow("Alternative System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
