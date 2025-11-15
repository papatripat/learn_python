import cv2
import winsound

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



camera = cv2.VideoCapture(0)

tracker = cv2.legacy.TrackerKCF_create()
tracking = False

while True:
    ret, frame = camera.read()
    if not ret:
        break

    if not tracking:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            tracker.init(frame, (x, y, w, h))
            tracking = True

            winsound.Beep(1000, 300)
    else:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking wajah...", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            tracking = False
            cv2.putText(frame, "Wajah hilang - Mencari ulang",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Deteksi & Tracking Wajah", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
