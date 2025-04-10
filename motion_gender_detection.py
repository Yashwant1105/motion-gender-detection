import cv2, time, numpy as np
from multiprocessing import Pool

face_net = cv2.dnn.readNet("res10_300x300_ssd_iter_140000.caffemodel", "deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

gender_list = ['Male', 'Female']

def detect_motion(args):
    prev, curr = args
    diff = cv2.absdiff(prev, curr)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return cv2.dilate(thresh, None, iterations=2)

def detect_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return gender_list[preds[0].argmax()]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return print("Camera error!")

    pool, prev, count, start = Pool(2), None, 0, time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            count += 1

            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

            if prev is not None:
                thresh = pool.apply_async(detect_motion, ((prev, gray),)).get()

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                motion = False

                for c in contours:
                    if cv2.contourArea(c) < 500: continue
                    motion = True
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Face Detection for Gender
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                             (104.0, 177.0, 123.0), swapRB=False, crop=False)
                face_net.setInput(blob)
                detections = face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.7:
                        box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0],
                                                                   frame.shape[1], frame.shape[0]])
                        (x1, y1, x2, y2) = box.astype("int")

                        face = frame[y1:y2, x1:x2]
                        if face.shape[0] > 0 and face.shape[1] > 0:
                            gender = detect_gender(face)
                            label = f"{gender}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 0, 0), 2)

                fps = count / (time.time() - start)
                status = "Motion Detected" if motion else "No Motion"
                cv2.putText(frame, f"Status: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Motion & Gender Detection", frame)

            prev = gray.copy()
            if cv2.waitKey(30) & 0xFF == 27: break

    finally:
        pool.close(); pool.join(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
