import cv2
import time
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor

# Load pre-trained models
face_net = cv2.dnn.readNet("res10_300x300_ssd_iter_140000.caffemodel", "deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
gender_list = ['Male', 'Female']

def detect_motion(args):
    start_time = time.time()
    prev, curr = args
    diff = cv2.absdiff(prev, curr)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    motion_time = time.time() - start_time
    return thresh, motion_time

def detect_gender(face_img):
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.896), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    gender_time = time.time() - start_time
    return gender_list[preds[0].argmax()], gender_time

def process_frame(frame, prev_gray, use_parallel, executor):
    start_time = time.time()
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    motion = False
    motion_time = 0

    if prev_gray is not None:
        if use_parallel and executor:
            future = executor.submit(detect_motion, (prev_gray, gray))
            thresh, motion_time = future.result()
        else:
            thresh, motion_time = detect_motion((prev_gray, gray))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            motion = True
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    coords = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                faces.append(face)
                coords.append((x1, y1, x2, y2))

    gender_time = 0

    if faces:
        if use_parallel and executor:
            futures = [executor.submit(detect_gender, face) for face in faces]
            for future, (x1, y1, x2, y2) in zip(futures, coords):
                gender, g_time = future.result()
                gender_time += g_time
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            for face, (x1, y1, x2, y2) in zip(faces, coords):
                gender, g_time = detect_gender(face)
                gender_time += g_time
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    frame_processing_time = time.time() - start_time
    return frame, gray, motion, motion_time, gender_time, frame_processing_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['webcam', 'video', 'image'], required=True)
    parser.add_argument('--file', help='Path to input file')
    parser.add_argument('--process', choices=['sequential', 'parallel'], default='sequential')
    parser.add_argument('--save', action='store_true', help='Save output')
    args = parser.parse_args()

    use_parallel = args.process == 'parallel'
    executor = ThreadPoolExecutor(max_workers=4) if use_parallel else None
    prev_gray = None
    count, start = 0, time.time()

    if args.mode == 'image':
        image = cv2.imread(args.file)
        frame, _, _, motion_time, gender_time, frame_processing_time = process_frame(image, None, False, None)
        print(f"Motion detection took {motion_time:.4f} seconds")
        print(f"Gender detection took {gender_time:.4f} seconds")
        print(f"Frame processing took {frame_processing_time:.4f} seconds")
        if args.save:
            cv2.imwrite('output_image.jpg', frame)
        cv2.imshow("Image Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(0 if args.mode == 'webcam' else args.file)
    if not cap.isOpened():
        print("Failed to open video source.")
        return

    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            frame, prev_gray, motion, motion_time, gender_time, frame_processing_time = process_frame(
                frame, prev_gray, use_parallel, executor)

            fps = count / (time.time() - start)
            status = "Motion Detected" if motion else "No Motion"
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if args.save:
                out.write(frame)

            cv2.imshow("Motion & Gender Detection", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
    finally:
        if executor:
            executor.shutdown()
        cap.release()
        if args.save:
            out.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start
        fps = count / total_time
        process_type = "[SEQUENTIAL]" if not use_parallel else "[PARALLEL]"
        print(f"{process_type} Processed {count} frames in {total_time:.1f} seconds")
        print(f"{process_type} FPS: {fps:.2f}")

if __name__ == '__main__':
    main()
