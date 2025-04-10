import cv2, time, numpy as np, os, argparse
from multiprocessing import Pool

# Load pre-trained models
face_net = cv2.dnn.readNet("res10_300x300_ssd_iter_140000.caffemodel", "deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
gender_list = ['Male', 'Female']

# Gender detection function
def detect_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.426, 87.768, 114.896), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return gender_list[preds[0].argmax()]

# Motion detection
def detect_motion(args):
    prev, curr = args
    diff = cv2.absdiff(prev, curr)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return cv2.dilate(thresh, None, iterations=2)

# Create folder if saving enabled
def make_output_dir():
    path = "output_frames"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Process a single frame (for video/webcam)
def process_frame(frame, prev_gray, pool):
    motion = False
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    if prev_gray is not None:
        thresh = pool.apply_async(detect_motion, ((prev_gray, gray),)).get()
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 500: continue
            motion = True
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                gender = detect_gender(face)
                label = f"{gender}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

    return frame, gray, motion

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['webcam', 'video', 'image'], required=True,
                        help="Choose input mode: webcam/video/image")
    parser.add_argument('--file', help="Path to video or image file (required for video/image mode)")
    parser.add_argument('--save', action='store_true', help="Save output frames to folder")
    args = parser.parse_args()

    save_path = make_output_dir() if args.save else None
    pool = Pool(2)
    prev_gray = None
    count = 0
    start = time.time()

    if args.mode == 'image':
        if not args.file:
            print("Error: --file is required for image mode")
            return
        frame = cv2.imread(args.file)
        if frame is None:
            print("Error loading image.")
            return
        processed, _, _ = process_frame(frame, None, pool)
        cv2.imshow("Output", processed)
        if save_path:
            cv2.imwrite(os.path.join(save_path, "output_image.jpg"), processed)
        cv2.waitKey(0)

    else:
        cap = cv2.VideoCapture(0 if args.mode == 'webcam' else args.file)
        if not cap.isOpened():
            print("Error accessing video.")
            return

        while True:
            ret, frame = cap.read()
            if not ret: break
            count += 1

            processed, prev_gray, motion = process_frame(frame, prev_gray, pool)

            fps = count / (time.time() - start)
            status = "Motion Detected" if motion else "No Motion"
            cv2.putText(processed, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(processed, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Motion & Gender Detection", processed)

            if save_path:
                filename = f"frame_{count:04d}.jpg"
                cv2.imwrite(os.path.join(save_path, filename), processed)

            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap.release()
    pool.close()
    pool.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
