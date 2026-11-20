import cv2
import mediapipe as mp
import time
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "videos", "1.mp4")

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75) # increasing the minimum detection confidence to 75%
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
draw_keypoints = False
pTime = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        break # Exit if video ended or frame is empty
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting the image from BGR to RGB
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections): # Looping through the detections which contains label_id, score, location_data: relative bounding box (xmin, ymin, width, height) and 6 relative key points (x, y)
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            bboxC = detection.location_data.relative_bounding_box # Getting the relative bounding box
            h, w, c = img.shape # Getting the height, width and channels of the image
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) # Displaying the score on the image with scale 2, purple color and thickness 2
            # mpDraw.draw_landmarks(img, detection.location_data.relative_keypoints, mpFaceDetection.FACE_CONNECTIONS)
            if draw_keypoints:
                for i, keypoint in enumerate(detection.location_data.relative_keypoints):
                    cx, cy = int(keypoint.x * w), int(keypoint.y * h)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime > pTime else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Displaying the fps on the image with scale 3, purple color and thickness 3
    cv2.imshow("Image", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break