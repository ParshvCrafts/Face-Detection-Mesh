import cv2
import mediapipe as mp
import time
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "videos", "7.mp4")

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
mpDraw = mp.solutions.drawing_utils
red_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))
green_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))

cap = cv2.VideoCapture(video_path)
draw_keypoints = False
pTime = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        break # Exit if video ended or frame is empty
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting the image from BGR to RGB
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, red_drawing_spec, green_drawing_spec)
            for id, lm in enumerate(faceLms.landmark): # gets the x, y, z normialized coordinates of 468 landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)                
                # cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Displaying the fps on the image with scale 3, purple color and thickness 3
    cv2.imshow("Image", img)
    if cv2.waitKey(15) & 0xFF == ord("q"):
        break