import cv2
import mediapipe as mp
import time
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "videos", "2.mp4")


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=3, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.face = []
        self.faces = []

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils # Importing the drawing utils module to draw the landmarks and connections
        self.red_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))
        self.green_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting the image from BGR to RGB
        self.results = self.faceMesh.process(imgRGB) # Processing the image to detect the face, it outputs a class object 

        if self.results.multi_face_landmarks: # Checking if the face is detected from the class object
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.red_drawing_spec, self.green_drawing_spec)

                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.face.append([cx, cy])  # Appending the x, y coordinates of the landmarks to the face list
                self.faces.append(self.face) # Appending the face list to the faces list
                self.face = [] # Resetting the face list
        return img, self.faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        if not success or img is None:
            break # Exit if video ended or frame is empty
        img, faces = detector.findFaceMesh(img, draw=True)
        # if faces:
            # print(faces[0])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime > pTime else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Displaying the fps on the image with scale 3, purple color and thickness 3
        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # Releasing the camera and closing the window
    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()
