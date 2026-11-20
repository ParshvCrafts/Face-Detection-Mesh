import cv2
import mediapipe as mp
import time
import os

class FaceDetection:
    def __init__(self, detection_confidence=0.55, draw_keypoints=False):
        self.detection_confidence = detection_confidence
        self.bboxs = []
        self.landmark_id = 0
        self.draw_keypoints = draw_keypoints
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.detection_confidence) 
        self.mpDraw = mp.solutions.drawing_utils # Importing the drawing utils module to draw the landmarks and connections

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting the image from BGR to RGB
        self.results = self.faceDetection.process(imgRGB) # Processing the image to detect the face, it outputs a class object 
        
        
        if self.results.detections: # Checking if the face is detected from the class object
            for id, detection in enumerate(self.results.detections): # Looping through the detections which contains label_id, score, location_data: relative bounding box (xmin, ymin, width, height) and 6 relative key points (x, y)
                    # However, the x and y coordinates are in decimal places, so we need to convert them to pixels
                    h, w, c = img.shape # Getting the height, width and channels of the image
                    bboxC = detection.location_data.relative_bounding_box # Getting the relative bounding box
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    self.bboxs.append([id, bbox, detection.score])
                    
                    if draw:
                        img = self.DrawBoundingBox(img, bbox)
                        cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2) # Displaying the score on the image with scale 2, purple color and thickness 2
                        # mpDraw.draw_landmarks(img, detection.location_data.relative_keypoints, mpFaceDetection.FACE_CONNECTIONS)
                        if self.draw_keypoints:
                            for i, keypoint in enumerate(detection.location_data.relative_keypoints):
                                cx, cy = int(keypoint.x * w), int(keypoint.y * h)
                                cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)
                                
        return img, self.bboxs

    def DrawBoundingBox(self, img, bbox, draw=True, l = 30, thickness = 5, rec_thickness = 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h # x1 and y1 are diagonal points, x1 is bottom right x and y1 is bottom right y
        if draw:
            # Top Left x, y
            cv2.line(img, (x, y), (x + l, y), (255, 0, 255), thickness)
            cv2.line(img, (x, y), (x, y + l), (255, 0, 255), thickness)
            # Top Right x1, y
            cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), thickness)
            cv2.line(img, (x1, y), (x1, y + l ), (255, 0, 255), thickness)
            # Bottom Left x, y1
            cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), thickness)
            cv2.line(img, (x, y1), (x, y1 - l ), (255, 0, 255), thickness)
            # Bottom Right x1, y1
            cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), thickness)
            cv2.line(img, (x1, y1), (x1, y1 - l ), (255, 0, 255), thickness)

            cv2.rectangle(img, bbox, (255, 0, 255), rec_thickness)

        return img

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "videos", "7.mp4")
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetection(detection_confidence=0.50, draw_keypoints= False)
    while True:
        success, img = cap.read()
        if not success or img is None:
            break # Exit if video ended or frame is empty
        img, bboxs = detector.findFaces(img, draw= True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (5, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) # Displaying the fps on the image with scale 3, purple color and thickness 3
        cv2.imshow("Image", img)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    
    # Releasing the camera and closing the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()