import cv2      #imports all libraries
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)       #initializes the camera
if not cap.isOpened():
    print("Camera can't open")
    exit()
mpHands = mp.solutions.hands        #creates instances
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection
face_detection = mpFace.FaceDetection()

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

mpBodyMesh = mp.solutions.pose
bodyPose = mpBodyMesh.Pose()

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
pTime = 0
cTime = 0
while True:
    success, img = cap.read()       #checks and reads for image
    if not success:
        print("Failed frame read")
        continue
    if img is None:
        print("Empty frame is received")
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if imgRGB is None:
        print("Failed to convert color space of input")
        continue
    results = hands.process(img)        
    results_face = face_detection.process(imgRGB)
    results_mesh = faceMesh.process(imgRGB)
    results_body = bodyPose.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:            #the if statement is if the camera detecting something and for-loop is going through the loop looking for the object
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    if results_face.detections:
        for detection in results_face.detections:
            bounding_box = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x = int(bounding_box.xmin * w)
            y = int(bounding_box.ymin * h)
            width = int(bounding_box.width * w)
            height = int(bounding_box.height * h)
            cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
            
    if results_mesh.multi_face_landmarks:
        for faceLms in results_mesh.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=mpDraw.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))
    if results_body.pose_landmarks:
            mpDraw.draw_landmarks(img, results_body.pose_landmarks, mpBodyMesh.POSE_CONNECTIONS)
    
    cTime = time.time()             #fps counter
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (18,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255), 2)

    cv2.imshow("Image", img)        #display the image
    cv2.waitKey(1)              #1 milisecond to wait for key event to occur


