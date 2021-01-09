# Imports
from cv2 import cv2
import numpy as np
import os
import face_recognition

# Declaring constants
COLOR = (255,0,0)
FRAME_RESIZE = 0.5
STROKE = 2
TOLERANCE = 0.6
FONT_THICKNESS = 2
BUFFER_FRAME_LENGHT = 4
MODEL = "hog"
FOLDER_PATH = "Images"

# Initializing variables
buffer = []
frame_counter = 0
known_face_encodings = []
known_face_labels = []

# This function loads and encodes all images in 'Images' folder
def load_images(folder):

    for image in os.listdir(folder):
        image_directory = str(os.path.join(folder,image))

        my_image = face_recognition.load_image_file(image_directory)
        known_face_encodings.append(face_recognition.face_encodings(my_image)[0])

        label = str(image).strip(".jpg")
        label = label.strip(".png")

        known_face_labels.append(label)  

load_images(FOLDER_PATH)

# Creating a video capture
cap = cv2.VideoCapture(0)

while True:
    # Setting up video input
    ret, frame = cap.read()
    rescaled_frame = cv2.resize(frame, (0,0), fx = FRAME_RESIZE, fy = FRAME_RESIZE)

    # Detecting face locations and encodes faces in the frame
    face_locations = face_recognition.face_locations(rescaled_frame,model=MODEL)
    face_encodings = face_recognition.face_encodings(rescaled_frame,face_locations) 

    # Looping through all faces detected in the frame
    for location, encoding in zip(face_locations, face_encodings):
        
        name = "Unknown"

        # Compares face to known and identified images
        face = face_recognition.compare_faces(known_face_encodings, encoding, TOLERANCE)

        # Identifies the closest match from the face on the screen
        face_dis = face_recognition.face_distance(known_face_encodings,encoding)
        best_face_match = np.argmin(face_dis)
        
        # Checks if the closest match is a match to one of the known faces
        if face[best_face_match]:
            name = known_face_labels[best_face_match]
            buffer.clear()
        else:
            buffer.append(True)

        # Logs off the user if unknown face is detected
        if buffer == [True]*BUFFER_FRAME_LENGHT:
            os.system("shutdown -l")
        
        # Mapping cordinates of face to points
        top_left = (int(location[3]/FRAME_RESIZE), int(location[0]/FRAME_RESIZE))
        bottom_right = (int(location[1]/FRAME_RESIZE), int(location[2]/FRAME_RESIZE))
        top_label = (int(location[3]/FRAME_RESIZE), int(location[2]/FRAME_RESIZE))
        bot_label = (int(location[1]/FRAME_RESIZE), int(location[2]/FRAME_RESIZE) + 22)

        # Drawing labels and boxes for people 
        cv2.rectangle(frame, top_left, bottom_right, COLOR, STROKE)
        cv2.rectangle(frame, top_label, bot_label, COLOR, STROKE)
        cv2.putText(frame, name, (top_label[0] + 10, top_label[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    
    # Displaying Edited Camera View
    cv2.imshow('Camera', frame)

    # Program Terminated when 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()