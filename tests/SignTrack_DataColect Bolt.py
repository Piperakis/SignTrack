# Importing dependancies

import cv2
import cvzone
import numpy as np
import os
import mediapipe as mp
from pathlib import Path
from essentials import mediapipe_detection, extract_keypoints, display_styled_landmarks

# Dataset export location, Changing requires changes in Signtrack_Train
data_path = os.path.join('test')

# Actions that we try to detect, Changing requires changes in Signtrack_Train
signs = np.array(["sorry"])

# Number of sequences to be collected for each action
no_datapacks = 3

# Frames per sequence, Changing requires changes in Signtrack_Train
seq_length = 24

# Choose camera input
cap = cv2.VideoCapture(1)

# Resize camera input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def existing_data(sign):
    '''
    Checks for existing data in the
    dataset folder, to avoid errors while
    saving new data.
    '''
    existing_data = 0
    path = Path(data_path + '/' + sign)
    if path.exists():
        existing_data = len(os.listdir(data_path + '/' + sign))
    return existing_data


for sign in signs:
    '''
    Generates new empty forlders,
    after taking into account existing data,
    where new data wil be saved.
    '''
    exdt = existing_data(sign)
    for seq in range(no_datapacks * 2):
        try:
            os.makedirs(os.path.join(data_path, sign,
                        str((seq) + exdt)))
        except:
            pass

# Import image assets and resize them to fit the output frame

AssetCol = cv2.imread("Assets/Asset_col.png", cv2.IMREAD_UNCHANGED)
AssetCol = cv2.resize(AssetCol, (0, 0), None, 0.5, 0.5)

AssetNCol = cv2.imread("Assets/Asset_ncol.png", cv2.IMREAD_UNCHANGED)
AssetNCol = cv2.resize(AssetNCol, (0, 0), None, 0.5, 0.5)

AssetCircle = cv2.imread("Assets/Circle.png", cv2.IMREAD_UNCHANGED)
AssetCircle = cv2.resize(AssetCircle, (0, 0), None, 0.5, 0.5)

AssetBar = cv2.imread("Assets/Bar.png", cv2.IMREAD_UNCHANGED)
AssetBar = cv2.resize(AssetBar, (0, 0), None, 0.5, 0.5)


def Graphics(img, sign, sequence, collecting):
    '''
        Add the text and the assets to the final
        display output
    '''
    # Adds the circle around the number of sequence
    # It is displayed at the 1/3 of frame's width (centered)
    # And at the frame's heignt minus it's height
    img = cvzone.overlayPNG(
        img, AssetCircle, [round((wb/3-wc/2)/2), hb-hf])

    # Adds the bar around the current collected sign
    # It is displayed at the 1/4 of frame's width from the right (centered)
    # Vertically, it is located at the lowest part of the picture
    img = cvzone.overlayPNG(
        img, AssetBar, [round(wb-wb/4-wc/2), hb-hf])

    # This adds the text of the sequence counter
    # Its location is exactly the same as of its circle around it
    # Only difference is that its possition also changes
    #  regarding the number of letters
    cv2.putText(img, str(sequence), (round((wb/3-wc/2)/2 - (len(str(sequence)))*7.5 + 28),
                hb-hf+23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Prints the curent collected sign in the frame
    # Its location is exactly the same as of its bar around it
    # Only difference is that its possition also changes
    #  regarding the number of letters
    cv2.putText(img, str(sign), (round(wb-wb/4-wc/2 + wbar/2-len(sign)*7.5), hb-hf+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Displaying the collecting indication at the center
    if collecting == True:
        img = cvzone.overlayPNG(
            img, AssetCol, [round(wb/2-wf/2), hb-hf])
    else:
        img = cvzone.overlayPNG(
            img, AssetNCol, [round(wb/2-wf/2), hb-hf])
    return img


# Setting mediapipe parameters amd initiaize a diferent model for the flipped image
'''
Mediapipe uses an LSTM model, just like SignTrack, that means that the results are made 
based on a sequence of data. Thus when trying to make predictions on the flipped image it is 
important to utilize a different version of the Mediapipe model, to avoid the model's confusion.
'''


holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

holisticf = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Loading the dimentions of cap and the assets and image
hf, wf, cf = AssetCol.shape
hc, wc, cc = AssetCircle.shape
hbar, wbar, cbar = AssetBar.shape
hb, wb, cb = (480, 640, 3)

pics = []

def ProcessImg(pics,seq):
    framenum=0
    for frame in pics:
        img, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        npy_path = os.path.join(
            data_path, sign, str((2 * seq) + exdt), str(framenum))           
        np.save(npy_path, keypoints)

        # Save the landmarks of the flipped image as an numpy array
        frame_fliped = cv2.flip(frame, 1)

        img_flipped, results_flipped = mediapipe_detection(
            frame_fliped, holisticf)

        keypoints_flipped = extract_keypoints(results_flipped)
        npy_path_flipped = os.path.join(
            data_path, sign, str((2 * seq + 1) + exdt), str(framenum))
        np.save(npy_path_flipped,keypoints_flipped)
        framenum+=1
    pics = []


# Loop through each sign
img=np.zeros((480, 640, 3), np.uint8)
for sign in signs:
    exdt = existing_data(sign) - (no_datapacks * 2)
    # Loop through sequences aka videos
    for seq in range(no_datapacks):

        # Loop through video length aka sequence length
        for frame_num in range(seq_length):

            # Read camera feed
            ret, frame = cap.read()
            pics.append(frame)

            # Detect hand and pose landmarks
            img, results = mediapipe_detection(frame, holistic)

            # Loading the dimentions of cap and the assets

            # Draw landmarks
            display_styled_landmarks(img, results)

            cv2.imshow('SignTrack Data Collect',
                    Graphics(img, sign, seq, False))            
            cv2.imshow('SignTrack Data Collect',
                    Graphics(img, sign, seq, True))


            # Export keypoints
            keypoints = extract_keypoints(results)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        '''
            The wait logic,
            If the collected sequence is on its first frame, 
            changes the apearence of the image accordingly.
        '''

        cv2.imshow('SignTrack Data Collect',
                        Graphics(img, sign, seq, False))
        cv2.waitKey(1000)
        if pics:
            ProcessImg(pics,seq)
            pics.clear()

cap.release()
cv2.destroyAllWindows()
