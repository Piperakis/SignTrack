import numpy as np
import cv2
import numpy as np
import mediapipe as mp


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


phrases = {('how you'): 'how are you',
           ('fine'): "I'm fine",
           ('me fine me'): "I am fine",
           ('good.by.e'): "goodbye",
           ('Good.by.e'): "goodbye"}


def mediapipe_detection(img, model):
    """
    It is used to extract the hand and pose landmarks of the frame 
    """
    # COLOR CONVERSION BGR 2 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False                  # Image is no longer writeable
    results = model.process(img)                 # Make prediction
    img.flags.writeable = True                   # Image is now writeable
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return img, results


def draw_landmarks(img, results):
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections


def display_styled_landmarks(img, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(128, 128, 0), thickness=1, circle_radius=0),
                              mp_drawing.DrawingSpec(
                                  color=(128, 128, 0), thickness=1, circle_radius=0)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(128, 128, 0), thickness=1, circle_radius=0),
                              mp_drawing.DrawingSpec(
                                  color=(128, 128, 0), thickness=1, circle_radius=0)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(204, 153, 204), thickness=1, circle_radius=0),
                              mp_drawing.DrawingSpec(
                                  color=(204, 153, 204), thickness=1, circle_radius=0)
                              )


# Checks if the hands are on the scene
def HandsOnScene(results):
    '''
    Gets as an input the results from Medipipe predictions and outputs
    whether the hands are present in the frame
    '''
    if not results.left_hand_landmarks and not results.right_hand_landmarks:
        return False
    else:
        return True


def extract_keypoints(results):
    """
    This is utilized to convert the keypoint results to a flat numpy array, thats is easy to save and proccess
    """
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([lh, rh, pose])


def grammar_correct(sentence):
    """
    Grammar in sign language often differs from the on in written speech.
    Using this function, the sentence is corrected from simple grammatical 
    errors
    """
    for key in phrases:
        if key in sentence:
            sentence = sentence.replace(key, phrases[key])
    return sentence
