from re import I
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import cv2
import cvzone
import shutil
import numpy as np
import mediapipe as mp
from essentials import mediapipe_detection, display_styled_landmarks, extract_keypoints, HandsOnScene, grammar_correct
import Packr
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "oliverguhr/fullstop-punctuation-multilang-large")
model = AutoModelForTokenClassification.from_pretrained(
    "oliverguhr/fullstop-punctuation-multilang-large")

pun = pipeline('ner', model=model, tokenizer=tokenizer)


# The number of frames per sequence that the model has been trained on
seq_length = 24

# Choose camera input
cap = cv2.VideoCapture(1)

# Resize camera input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

shutil.rmtree('tmp', True)  # Erase previous temporary data

# Unpacks the Model.Pack file and loads the features(signs) of the model
signs = np.load(Packr.ModelIDUnpack())

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Setting up the model archtecture
model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(24, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

# Loading the model
model.load_weights("tmp/Insights/SignTrack.h5")

shutil.rmtree('tmp', True)  # Erase temporary data


# Import image assets and resize them to fit the output frame
TopBar = cv2.imread("Assets/TopBar.png", cv2.IMREAD_UNCHANGED)
TopBar = cv2.resize(TopBar, (0, 0), None, 0.36, 0.42)

BottomBar = cv2.imread("Assets/BottomBar.png", cv2.IMREAD_UNCHANGED)
BottomBar = cv2.resize(BottomBar, (0, 0), None, 0.36, 0.42)


def prob_vis(res, actions, input_frame):
    '''
    It adds to the image a visualization
    of the chances that an action apears in the
    image
    Returns the final image after the process

    '''
    output_frame = input_frame.copy()
    resfin = {}
    resfinshorted = {}
    to_add = 0

    # Create a dict including the possibility of each sign being in the frames as a percentage
    for num, prob in enumerate(res):
        resfin.update({round(prob*100, 2): actions[num]})

    # Adding '--' for non existing values
    to_add = 7 - len(resfin)
    if 0 < to_add:
        for i in range(to_add):
            resfin.update({0.0001*i: '--'})

    # Creating a shorted version of the dictionary with the most probable signs going first
    for i in sorted(resfin, reverse=True):
        resfinshorted[i] = resfin[i]

    # Initializing lists with the shorted signs and their probability of their presence
    ResValShorted = list(resfinshorted.values())
    ResKeysShorted = list(resfinshorted.keys())

    # Positioning the assets and the text on the image
    """ 
    Adds the bottom bar on the frame, the position is calculated :
    For the X axis: by calculating half of the width of the frame
     minus half of the width of the bottom bar
    For the Y axis: by calculating the height of the frame minus
     the height of the bar while leavig 17 pixels space from the top
    """

    output_frame = cvzone.overlayPNG(
        output_frame, BottomBar, [round(wc/2-wb/2), round(hc-hb-17)])

    """ 
    Adds the top bar on the frame, the position is calculated :
    For the X axis: by calculating half of the difference between 
    the width of the frame and the width of the bottom bar
    For the Y axis: by calculating the height of the bar minus 
    the height of the bar devided by 1.5
    """
    output_frame = cvzone.overlayPNG(
        output_frame, TopBar, [round((wc-wt)/2), round(ht-(ht/1.5))])

    for i in range(6):
        """
        Prints the 6 most probable signs on the top bar,
         the position is calculated knowing that:
        For the X axis: the distance between each window is
         105 pixels, while the first window is 52 pixels 
         from 0 and that each letter has aproximately a 5.25
         pixel width 
        The Y axis possition remains the same at 35 pixels
        """
        cv2.putText(output_frame, str(ResValShorted[i]), ((round((52 + 105*(i)-len(ResValShorted[i])*5.25))),
                                                          35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(3):
        """
        Prints the probabilities of the 3 most probable signs
         on the top bar, the position is calculated knowing that:
        For the X : the distance between each window is
         105 pixels, while the first window is 46 from 0axis
        The Y axis possition remains the same at 53 pixels
        """
        cv2.putText(output_frame, str(round(ResKeysShorted[i], 1)), ((round((46 + 105*(i)))),
                                                                     53), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

    """
    Adds the sentence text on the bottom bar, the position is calculated by:
    For the X axis: dividing the width of the frame by 6
    For the Y axis: calculating the difference of the height of the
     frame and the height of the bar while keeping a 3 pixel 
     distance from the top of the bar
    """
    cv2.putText(output_frame, text.capitalize(), (round(wc/6), hc-hb+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    return output_frame


# Initializing empty values
text = ''
seq, sentence = [], []
HandsOnPrevFrames = [False]
threshold = 0.90
res = np.zeros(shape=signs.shape[0])
# Set mediapipe model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)


while cap.isOpened():

    # Read feed
    ret, frame = cap.read()

    # Make detections
    img, results = mediapipe_detection(frame, holistic)

    # Loading the dimentions of cap and the assets
    hb, wb, cb = BottomBar.shape
    hc, wc, cc = img.shape
    ht, wt, ct = TopBar.shape
    # Draw landmarks
    display_styled_landmarks(img, results)

    # 2. Prediction logic

    # Creates a history the Hands on Scene results
    HandsOnPrevFrames.append(HandsOnScene(results))
    HandsOnPrevFrames = HandsOnPrevFrames[-44:]

    keypoints = extract_keypoints(results)

    # If the hands are in the frame append the leypoints in seq to call the model to make predictions later
    if True in HandsOnPrevFrames[-5:]:
        if HandsOnScene(results):
            seq.append(keypoints)
            seq = seq[-24:]
            text = grammar_correct((' '.join(sentence)))
        # Else if hands are not in the scene for the last 5 frames clear sequence data
        else:
            text = grammar_correct((' '.join(sentence)))
            seq = []
            res = np.zeros(shape=len(signs))

# if the hands are not visible in teh last 44 frames clear the sentence and process the displayed text
    if not True in HandsOnPrevFrames:
        if len(sentence) > 0:
            sentence = text.capitalize()
            # Calling the text model to add punctuation
            output_json = pun(sentence)
            sentence = ""
            text = ''
            # Adding the predicted punctuation in the final sentence
            for n in output_json:
                result = n['word'].replace(
                    '▁', ' ') + n['entity'].replace('0', '')
                sentence += result
            # Capitalizing the needed letters
            sentence = sentence.split('. ')
            for word in sentence:
                text += word[0].upper() + word[1:] + '. '
            text = (grammar_correct(text[1:-2])).capitalize()
            sentence = []

    # If there are 24 frames in seq then call the model to predict
    if len(seq) == seq_length:
        res = model.predict(np.expand_dims(seq, axis=0))[0]

        """
        In case there is more than 65% the amount of needed data
        call the model to predict in a new version of seq 
        with randomly duplicated frames
        """

    elif len(seq) >= seq_length * 0.65:
        missing = seq_length - len(seq)
        seqpros = seq
        for i in range(missing):
            rand = random.randint(0, len(seq)-1)
            seqpros.insert(rand, seq[rand])
        res = model.predict(np.expand_dims(seqpros, axis=0))[0]
        seqpros = []
        res = np.zeros(shape=len(signs))

    # 3. Viz logic

    # If the probabillity of the most probable sign is more than the thershold
    if res[np.argmax(res)] > threshold:
        # Checking whether it is different than the last prediction then append it in the sentence
        if len(sentence) > 0:
            if signs[np.argmax(res)] != sentence[-1]:
                sentence.append(signs[np.argmax(res)])
        # If it is empty just add the prediction in the sentence
        else:
            sentence.append(signs[np.argmax(res)])

    # Keep the last 6 phrases in sentence
    sentence = sentence[-6:]

    # Viaualizing probabilities
    img = prob_vis(res, signs, img)

    # Display the final image
    cv2.imshow('SignTrack', img)

    # End properly
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# Terminating the window
cap.release()
cv2.destroyAllWindows()
