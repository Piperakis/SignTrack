<p align="center">
    <img src="Assets/readme/SignTrack.png">
</p>

## In this guide you can learn about

* [What is SignTrack](#What&nbsp;is&nbsp;SignTrack)
* [How it works](#How&nbsp;it&nbsp;works)
* [Showcase](#Quick&nbsp;Guide)
  * [Data&nbsp;Collection](#Data&nbsp;Collection)
  * [Model&nbsp;Training](#Model&nbsp;Training)
  * [SignTrack&nbsp;Main](#SignTrack&nbsp;Main)
  * [Dataset](#Dataset)
* [Dependances](#Dependances)
* [Setup](#setup)
* [Troubleshoting](#troubleshoting)
* [Credits](#Credits)

## What&nbsp;is&nbsp;SignTrack

SignTrack is a sign language transcriber. It analyzes, processes, and recognizes sign language in real-time, with exceptional accuracy and efficiency. SignTrack helps make computers more open for everyone by taking a human-centric approach to computing.

<br />

## How&nbsp;it&nbsp;works

SignTrack utilizes a state-of-the-art LSTM model that predicts based on a sequence of data, enabling the detection of whole phrases and moving signs. To further improve precision and efficiency, SignTrack has been trained only on key hand and pose landmarks, that have been extracted using MediaPipe. That way it also remains accurate on every skin shade.

<br />

## Showcase

### Data&nbsp;Collection

Making the data collecting experience friendlier for the user has been one of our top priorities. That's why we made sure to create an easy-to-use data collection user interface, even for those with minimal coding skills.

Data plays a fundamental role in creating a great model that's both accurate and has great performance. The data collection program has been designed to generate data suitable for training the model. Added to that, it has been made to help our users to create an accurate model. With breaks between training sessions and intuitive design, anyone can create a good training dataset.

Another feature of SignTrack Data Collect is flipping the image to generate data as if you signed with the other hand too! Creating a model that can make equally accurate predictions on both hands.

Privacy is at the center of SignTrack. The collected data is free of personal data, like raw images. It only stores numerical values of the keypoint positions as NumPy arrays, making users feel more comfortable exchanging datasets.

<p align="center">
    <img src="Assets/readme/Collecting1.gif">
    <img src="Assets/readme/Collecting2.gif">
</p>

## Model&nbsp;Training

<p align="center">
    <img img width="300" height="50" src="Assets/readme/AutoTrain.png">
</p>

Training a neural network can become confusing. Everything in SignTrack is automated, with the power of AutoTrain. AutoTrain automatically sets training parameters, such as the number of epochs and data split. Forming a training process that requires minimal or no adjustment from the user.

AutoTrain makes sure that you get the most out of your dataset. It sets all the training parameters for you while automatically saving the best-performing model. What's more AutoTrain has now been redefined to make for an even faster tracking experience. FastTrack has been implemented right through the model training proccess, a new way of data augmentation, that improves the overall performance of the model, not only regarding FastTrack.

## SignTrack&nbsp;Main

<p align="center">
    <img img width="300" height="50" src="Assets/readme/FastTrack.png">
</p>

Utilizing the created model turned out to be an equally fundamental part of the project. Some people sign faster than the 24 frames that the model requires for making predictions. FastTrack is built into the __SignTrack&nbsp;main.py__ to solve this problem. It randomly duplicates the frames that the model's predictions are based on, until it gets the desired amount of frames. At the same time, making sure that the model makes predictions only on frames in which the hands are on the frame, enhancing once again the overall performance.

Optimizations form an uninterrupted experience. SignTrack is made to use resources only when needed. For example, the SignTrack model is only called to make predictions when the hands have been visible on the scene. While the needed punctuation is only predicted after the user has completed forming the sentence.

The consistent, unique, and identifiable design continues in the main program while keeping on the display the needed information to understand how the model performs on specific signs. That can save a lot of time for those who work on making their SignTrack model.

<p align="center">
    <img src="Assets/readme/Detect1.gif">
    <img src="Assets/readme/Detect2.gif">
</p>

## Dataset

SignTrack comes with a dataset, consisted of 193,945 collected keypoint sets for ASL, and can be easily extended. If the given model does not perform as expected you can always add more data using DataCollect. When creating your dataset remember to make sure to try signing from different angles and positions to create a more generalized model, that will work on a broader spectrum of angles.

### Illustration

All of the illustrations have been designed from the ground up for SignTrack. In the assets is a directory with SVG versions of them. You can always tinker with them using your favorite open-source illustration tool, like Inkscape.
<br />

## Dependances

This project has been developed using:

* Python: 3.7
* Tensorflow: 2.5
* OpenCV: 4.1.2.30
* Scikit-Learn
* Matplotlib
* Mediapipe
* Cvzone
* Sentencepiece
* Transformers

<br />

## Setup

SignTrack uses python poetry to make installation a breeze. Dealing with TensorFlow can often be difficult. We made sure that this is not the case with SignTrack.

### Windows installation

* Install [Python&nbsp;3.7](https://www.python.org/downloads/windows/)

* Install Python [Poetry](https://python-poetry.org/docs/)

* Before installing Tensorflow you also have to install [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

* Open the location where SignTrack is downloaded on your terminal

* Run the command: __poetry&nbsp;install__

* ✨Now a virtualenv has been created containing the required dependances✨

### Linux & MacOS installation

* Install __Python&nbsp;3.7__ [Ubuntu&nbsp;](https://askubuntu.com/questions/1251318/how-do-you-install-python3-7-to-ubuntu-20-04)/[&nbsp;MacOS](https://www.python.org/downloads/macos/)

* Install Python [Poetry](https://python-poetry.org/docs/)

* Open the location where SignTrack is downloaded on your terminal

* Run the command: __poetry&nbsp;install__

* ✨Now a virtualenv has been created containing the required dependances✨

#### To run SignTrack( Applicable on all Operating Systems )

* Navigate to the SignTrack main script (signtrack/main)
* Open on terminal
* Type and run on terminal __poetry&nbsp;shell__
* Type and run on terminal __python&nbsp;main.py__

<br />

## Troubleshoting

### OpenCV errors

#### Try changing camera input

You can simply try to change the camera input selection in the first lines of code either on __DataCollect__ or __main__ file in the code.

* Find this line of code __cap&nbsp;=&nbsp;cv2.VideoCapture(0)__

* Change the default value zero to another one, like 1 or 2

#### Reinstall OpenCV

If that does not work try reinstalling OpenCV

* Type __poetry&nbsp;shell__ on your terminal on the SignTrack directory

* Then type __pip&nbsp;uninstall&nbsp;opencv-python__

* Finally type __pip&nbsp;install&nbsp;opencv-python==4.1.2.30__

<br />

### Failling to install a package

#### Try installing manually from the poetry env

* Type __poetry&nbsp;shell__ on your terminal on the SignTrack directory

* Type __pip&nbsp;install__ and continue with typing the failed to install package, making sure that it is the matching version

<br />

## Credits

Creating this amazing project would not have been possible without the help of:

* My mentor, Barbara Pongračić, for helping me focus on what is important.
* My friend, Fotini Tsiami, for assisting me throught the illustration designing proccess.
* DuckDuckGo and YouTube tutorials, well, for obvious reasons.
