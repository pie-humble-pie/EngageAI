import cv2
import numpy as np
from time import sleep
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing.image import img_to_array
import imutils

# Model used
train_model =  "ResNet"
img_width, img_height = 197, 197
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


model = load_model('models/ResNet-50.h5')
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')
distract_model = load_model('models/distraction_model.hdf5', compile=False)


frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5


video_capture = cv2.VideoCapture(0)

def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))  # Resizing images for the trained model
    ret = np.empty((img_height, img_width, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis = 0)
    x -= 128.8006   # np.mean(train_dataset)
    x /= 64.6497    # np.std(train_dataset)
    return x


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
    else:
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=frame_w)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
        	# image:		Matrix of the type CV_8U containing an image where objects are detected
        	# scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
        	# minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
        	# minSize:		Minimum possible object size. Objects smaller than that are ignored

        faces = faceCascade.detectMultiScale(
            gray_frame,
            scaleFactor		= scale_factor,
            minNeighbors	= min_neighbours,
            minSize			= (min_size_h, min_size_w))

        prediction = None
        x, y = None, None

        for (x, y, w, h) in faces:

            ROI_gray = gray_frame[y:y+h, x:x+w]
            ROI_color = frame[y:y+h, x:x+w]
            # Draws a simple, thick, or filled up-right rectangle
                # img:          Image
                # pt1:          Vertex of the rectangle
                # pt2:          Vertex of the rectangle opposite to pt1
                # rec:          Alternative specification of the drawn rectangle
                # color:        Rectangle color or brightness (BGR)
                # thickness:    Thickness of lines that make up the rectangle. Negative values, like CV_FILLED ,
                #               mean that the function has to draw a filled rectangle
                # lineType:     Type of the line
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            emotion = preprocess_input(ROI_gray)
            prediction = model.predict(emotion)
            print(emotions[np.argmax(prediction)] + " predicted with accuracy " + str(max(prediction[0])))
            top = emotions[np.argmax(prediction)]

            eyes = eye_cascade.detectMultiScale(ROI_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))


            probs = list()

            # loop through detected eyes
            for (ex,ey,ew,eh) in eyes:
                # draw eye rectangles
                cv2.rectangle(ROI_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                # get colour eye for distraction detection
                roi = ROI_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
                # match CNN input shape
                roi = cv2.resize(roi, (64, 64))
                # normalize (as done in model training)
                roi = roi.astype("float") / 255.0
                # change to array
                roi = img_to_array(roi)
                # correct shape
                roi = np.expand_dims(roi, axis=0)

                # distraction classification/detection
                pred = distract_model.predict(roi)
                # save eye result
                probs.append(pred[0])

            # get average score for all eyes
            probs_mean = np.mean(probs)

            # get label
            if probs_mean <= 0.5:
                label = 'distracted'
            else:
                label = 'focused'

            text = top + ' + ' + label
            cv2.putText(frame, text, (x, y+(h+50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
