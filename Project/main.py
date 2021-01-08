import cv2 as cv
import numpy as np

def showWebcam(model):
    """ 
    Draws a rectangle based on objects detected using the given model.

    Parameters
    ----------
    model <CascadeClassifier object>: an xml Cascade model
    """ 

    cap = cv.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        detectUsingModel(model,frame)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def detectUsingModel(model,img):
    """
    Returns an image with rectangular borders representing the
    detected objects in an xml model.

    Parameters
    ----------
    model <CascadeClassifier object>: an xml Cascade model img: opencv
    image object
    """


    rectangles = model.detectMultiScale(img)

    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    for (x, y, w, h) in rectangles:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv.rectangle(img, top_left, bottom_right, line_color, lineType=line_type)
    
    return img
    
if __name__ == "__main__":

    # OpenCv Eyes Cascade Model
    # model = cv.CascadeClassifier('cvModels/haarcascade_frontalface_default.xml')

    # OpenCv Frontal Face Cascade Model
    # model = cv.CascadeClassifier('cvModels/haarcascade_eye.xml')


    # My Model
    model = cv.CascadeClassifier('cascade/cascade.xml')

    # From Webcam Stream
    showWebcam(model)

    # From an image
    photo = cv.imread('Dataset/testing/face2.jpeg')
    # photo = cv.imread('Dataset/faces/pic00088.jpeg')

    photo = detectUsingModel(model, photo)

    cv.imshow('photo',photo)
    cv.waitKey(0)
    cv.destroyAllWindows()