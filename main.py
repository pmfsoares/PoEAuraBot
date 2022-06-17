import cv2 as cv
import numpy as np
import os, sys
from time import time
from windowcapture import WindowCapture
from vision import Vision
import torch, traceback, sys
from PIL import Image


# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class PlayerDetection():

    def __init__(self):
    # initialize the WindowCapture class
        self.wincap = WindowCapture('Path of Exile')
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/runs/train/exp15/weights/best.pt', force_reload=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """
    def score_frame(self, frame):
        try:

            self.model.to(self.device)
            
            frame_t = torch.tensor(np.transpose(frame, (2, 1, 0)))
            frame_t = frame_t.unsqueeze(0)

            results = self.model(frame_t)

            labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
            print(results.pandas().xyxy[0])
            
            return labels, cord
        except Exception:
            traceback.print_exc()
            sys.exit("Error");

    """
    The function below takes the results and the frame as input and plots boxes over all the objects which have a score higer than our threshold.
    """
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[0], frame.shape[1]
        print(frame.shape)
        for i in range(n):
            row = cord[i]
            # If score is less than 0.2 we avoid making a prediction.
            if row[4] < 0.2: 
                continue
            x1 = int(row[0]*x_shape)
            y1 = int(row[1]*y_shape)
            x2 = int(row[2]*x_shape)
            y2 = int(row[3]*y_shape)
            bgr = (0, 255, 0) # color of the box
            classes = self.model.names # Get the name of label index
            label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.
            
            im_arr = frame.astype(np.uint8)
            cv.rectangle(im_arr, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
            #cv.putText(im_arr, "teste", (x1, y1), label_font, 0.9, bgr, 2) #Put a label over box.
            cv.putText(im_arr, classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr, 2)
            print(classes[int(labels[i])])
            return im_arr

    def __call__(self):
        # load the trained model
        cascade_player = cv.CascadeClassifier('cascade/cascade.xml')
        # load an empty Vision class
        cascade_vision = Vision(None)

        loop_time = time()
        scPos = False
        scNeg = False
        sc_Time = time()
        while(True):


            #Cascade Functions

            # get an updated image of the game
            screenshot = self.wincap.get_screenshot()

            #Comentar estas para usar o pytorch
            # do object detection
            rectangles = cascade_player.detectMultiScale(screenshot)
            # draw the detection results onto the original image
            detection_image = cascade_vision.draw_rectangles(screenshot, rectangles)
            cv.imshow('Matches', detection_image)



            #Para Pytorch descomentar funcoes a baixo e comentar as de 3 de cima
            #Pytorch Yolov5 functions
            #results = self.score_frame(screenshot)
            #frame = self.plot_boxes(results, screenshot)
            #cv.imshow("Matches", frame)

            print('FPS {}'.format(1 / (time() - loop_time)))
            loop_time = time()

            # display the images

            # debug the loop r1
            key = cv.waitKey(1)
            now = time()
            

            #Take a screenshot every X seconds
            #if(now - sc_Time >= 5):
            #   sc_Time = now
            #    if(scPos):
            #        print("Wrote Positive: positive/{}.jpg".format(loop_time))
            #        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
            #    if(scNeg):
            #        print("Wrote Negative: negative/{}.jpg".format(loop_time))
            #        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)
            
            if key == ord('q'):
                cv.destroyAllWindows()
                break
            elif key == ord('f'):
                print("Started positive capture")
                scPos = not scPos
            elif key == ord('d'):
                print("Started Negative capture")
                scNeg = not scNeg
        print('Done.')


detection = PlayerDetection()
detection()