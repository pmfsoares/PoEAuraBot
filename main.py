import cv2 as cv
import numpy as np
import os, sys, pdb;
from time import time
from windowcapture import WindowCapture
from vision import Vision
import torch, traceback, sys
import torch.nn.functional as F
from PIL import Image

# Change the working directory to the folder this script is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class PlayerDetection():

    def __init__(self):
    # initialize the WindowCapture class
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=False)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = self.model.names
        self.model.to(self.device)
        self.wc = WindowCapture()
        print(self.classes)
    """
    The function below identifies the device which is availabe to make the prediction and uses it to load and infer the frame. Once it has results it will extract the labels and cordinates(Along with scores) for each object detected in the frame.
    """
    def score_frame(self, frame):
        try:

            frame_re = frame.reshape((1, frame.shape[2], frame.shape[0], frame.shape[1]))
            frame_re = frame_re[:, :3, :, :]
            
            frame_t = torch.tensor(frame_re)
            results = self.model(frame_t)
            labels, cord  = results[0][:, -1], results[0][:, :-1]
#            pdb.set_trace()
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
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        print(frame.shape)
        bgr = (0, 255, 0) # color of the box
        classes = self.model.names # Get the name of label index
        label_font = cv.FONT_HERSHEY_SIMPLEX #Font for the label.
        for i in range(n):
            row = cord[i]
#            pdb.set_trace()
            # If score is less than 0.2 we avoid making a prediction.
            if row[4].item() < 0.2: 
                continue
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
 #           pdb.set_trace()
            im_arr = frame.astype(np.uint8)
            cv.rectangle(im_arr, (x1, y1), (x2, y2), bgr, 2) #Plot the boxes
            #cv.putText(im_arr, "teste", (x1, y1), label_font, 0.9, bgr, 2) #Put a label over box.
            try: 
                cv.putText(im_arr, classes[0], (x1, y1), label_font, 0.9, bgr, 2)
            except Exception:
                print("Label: " + str(i) + " . Not found.")
                #traceback.print_exc()
            return im_arr

    def __call__(self):
        loop_time = time()
        scPos = False
        scNeg = False
        sc_Time = time()
        self.wc.Start()
        while(True):

            #screenshot = WindowCapture.get_screenshot(self)
            screenshot = self.wc.GetScreen()

            #score the screenshot to see if there are any positive matches
            results = self.score_frame(screenshot)
            #draw a rectangle around the matches found
            frame = self.plot_boxes(results, screenshot)
            cv.imshow("Matches", frame)
            print('FPS {}'.format(1 / (time() - loop_time)))
            loop_time = time()
            
            # debug the loop r1
            key = cv.waitKey(1)
            now = time()
            
            if key == ord('q'):
                cv.destroyAllWindows()
                break
        
        print('Done.')


detection = PlayerDetection()
detection()
