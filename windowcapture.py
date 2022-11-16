import numpy 
import cv2
import mss
import mss.tools
import time
from threading import Thread, Lock

class WindowCapture:

    def __init__(self):
        self.mut = Lock()
        self.hwnd = None
        self.its = None        #Time stamp of last image
        self.i0 = None   #i0 is the latest image;
        self.i1 = None   # i1 is used as a temporary var
        self.cl = False  #Continue looping flag

    def GetScreen(self):
        while self.i0 is None:
            pass
        self.mut.acquire()
        s = self.i0
        self.mut.release()
        return s

    def Start(self):

        self.cl = True
        thrd = Thread(target = self.ScreenUpdateT)
        thrd.start()
        return True
    def Stop(self):

        self.cl = False

    def ScreenUpdateT(self):

        while self.cl:
            self.i1 = self.get_screenshot()
            self.mut.acquire()
            self.i0 = self.i1
            self.its = time.time()
            self.mut.release()

    def get_screenshot(self):
        # get the window image data
        with mss.mss() as sct:
            monitor_number = 2
            mon = sct.monitors[monitor_number]
            monitor = {
                    "top"   : 27,
                    "left"  : 1920,
                    "width" : 800,
                    "height": 640
            }
            img = numpy.array(sct.grab(monitor))
            return img#.reshape((1, img.shape[2], img.shape[0], img.shape[1]))#[: ,:,:3]
