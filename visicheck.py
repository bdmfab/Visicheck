#!/usr/bin/env python

import gi
import time
import board
import neopixel
import numpy as np
import cv2
import math, copy, time, logging, sys

gi.require_version('Gdk', '3.0')
gi.require_version("Gtk", "3.0")

from gi.repository import Gtk, Gdk


def connectBtn(self):
    if (self.feedLive == False): 
            iniCam(self)           
            self.feedLive = True
            builder.get_object("lblStatus").set_text("Feed is live!")
            
            while self.feedLive == True:
                ret, frame = self.cap.read()
                if ret:                       
                    cv2.imshow('LiveFeed', frame)  

                k = cv2.waitKey(30)
                if k == 27: #esc
                    stopfeed(self) 
                
                if (cv2.getWindowProperty('LiveFeed', 1) < 1):
                    stopfeed(self)        
                                     
                if self.saveRef == True:
                    self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(self.imgname, self.gray)
                    startSketch(self)                    
                            
    else:
        stopfeed(self)

def captureFrame(self):
    iniCam(self)
    ret, frame = self.cap.read()
    if ret:
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(self.imgname, self.gray)
        startSketch(self)

def startSketch(self):
    #Stop live video
    stopfeed(self)                    
    self.feedLive = False  
    self.saveRef = False 
    #Setup Reference Image
    self.img = cv2.imread(self.imgname, 1) 
    self.orig = copy.deepcopy(self.img)
    cv2.namedWindow(self.imgname)  
    cv2.setMouseCallback(self.imgname, self.draw_shape)
    refreshimg(self)

def stopfeed(self):
        self.feedLive = False
        self.cap.release()
        cv2.destroyAllWindows() 
        builder.get_object("lblStatus").set_text("-")   

def iniCam(self):
    print("Cam_Start")
    self.cap = cv2.VideoCapture(0)
    self.feedLive = True

def refreshimg(self):
        while True:
            cv2.imshow(self.imgname, self.img)
            k = cv2.waitKey(5)
            if (k == 27):
                cv2.destroyAllWindows()
                cv2.setMouseCallback(self.imgname, lambda *args : None)
                break
            if self.ref_avail == True:
                response = messageBox("Attention", "Save Inspection?")
                if response == Gtk.ResponseType.YES:
                    builder.get_object("lblStatus").set_text(self.inspectionname + " - Saved!")
                    cv2.imwrite(self.inspectionname, self.crop_image)
                    cv2.setMouseCallback(self.imgname, lambda *args : None) 
                    cv2.destroyAllWindows()
                    self.ref_avail = False
                    break
                    
                else:
                    builder.get_object("lblStatus").set_text("Try Again!")
                    self.img = cv2.imread(self.imgname, 0) 
                    self.orig = copy.deepcopy(self.img) 
                    self.ref_avail = False 
                    cv2.destroyWindow("Inspection Area")                              

def messageBox(text1,text2 = ""):
    dialog = Gtk.MessageDialog(
                    #transient_for = self,                    
                    flags = 0,
                    message_type = Gtk.MessageType.QUESTION,
                    buttons = Gtk.ButtonsType.YES_NO,
                    text = text1,
                )
    dialog.format_secondary_text(
        text2
    )
    response = dialog.run()
    dialog.destroy()                
    return response

def getFile():
    dialog = Gtk.FileChooserDialog(
        title = "Please choose a file",
        action = Gtk.FileChooserAction.OPEN
    )
    dialog.add_buttons(
        Gtk.STOCK_CANCEL,
        Gtk.ResponseType.CANCEL,
        Gtk.STOCK_OPEN,
        Gtk.ResponseType.OK
    )

    response = dialog.run()
    if (response == Gtk.ResponseType.OK):
        r = dialog.get_filename()
        dialog.destroy()
        return r
    else:
        r = "None"
        dialog.destroy()
        return r

def LED_Update(R=0, G=0, B=0 ): 
    if (builder.get_object("tbNPixels").get_text() == ""):
        nP_s = "7"
    else:
        nP_s = builder.get_object("tbNPixels").get_text()

    nP_i = int(nP_s)
    
    print("R-" + str(R) + " G-" + str(G) + " B-" + str(B))
    pixels = neopixel.NeoPixel(board.D21, nP_i)
    for x in range(7):
        pixels[x] = (R, G, B)             

def templateMatch(self, source):
        methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
        types = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCORR",
            "cv2.TM_CCORR_NORMED", "cv2.TM_SQDIFF", "cv2.TM_SQDIFF_NORMED"]
        #captureFrame(self)
        lf = cv2.imread(self.inspectionname)

        if source != False:
            builder.get_object("lblStatus").set_text("Press any key to continue...")
            cv2.waitKey(0)            
            cv2.imshow("Looking For", lf)
            builder.get_object("lblStatus").set_text("-")

        loop = 0
        
        print("")
        print("- - - - - - - - - - - - -")
        for method in methods:
            img2 = cv2.imread(self.imgname, 0)
            #img2 = self.img2check.copy()
            template = cv2.imread(self.inspectionname, 0)
            h,w = template.shape
            result = cv2.matchTemplate(img2, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                location = min_loc
            else:
                location = max_loc

            type = str(types[method]) + " - " + str(method)
            print(type) 
            print("min = " + str(min_val))
            print("max = " + str(max_val))
            print("- - - - - - - - - - - - -")
            loop = loop + 1
            bottom_right = (location[0] + w, location[1] + h)    
            cv2.rectangle(img2, location, bottom_right, 0, 2)
            s = type + " >> " + str(min_val) + " to " + str(max_val)
            cv2.imshow(s, img2)
            builder.get_object("lblStatus").set_text("Press any key to continue... " + str(loop) + " of 6")
            k = -1
            while k == -1:
                k = cv2.waitKey(100)

            #cv2.destroyAllWindows()  
                       
        builder.get_object("lblStatus").set_text(" All inspections complete")

def blobdetect(self):
        filename = getFile()
        if (filename == "None"):
            return

        val1 = int(builder.get_object("sclBlobLL").get_value())
        val2 = int(builder.get_object("sclBlobHL").get_value())    
        
        print(filename)
        lf = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        cv2.imshow("Looking For", lf) 
        params = cv2.SimpleBlobDetector_Params() 
        params.minThreshold = val1   #was 50
        params.maxThreshold = val2   #was 200
        params.filterByArea = True
        params.minArea = 100
        params.filterByCircularity = True
        params.minCircularity = .8
        params.maxCircularity = 1
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(lf) 
        y_pos = int(keypoints[0].pt[0])
        print("X = " + str(y_pos)) 
        x_pos = int(keypoints[0].pt[1])
        print("Y = " + str(x_pos))
        print("Size = " + str(keypoints[0].size))
        blobs = cv2.drawKeypoints(lf, keypoints, np.array([]), (0, 100, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT) 
        cv2.imshow("Blobs", blobs)
        mask = np.zeros(lf.shape[:2], dtype="uint8")
        cv2.circle(mask, (y_pos, x_pos), 200, (255,255,255), -1)
        #cv2.imshow("Mask", mask)
        masked = cv2.bitwise_and(lf, lf, mask=mask)
        cv2.imshow("Masked", masked)
        cv2.imwrite("masked.png", masked)        
        k = -1
        while k == -1 :
            k = cv2.waitKey(100) 
        cv2.destroyAllWindows()

def threshold(self):
        lf = cv2.imread(self.inspectionname)
        lf = cv2.cvtColor(lf, cv2.COLOR_RGB2GRAY)
        cv2.imshow("Orig", lf)
        thresh1 = int(builder.get_object("threshLL").get_value())
        thresh2 = int(builder.get_object("threshHL").get_value())
        ret, thresh = cv2.threshold(lf, thresh1, thresh2, cv2.THRESH_BINARY)
        h, w = lf.shape
        print("Height & Width : ", h, w)
        size = lf.size
        print("Size of image in pixels ", size)
        count = cv2.countNonZero(thresh)
        print("Non zero count = ", str(count))
        print("Black Pixels = ", str(size-count))
        cv2.imshow("Thresh", thresh)
        k = -1
        while k == -1:
            k = cv2.waitKey(100)
        cv2.destroyAllWindows()

def edgeDetect(self):
        filename = getFile()
        print(filename)
        if (filename == "None"):
            return
        
        img = cv2.imread(filename)
        cv2.imshow("Image", img)
        k = int(builder.get_object("edgeK").get_value())
        if builder.get_object("edgeGausian").get_active():
            img = cv2.GaussianBlur(img, (k,k), 0)
            cv2.imshow("Blurred", img)
                        
        thresh1 = int(builder.get_object("edgeLL").get_value())
        thresh2 = int(builder.get_object("edgeHL").get_value())
        cedges = cv2.Canny(image=img, threshold1=thresh1, threshold2=thresh2 )
        cv2.imshow("Canny", cedges)
        sedges = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=k)
        cv2.imshow("Sobel", sedges)
        builder.get_object("lblStatus").set_text("C = Canny  S = Sobel  B = Blur")
        ks = -1
        while ks == -1:
            ks = cv2.waitKey(100)
            if ks == ord("c"):    
                cv2.imwrite("canny.png", cedges)
            if ks == ord("s"):
                cv2.imwrite("sobel.png", sedges)
            if ks == ord("b"):
                cv2.imwrite("blur.png", img)
        cv2.destroyAllWindows()
        builder.get_object("lblStatus").set_text("-")

def orb(self):
        filename = getFile()
        print(filename)
        if (filename == "None"):
            return
        img = cv2.imread(filename, 0)
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        cv2.imshow("Keypoints", img2)

class Handler:

    def __init__(self):
        self.ix = -1
        self.iy = -1
        self.feedLive = False
        self.drawing = False
        self.cap = []
        self.orig = []
        self.crop_image = []
        self.img = []
        self.img2check = []
        self.feedLive = False
        self.drawing = False
        self.mode = True
        self.imgname = "Captured_Frame.png"
        self.inspectionname = "Inspection Area.png"
        self.curinspect = "Current Inspection.png"
        self.ref_avail = False
        self.saveRef = False

    def onDestroy(self, *args):
        Gtk.main_quit()

    def startFeed(self, button):       
        connectBtn(self)

    # Updates color to color picker
    def colorChange(self, widget):
        valr = builder.get_object("tbR").get_text()
        #print("Red = " + valr)
        valg = builder.get_object("tbG").get_text()
        #print("Green = " + valg)   
        valb = builder.get_object("tbB").get_text()
        #print("Blue = " + valb)
        LED_Update(int(valr), int(valg), int(valb))
        fR = int(valr)/255
        fG = int(valg)/255
        fB = int(valb)/255
        cC = builder.get_object("cp1")        
        cC.set_rgba(Gdk.RGBA(fR, fG, fB, 1)) 

    # updates and converts vales in fields from color picker
    def onColor(self, widget):
        vals = widget.get_rgba() 
        r = int(vals.red * 255)
        b = int(vals.blue * 255)    
        g = int(vals.green * 255)
        #print("Red = " + str(r) + " Green = " + str(g) + " Blue = " + str(b))
        builder.get_object("tbR").set_text(str(r))
        builder.get_object("tbG").set_text(str(g))
        builder.get_object("tbB").set_text(str(b))

    def onCapture(self, widget):
        captureFrame(self)

    def onSaveFrame(self, widget):
        if (self.feedLive == True):
            self.saveRef = True
        else:
            builder.get_object("lblStatus").set_text("No Camera Feed!")

# mouse callback function
    def draw_shape(self,event,x,y,flags,params):        
        
        #print("Mouse Event = " + str(event))
        #print("X = " + str(x))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x,y
            builder.get_object("lblStatus").set_text("Drawing....")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                self.img = copy.deepcopy(self.orig) #overwrite the frame to eliminate archiving
                if self.mode == True:                
                    cv2.rectangle(self.img,(self.ix,self.iy),(x,y),(100,100,100),1)
                else:
                    xd,yd = self.ix - x, self.iy - y
                    rad = int(math.sqrt((xd*xd) + (yd*yd)))
                    cv2.circle(self.img,(self.ix,self.iy),rad,(100,100,100),1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == True: #drawing a rectangle
                cv2.rectangle(self.img,(self.ix,self.iy),(x,y),(255,255,255),2)
                self.orig = copy.deepcopy(self.img)            

                if self.iy > y:
                    y1 = y
                    y2 = self.iy
                else:
                    y1 = self.iy
                    y2 = y 

                if self.ix > x:
                    x1 = x
                    x2 = self.ix
                else:
                    x1 = self.ix
                    x2 = x  

            self.crop_image = self.img[y1:y2, x1:x2]
            self.crop_image = cv2.cvtColor(self.crop_image, cv2.COLOR_RGB2GRAY)
            cv2.imshow("Inspection Area", self.crop_image)
            self.ref_avail = True
            builder.get_object("lblStatus").set_text("-")
                                   
        else: #drawing a circle
            xd,yd = self.ix - x, self.iy - y           
            rad = int(math.sqrt((xd*xd) + (yd*yd)))            
            cv2.circle(self.img,(self.ix,self.iy),rad,(200,200,200),2)
            orig = copy.deepcopy(self.img)       
# end mouse callback  
    
    def onTemplateMatch(self, widget):
        source = True
        templateMatch(self, source)

    def onBlob(self, widget):
        blobdetect(self)

    def onThresh(self, widget):
        threshold(self)

    def onEdge(self, widget):
        edgeDetect(self)

    def onOrb(self, widget):
        orb(self)

    def onScaleTest(self, widget):
        val1 = int(builder.get_object("sclTempLL").get_value())
        val2 = int(builder.get_object("sclTempHL").get_value())
        print("Val1 = " + str(val1))
        print("Val2 = " + str(val2))


builder = Gtk.Builder()
builder.add_from_file("visicheck.glade")
builder.connect_signals(Handler())

window = builder.get_object("appwin")
window.show_all()

Gtk.main()    
    
    
