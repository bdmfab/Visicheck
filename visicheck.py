#!/usr/bin/env python

import gi
import time
import board
import neopixel


gi.require_version('Gdk', '3.0')

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

def Cam_Start():
    print("Cam_Start")

def LED_Update(R=0, G=0, B=0 ): 
    #print("R-" + str(R) + " G-" + str(G) + " B-" + str(B))
    board.pin18
    
class Handler:
    def onDestroy(self, *args):
        Gtk.main_quit()

    def startFeed(self, button):       
        Cam_Start()

    def colorChange(self, widget):
        valr = builder.get_object("tbR").get_text()
        #print("Red = " + valr)
        valg = builder.get_object("tbG").get_text()
        #print("Green = " + valg)   
        valb = builder.get_object("tbB").get_text()
        #print("Blue = " + valb)
        #LED_Update(int(valr), int(valg), int(valb))
        fR = int(valr)/255
        fG = int(valg)/255
        fB = int(valb)/255
        cC = builder.get_object("colorChooser")        
        cC.set_rgba(Gdk.RGBA(fR, fG, fB, 1)) 

    def onColor(self, widget):
        vals = widget.get_rgba() 
        r = int(vals.red * 255)
        b = int(vals.blue * 255)    
        g = int(vals.green * 255)
        #print("Red = " + str(r) + " Green = " + str(g) + " Blue = " + str(b))
        builder.get_object("tbR").set_text(str(r))
        builder.get_object("tbG").set_text(str(g))
        builder.get_object("tbB").set_text(str(b))


builder = Gtk.Builder()
builder.add_from_file("visicheck.glade")
builder.connect_signals(Handler())

window = builder.get_object("appwin")
window.show_all()

Gtk.main()    
    
    
