#!/usr/bin/env python

import gi
import time
import board
import neopixel

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
        LED_Update(int(valr), int(valg), int(valb))


builder = Gtk.Builder()
builder.add_from_file("visicheck.glade")
builder.connect_signals(Handler())

window = builder.get_object("appwin")
window.show_all()

Gtk.main()    
    
    
