#!/usr/bin/env python
'''
Copyright (C) 2012- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## rave_ql.py - A QuickLook viewer for 2-D NumPy arrays. Default palette is a
#               continuous color scale which starts at black and ends at white,
#               cycling through B,C,G,Y,R,M. If array's type is not 'B' (uint8) 
#               then the array is scaled and converted to 'B' before viewing.
#               This does not affect the original array.
#
#  Functionality:
#               Pressing the 'h' or 'H' keys in the window will lauch a help 
#               panel.
#               Pressing the 's' or 'S' keys in the window save the image to
#               graphics file.
#               Pressing the 'a' or 'A' keys in the window will show an About
#               popup.
#               Pressing the 'q' or 'Q' keys in the window will kill it.
#               Clicking the left mouse button will zoom in.
#               Clicking the right mouse button will zoom out.
#               Clicking the middle mouse button will restore the original size.

## @file
## @author Daniel Michelson, SMHI
## @date 2012-01-16

import sys
try:
    import pygtk
except ImportError:
    raise ImportError("This module requires PyGTK.")
pygtk.require('2.0')

try:
    import gtk
except ImportError:
    raise ImportError("This module requires GTK.")

try:
    import threading
except ImportError:
    raise ImportError("You need Python compiled with threading support for the QuickLook module.")

import os, StringIO
import numpy
import Image
import rave_win_colors
from rave_defines import RAVEICON

gtk.gdk.threads_init()

## Default palette
palette = rave_win_colors.continuous_dBZ

# These constants aren't sophisticated, but they should work anyway...
scrollbar_width = 20
max_win_dim = 720

# Global use of the one and only (so far) RAVE icon.
gtk.window_set_default_icon_from_file(RAVEICON)

## Converts an array to typecode 'B' and stretches its contents to fit in the
# interval 0-255.
# @param a array of almost any type (except complex)
# @param return a 'B' (uint8) array
def stretch(a):
    MIN = numpy.minimum.reduce(a.flat)
    if MIN < 0:
	a = a + abs(MIN)
    MIN = numpy.minimum.reduce(a.flat)
    MAX = numpy.maximum.reduce(a.flat)
    spread = float(MAX-MIN)
    a = ((1 - ((MAX - a) / spread)) * 255).astype('B')
    return a


## Converts a NumPy array to a PIL object of type P ("panchromatic")
# @param a array
# @return PIL object of type P
def array2pilp(a):
    i = Image.new('P', (a.shape[1], a.shape[0]))
    i.fromstring(a.tostring())
    return i


## Converts a PIL mode P Image to a PyGTK pixbuf through PNG
# @param image PIL type P image
# @return PyGTK pixbuf
def Image_to_GdkPixbuf(image):
    file = StringIO.StringIO()
    image.save(file, 'png')
    contents = file.getvalue()
    file.close()
    loader = gtk.gdk.PixbufLoader('png')
    loader.write(contents, len(contents))
    pixbuf = loader.get_pixbuf()
    loader.close()
    return pixbuf


## QuickLook object
class ql:
    ## Initializer
    # @param a NumPy array of almost any type (except complex)
    # @param pal PIL palette object
    # @param title string title for the window
    def __init__(self, a, pal=palette, title="QuickLook"):
	self.data = a  # the array
        self.scale_factor = 1  # for zooming & rescaling

	# linear scaling to 8-bits if necessary
	if self.data.dtype.char != 'B':
            self.data = stretch(self.data)
	    
        # Double conversion, first to PIL, then to a pixbuf
	self.pil = array2pilp(self.data)
	self.pil.putpalette(pal)
        self.xsize, self.ysize = self.pil.size[0], self.pil.size[1]

        self.pixbuf = Image_to_GdkPixbuf(self.pil)

        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title(title)
        self.window.connect("delete_event", self.quit_ql)
        self.window.connect("button_press_event", self.button_press_event)
        self.window.connect("key_press_event", self.key_press_event)
        self.window.set_border_width(0)
        self.window.show()

        # a horizontal box to hold the image
        self.hbox = gtk.HBox()
        self.hbox.show()
        self.window.add(self.hbox)

        self.image = gtk.Image()
        self.image.set_from_pixbuf(self.pixbuf)
#        self.image.show()

        self.swin = gtk.ScrolledWindow()
        self.hbox.pack_start(self.swin)
        self.swin.add_with_viewport(self.image)
        self.window.resize(min(self.xsize,max_win_dim)+scrollbar_width, 
                           min(self.ysize,max_win_dim)+scrollbar_width)

        self.window.show_all()


    ## Terminator. Invoked via signal delete_event.
    # @param widget object to terminate
    # @param event (never used...)
    # @param data (never used)
    # @return False boolean, always
    def quit_ql(self, widget, event, data=None):
        self.window.hide()
        self.destroy(widget)
        gtk.main_quit()
        return False


    ## Widget destroyer
    # @param widget the widget to destroy
    # @return False boolean, always
    def destroy(self, widget):
        gtk.main_quit()
        return False


    ## OK selector
    # @param w widget object
    def file_ok_sel(self, w):
#        self.filew.output_graphic = self.filew.get_filename()
        if self.response == gtk.RESPONSE_OK:
            self.filew.output_graphic = self.filew.get_filename()
            self.save_pil()
        self.filew.destroy()


    ## Formatter for title
    # @param msg_type type of message
    # @param title string message title
    # @param buttons button formatter
    def simple_message(self, msg_type, title, message, buttons=gtk.BUTTONS_OK):
        dialog = gtk.MessageDialog(type=msg_type,
                                   message_format=title,
                                   buttons=buttons)
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()


    ## Saves a PIL object to graphics file
    # @param filename string of the file name to save, extension will determine
    # the file format, e.g. .png
    def save_pil(self, filename=None):
        if self.filew.output_graphic and not filename:
            save_pil(self.pil, filename=self.filew.output_graphic)
        elif filename:
            save_pil(self.pil, filename=filename)


    ## File-select widget for saving graphics images
    def file_select(self):
        self.filew = gtk.FileChooserDialog(title="Save as...",
                                           action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                           buttons=(gtk.STOCK_CANCEL,
                                                    gtk.RESPONSE_CANCEL,
                                                    gtk.STOCK_SAVE,
                                                    gtk.RESPONSE_OK))
        self.filew.set_default_response(gtk.RESPONSE_OK)
        self.filew.set_do_overwrite_confirmation(True)

        self.filew.response = self.filew.run()
        if self.filew.response == gtk.RESPONSE_OK:
            self.filew.output_graphic = self.filew.get_filename()
            try:
                self.save_pil()
                self.filew.destroy()
            except:
                self.simple_message(gtk.MESSAGE_ERROR, "PIL Error",
                                   "Cannot save graphic. File name incompatible with PIL.")
                self.filew.destroy()
                self.file_select()
        else:
            self.filew.destroy()


    ## Simple widget containing About information
    def about_window(self):
        about = gtk.AboutDialog()
        about.set_name('About this software')
        about.set_program_name('BALTRAD Toolbox')
        about.set_authors(['Daniel Michelson','Anders Henja',
                           'Guenther Haase','Your Name Here'])
        about.set_comments('RAVE Product Generation Framework for BALTRAD')
        #about.set_version('Third generation')
        about.set_website_label('baltrad.eu')
        about.run()
        about.destroy()


    ## Manager of key pressings
    # @param widget the widget to manage
    # @param event the event to associate with the key pressing
    def key_press_event(self, widget, event): 
        if event.keyval in (gtk.keysyms.q, gtk.keysyms.Q):
            self.quit_ql(widget, event)
        elif event.keyval in (gtk.keysyms.s, gtk.keysyms.S):
            self.file_select()
        elif event.keyval in (gtk.keysyms.a, gtk.keysyms.A):
            self.about_window()
        elif event.keyval in (gtk.keysyms.h, gtk.keysyms.H):
            self.simple_message(gtk.MESSAGE_INFO, "Hot Keys and Buttons",
                                'q\tQuit the QuickLook viewer.\n\ns\tSave the image to graphics file.\n\nh\tThis help dialog.\n\na\tAbout this software.\n\nLeft mouse button\t\tZoom in.\n\nRight mouse button\tZoom out.\n\nMiddle mouse button\tRestore original image size.',
                                gtk.BUTTONS_CLOSE)


    ## Zoomer
    def zoom(self):
        if self.scale_factor != 1:
            if self.scale_factor < 1:
                scaled_pixbuf = self.pixbuf.scale_simple(int(self.xsize*self.scale_factor)+1, int(self.ysize*self.scale_factor)+1, int(gtk.gdk.INTERP_BILINEAR))
            else:
                scaled_pixbuf = self.pixbuf.scale_simple(self.xsize*self.scale_factor, self.ysize*self.scale_factor, gtk.gdk.INTERP_BILINEAR)
        else:
            scaled_pixbuf = self.pixbuf

        self.image.set_from_pixbuf(scaled_pixbuf)
        self.image.show()

        if self.scale_factor < 1:
            self.window.resize(min(int(self.xsize*self.scale_factor),
                                   max_win_dim)+1+scrollbar_width, 
                               min(int(self.ysize*self.scale_factor),
                                   max_win_dim)+1+scrollbar_width)
        else:
            self.scale_factor = int(round(self.scale_factor))
            self.window.resize(min(self.xsize*self.scale_factor,
                                   max_win_dim)+scrollbar_width, 
                               min(self.ysize*self.scale_factor,
                                   max_win_dim)+scrollbar_width)
        self.window.show_all()


    ## Button pressing manager
    # @param widget the widget to manage
    # @param event the event to associate with the button pressing
    def button_press_event(self, widget, event): 
        if event.button == 1:
            if self.scale_factor <= 1:
                self.scale_factor *= 2
            else:
                self.scale_factor += 1

        elif event.button == 2:
            self.scale_factor = 1

        elif event.button == 3:
            if self.scale_factor <= 1:
                self.scale_factor /= 2.0
            else:
                self.scale_factor -= 1
        self.zoom()


    ## Main method that fires up the GTK engine
    def main(self):
        gtk.main()



## Helper function to save PIL object as a graphics image
# @param pilobj PIL object
# @param filename string of the file name to save, where the extension will
# determine the file format, e.g. .png
def save_pil(pilobj, filename=None):
    try:
        pilobj.save(filename)
    except:
        raise IOError("PIL couldn't write a file of that name.")



__all__ = ['stretch','ql','array2pilp','Image_to_GdkPixbuf']

if __name__ == "__main__":
    import rave

    this = rave.open(sys.argv[1])
    that = ql(this.get('/dataset1/data1/data'))
    that.main()
