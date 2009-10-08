#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_tempfile.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: rave_tempfile.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_tempfile.py - Re-defines the template of tempfiles for RAVE.
"""
import os, tempfile
import rave, rave_defines

tempfile.tempdir = os.path.split(rave.__file__)[0] + '/../tmp'
if os.path.isdir(tempfile.tempdir) is False:
    os.makedirs(tempfile.tempdir)

RAVETEMP = tempfile.tempdir


def mktemp():
    """
    Temporary file management in RAVE redefines the template in the tempfile
    module to write files in $RAVEROOT/tmp with a prefix 'rave-<pid>'.
    
    Arguments: None

    Returns:
      a tuple containing
        int: OS-level handle to an open file as would be returned by os.open()
             This can be closed with os.close().
        string: the absolute pathname of the file.

    NOTE: the file is created and opened. In order to prevent too many open
          files, you may have to close this file before continuing.
    """
    PREFIX = "rave%d-" % os.getpid()
    return tempfile.mkstemp(prefix=PREFIX)



if __name__ == "__main__":
    print __doc__
