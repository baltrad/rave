#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_xml.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2006-
#                All rights reserved.
#
# $Log: rave_xml.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_xml.py

Helpers for parsing and managing XML tags and attributes.
"""
import re
from rave_defines import ENCODING

class xmlmap:
    """
    This is a generic container object designed to hold contents of
    simple XML-based objects passed from ElementTree space. The contents of
    a single Element is mapped to an xmlmap instance. This is useful for
    managing configuration information from config-files. 
    """
    def __init__(self, E=None):
        if E:
            self.getArgs(E)

    def add(self, e):
        add(self, e)

    def getArgs(self, E):
        """
        Maps the contents of Element E to attributes of this object
        """
        for a in E.findall('arg'):
            self.add(a)

    def validate(self, attrs):
        for attr in attrs:
            if not hasattr(self, attr):
                raise AttributeError("object lacks mandatory attribute "+attr)


def add(o, e):
    """
    Adds the contents of Element e to Python object o.
    """
    i, t, text = e.get('id'), e.get('type'), e.text.encode(ENCODING)

    # If the attribute already exists, move it to a list so that
    # multiple variables can co-exist
    if hasattr(o, i):
        tmp = []
        tmp.append(getattr(o, i))
        setattr(o, i, tmp)
        append(o, i, t, text)
        return

    if t == 'int':
        setattr(o, i, int(text))
    elif t == 'float':
        setattr(o, i, float(text))
    elif t == 'sequence':
        text = re.sub(b" ", b"", text)  # weed out spacebars
        L = []
        for item in text.split(b','):
            L.append(eval(item))
        setattr(o, i, tuple(L))
    else:
        setattr(o, i, text)  # fallback: string


def append(o, i, t, text):
    """
    Appends the contents of Element e to Python object o.
    """
    tmp = getattr(o, i)
    if t == 'int':
        tmp.append(int(text))
    elif t == 'float':
        tmp.append(float(text))
    else:
        tmp.append(text)
    setattr(o, i, tmp)
    

def traverse_map(E, info, ID=''):
    """
    Maps the contents of Element object E to INFO object info.
    Does not map datasets!
    """
    import H5radHelper

    for e in E.getchildren():
        if e.getchildren():
            IDA = ID + "/"+ e.tag
            traverse_map(e, info, ID=IDA)
        else:
            IDA = ID + "/" + e.tag
            typ = e.get("type", "string")
            if typ == 'dataset': continue
            value = H5radHelper.geth5attr(e, {})
            info.put(IDA, value)



__all__ = ["xmlmap", "add", "append"]

if __name__ == "__main__":
    print(__doc__)
