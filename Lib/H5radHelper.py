#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: H5radHelper.py,v 1.2 2006/12/18 09:34:16 dmichels Exp $
#
# Author: Daniel Michelson
#         (adapted from NORDRAD2)
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: H5radHelper.py,v $
# Revision 1.2  2006/12/18 09:34:16  dmichels
# *** empty log message ***
#
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""

"""
import string
from numpy import ndarray
from rave_defines import ENCODING
from struct import calcsize

# If a long is the same size as an int, ensure that longs are saved as llongs
if calcsize('i') == calcsize('l') == 4:
    LONG = 'llong'
else:
    LONG = 'long'


def typeconv(typ, val):
    if typ in ["int", "long", "llong"]:
        return int(val)
    elif typ in ["float", "double"]:
        return float(val)
    elif typ in ("string", "sequence", "dataset"):
        return val

def findelem(root, attr):
    path = attr.split("/")[1:]
    e = root
    for i in path:
        e = e.find(i)
        if e is None:
            break
    return e

def geth5attr(e, dictionary):
    typ, val = type_val(e)

    if typ == "dataset":
        dict_val = dictionary[val]
        if type(dict_val) is bytes:
          dict_val = dict_val.decode(ENCODING)
        return dict_val
    elif typ == "sequence":
        nodes = val.split(",")
        for n in range(len(nodes)):
            try:     # detects ints and floats but not strings
                nodes[n] = eval(nodes[n].strip()[1:-1])
            except:  # fallback to string
                nodes[n] = nodes[n].strip()[1:-1]
        return nodes
    else:
        return typeconv(typ, val)

def type_val(e):
    typ = e.get("type", "string")
    val = e.text
    if type(val) is bytes:
      val = val.decode(ENCODING)
    return typ, val

def h5type(value):
    if type(value) is int:
        return LONG
    elif type(value) is float:
        return "double"
    elif type(value) is str or type(value) is bytes:
        return "string"
    elif type(value) in [list, tuple]:
        for i in value:
            if type(i) not in [string, int, float, bytes]:
                return None
        return "sequence"
    elif type(value) is ndarray:
        return "dataset"
    else:
        return None


def seth5attr(e, attr_to_val_dict, h5typ, attribute, value):
    if h5typ is not "string":
        e.attrib["type"] = h5typ
    else:
        try:
            del e.attrib["type"]
        except KeyError:
            pass
    if h5typ is "sequence":
        # convert list to string
        nodes = []
        for n in value:
            node = str(n).strip()
            nodes.append(("'"+node+"'"))
        text = ", ".join(nodes)
    elif h5typ is "dataset":
        attr_to_val_dict[attribute] = value
        text = attribute
    elif h5typ in ["int", "long", "llong", "float", "double"]:
        text = str(value)
    elif h5typ is "string":
        text = value
    else:
        raise ValueError("Illegal type")
      
    e.text = text.encode(ENCODING)

def addelem(root, attribute):
    from xml.etree.ElementTree import SubElement

    path = attribute.split("/")[1:]
    e = root
#    result = None
    for i in path:
        new = e.find(i)
        if new is None:
            # add element as child to e
            new = SubElement(e, i)
        e = new
    return e



if __name__ is "__main__":
    print(__doc__)
