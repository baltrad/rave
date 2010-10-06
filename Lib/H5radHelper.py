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
import sys, string
from types import IntType, FloatType, StringType, ListType, TupleType, LongType, UnicodeType
#from numpy import ArrayType
from numpy import ndarray
from rave_defines import ENCODING, UTF8
from struct import calcsize

# If a long is the same size as an int, ensure that longs are saved as llongs
if calcsize('i') == calcsize('l') == 4:
    LONG = 'llong'
else:
    LONG = 'long'


def typeconv(typ, val):
    if typ in ["int", "long", "llong"]:
        return string.atoi(val)
    elif typ in ["float", "double"]:
        return string.atof(val)
    if typ in ("ustring", "sequence", "dataset"):
        return val.encode(UTF8)
    elif typ == "string":
        if type(val) is UnicodeType:
            return val.encode(UTF8)
        else:
            return val.decode(ENCODING)

def findelem(root, attr):
    path = string.split(attr, "/")[1:]
    e = root
    result = None
    for i in path:
        e = e.find(i)
        if e is None:
            break
    else:
        result = e
    return e

def geth5attr(e, dict):
    typ, val = type_val(e)

    if typ == "dataset":
        return dict[val]
    elif typ == "sequence":
        nodes = string.split(val, ",")
        for n in range(len(nodes)):
            try:     # detects ints and floats but not strings
                nodes[n] = eval(string.strip(nodes[n])[1:-1].encode(ENCODING))
            except:  # fallback to string
                nodes[n] = string.strip(nodes[n])[1:-1].encode(ENCODING)
        return nodes
    else:
        return typeconv(typ, val)

# Does this always return typ="string"?
def type_val(e):
    typ = e.get("type", "string")
    val = e.text
    return typ, val

def h5type(value):
    if type(value) is IntType:
        return "int"
    elif type(value) is LongType:
        return LONG
    elif type(value) is FloatType:
        return "double"
    elif type(value) is StringType:
        return "string"
    elif type(value) is UnicodeType:
        return "ustring"
    elif type(value) in [ListType, TupleType]:
        for i in value:
            if type(i) not in [StringType, UnicodeType,
                               IntType, FloatType, LongType]:
                return None
        return "sequence"
    elif type(value) is ndarray:
        return "dataset"
    else:
        return None


def seth5attr(e, dict, h5typ, attribute, value):
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
            node = string.strip(str(n))
            nodes.append(("'"+node+"'"))
        e.text = str(string.join(nodes, ", "))
        #e.text = unicode(str(string.join(nodes, ", ")), ENCODING)
    elif h5typ is "dataset":
#        label = string.split(attribute, '/')[1]  # no heirarchy
#        dict[label] = value                      # no heirarchy
#        e.text = label                           # no heirarchy
        dict[attribute] = value
        e.text = attribute
        #e.text = unicode(attribute, ENCODING)
    elif h5typ in ["int", "long", "llong", "float", "double"]:
        e.text = str(value)
        #e.text = unicode(str(value), ENCODING)
    elif h5typ in ["string", "ustring"]:
        e.text = value
        #e.text = unicode(value, ENCODING)
    else:
        raise ValueError, "Illegal type"

def addelem(root, attribute):
    from xml.etree.ElementTree import SubElement

    path = string.split(attribute, "/")[1:]
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
    print __doc__
