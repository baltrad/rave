#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_info.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: rave_info.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_info.py - Metadata structures and ways to deal with them.
"""
# Standard python libs:
import sys
import os
import types
# import cElementTree as ElementTree
from xml.etree.ElementTree import Element
# from xml.etree.ElementTree import _ElementInterface, ElementTree


# Module/Project:
import rave_IO, rave_h5rad
import H5radHelper
from rave_defines import *

# Stupid constants
TYPES = {
    tuple: 'sequence',
    list : 'sequence',
    str  : 'string',
    int  : 'int',
    float: 'float'
}


class INFO(Element):
    """
    Fundamental object for managing metadata.
    Based almost entirely on Fredrik Lundh's ElementTree project.
    http://effbot.org/zone/element-index.htm

    Arguments:

    Returns:
    """
    
    def __init__(self, tag="h5rad", attrib={}, **extra):
        """
        Initializes an INFO object.

        Arguments:

        Returns:
        """
        self.tag = tag
        self.attrib = attrib.copy()
        self.attrib.update(extra)
        self._children = []
    
    def __repr__(self):
        """Internal method; don't bother asking..."""
        return "<INFO object at %x>" % id(self)
    
    def eval(self, path=None):
        """
        Evaluates the attribute given by "path" and returns it as its proper
        type. This method queries the "type" attribute of the given Element
        and uses it to cast the Element's text properly.
        Opposite of the put() method.

        Arguments:
          string path: the attribute's path. If this path doesn't exist in this
                       INFO object, then an AttributeError exception is raised.

        Returns: a Python int, float, string, or sequence, depending on
                 the contents of the given attribute.
        """
        path = CheckPath(path)
        this = self.find(path)
        if this is None:
            raise AttributeError("No such attribute: %s" % path)
        else:
            t = this.get('type')
            if t == 'int':
                return int(this.text)
            elif t == 'float':
                return float(this.text)
            elif t == 'sequence':
                return eval(this.text)  # This might be tricky...
            elif t in ['dataset', None]:
                return str(this.text).encode(ENCODING)
            else:
                raise TypeError('Unknown type "%s"' % t)
    
    def parse(self, filename):
        """
        Reads the contents XML file and adds them to this instance.

        Arguments:
          string filename: the name of the file to parse.

          Returns: Nothing. NOTE that this method can only read XML
                            written by asXML() or pureXML().
        """
        from xml.etree import ElementTree
        import rave_xml
        
        E = ElementTree.ElementTree()
        E = E.parse(filename)
        
        rave_xml.traverse_map(E, self)
    
    def put(self, path, value):
        """
        Creates and sets the value of a new infoset element.

        Arguments:
          string path: This is given in the infoset format, ie. "/path/name".
                       If the payload already exists, it will be modified with
                       the new value. If not, it will be created. Groups are
                       created automatically.

          value: The value to set. This must be an int, float, string, or list
                 of strings. numpy arrays will just be recycled...

        Returns: Nothing if successful, a ValueError exception if the data
                 type is invalid or the payload is illegal.
        """
        from H5radHelper import h5type, findelem, seth5attr, addelem
        from xml.etree.ElementTree import Element
        
        h5typ = h5type(value)
        if not h5typ:
            raise ValueError("Unsupported type %s" % type(value))
        if self is None:
            self = Element("h5rad", version=H5RAD_VERSION)
        e = findelem(self, path)
        if e is None:
            e = addelem(self, path)
        seth5attr(e, {}, h5typ, path, value)
    
    def delete(self, path):
        """
        Deletes infoset element specified by "path".

        Arguments:
        string path: The element to delete, given in the infoset format, ie.
                     "/path/name".

        Returns: Nothing if successful, otherwise exceptions cast by
                 the ElementTree module if the data type is invalid or the
                 payload is illegal.
        """
        path = CheckPath(path)
        depth = path.split('/')
        if len(depth) == 1:
            self.remove(self.find(path))
        else:
            parentpath, childpath = os.path.split(path)
            parent = self.find(parentpath)
            parent.remove(parent.find(childpath))
    
    def asXML(self, file=None, encoding=ENCODING):
        """
        Outputs a metadata representation of an INFO object to XML
        with line breaks, either to file or to stdout.

        Arguments:

        Returns:
        """
        if file is None:
            rave_IO.prettyprint(self)
        else:
            import __builtin__
            
            fd = __builtin__.open(file, 'w')
            sys.stdout = fd
            print("<?xml version='1.0' encoding='%s'?>" % encoding)
            rave_IO.prettyprint(self)
            fd.close()
            sys.stdout = sys.__stdout__
    
    def pureXML(self, file, encoding=ENCODING):
        """
        Outputs a metadata representation of an INFO object to XML
        without line breaks, either to file or to stdout.

        Arguments:

        Returns:
        """
        e = ElementTree(self)
        e.write(file=file, encoding=encoding)
    
    def addDataset(self, prefix=None, set=None, **args):
        """
        Adds a dataset Group to this INFO.

        Arguments:

        Returns:
        """
        prefix = CheckPath(prefix)
        if set is None:
            raise AttributeError("Need number for this dataset.")
        
        else:
            newset = rave_h5rad.DatasetGroup(prefix=prefix, set=set, **args)
            self.append(newset)
            sets = self.eval('/what/sets')
            sets += 1
            self.put('/what/sets', sets)
    
    def CopyDatasetAttributes(self, info, ipath='', oset=1):
        """
        Copies the contents of "self" to output "info" object, optionally
        starting at "ipath". The output dataset attributes to write are given
        by the "oset" argument. This will only work for non-recursive Elements,
        ie. not for a complete INFO object. Suitable for copying the contents
        of e.g. /imageN/what

        Arguments:
        INFO object: object to which attributes are copied
        string ipath: absolute path to the element from which to copy
                      attributes
        int oset: index of the dataset, starting at 1, to which to write

        Returns: nothing
        """
        opath, otag = os.path.split(ipath)
        opath = "%s%i" % (opath[:-1], oset)
        
        for e in self.find(ipath[1:]).getiterator():
            if e.tag != otag:
                path = os.path.join(opath, otag)
                path = os.path.join(path, e.tag)
                t = e.get('type')
                if t:
                    info.put(path, eval(e.text.__str__()))
                else:
                    info.put(path, e.text.__str__())


def CheckPath(path):
    """
    Check that the path to an element is correctly formatted.

    Arguments:
    string path: The absolute path to the given infoset attribute.

    Returns: The relative path to the given infoset attribute.
    """
    if path is None:
        raise IOError("Given path is None.")
    elif len(path) == 0:
        raise IOError("Zero-length path given.")
    elif path[0] != '/':
        raise SyntaxError("Non-absolute path to element: %s\nAdd leading slash." % path)
    else:
        return path[1:]


__all__ = ['INFO', 'CheckPath']

if __name__ == "__main__":
    print(__doc__)
