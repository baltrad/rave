#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: rave.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave.py

Contains initialization functions for RAVE and fundamental class definitions,
along with methods for general product generation.
"""
import sys, os

import rave_IO, rave_info, rave_transform
import area
import _pyhl
from rave_defines import *

class RAVE:
    """
    Fundamental RAVE class
    
    Arguments:

    Returns:
    """
    
    def __init__(self, filename=None, **args):
        """
        Initializes a RAVE object.
        
        Arguments:

        Returns:
        """        
        self._fcp = None  # HDF5 file-creation properties

        if filename is not None:
            self.open(filename)

        else:
            import rave_info
            
            self.info = rave_info.INFO("h5rad", version=H5RAD_VERSION)
            self.data = {}
            self._h5nodes = None  # Only used when reading from HDF5 file
            self.__file__ = None  # Set when reading or writing HDF5 file

            if len(args.keys()):
                self.new(args)


    def __repr__(self):
        """Internal method; don't bother asking..."""
        return "<RAVE object at %x>" % id(self)


    def new(self, args):
        """
        """
        from rave_h5rad import TopLevelWhat, Where

        radarid = areaid = None
        sets = 0         # Defaults which should be overridden
        typecode = 'B'   # where appropriate.
        nodata = 255.0
##        undetect = 0
##        gain = 1.0
##        offset = 0.0

        if len(args) > 0:
            for k, i in args.items():
                if k == 'radar':
                    radarid = i
                elif k == 'area':
                    areaid = i
                elif k == 'sets':
                    sets = i
                elif k == 'typecode':
                    typecode = i
                elif k == 'nodata':
                    nodata = float(i)
##                elif k == 'undetect':
##                    undetect = i
##                elif k == 'gain':
##                    gain = i
##                elif k == 'offset':
##                    offset = i
                else:
                    raise KeyError("Unrecognized argument: %s" % k)
        else:
            raise AttributeError("No arguments to initialize new object")

        if areaid is None and radarid is None:
            raise AttributeError('"areaid" or "radarid" missing')

        if areaid:
            import pcs
            from types import StringType

            a = area.area(areaid)
            if type(a.pcs) is StringType:
                p = pcs.pcs(a.pcs)
            else:
                p = a.pcs

            self.info.insert(1, TopLevelWhat(obj = 'image',
                                             sets = sets,
                                             version = 'H5rad 1.2'))
            self.info.insert(1, Where(obj = 'image',
                                      xsize = a.xsize,
                                      ysize = a.ysize,
                                      xscale = a.xscale,
                                      yscale = a.yscale,
                                      projdef = p.tostring()))
            self.set('/how/software', 'RAVE')
            self.set('/how/area', areaid)
            prefix = 'image'

            if not sets: sets = 1
            for i in range(sets):
                self.addDataset(prefix, i+1, typecode,
                                a.xsize, a.ysize, nodata)
            
        elif radarid:
            import radar
            r = radar.radar(radarid)
            xsize, ysize = r.bins, r.rays
            prefix = 'scan'
            if not sets: sets = len(r.angles)
            # Keep working...


    def addDataset(self, prefix='image', set=1, typecode='B',
                   xsize=None, ysize=None, initval=None, **args):
        """
        Adds a complete dataset to the object.
        A new dataset Group is added to INFO and a new array is added to .data.

        Arguments:

        Returns: nothing, the new dataset is added to self.
        """
        from rave_h5rad import DatasetArray

        sets = self.get('/what/sets')

        if sets >= set:
            xsize, ysize = self.get('/where/xsize'), self.get('/where/ysize')
            set = sets+1  # failsafe

        self.data[prefix + str(set)] = DatasetArray(xsize, ysize,
                                                    typecode, initval)
        self.info.addDataset(prefix='/'+prefix, set=set, **args)


    def typecode(self, set=1):
        """
        Returns the typecode of the dataset given by "set".

        Argument:
          int set: the dataset

        Returns: the tyecode string returned by array.typecode()
        """
        prefix = self.data.keys()[0][:-1]
        return self.data[prefix + str(set)].dtype.char


    def get(self, path):
        """
        Get the contents of an infoset element. This method can be used to
        get scalar attributes as well as dataset data.
        
        Arguments:
          string path: the element to find. This is given in the infoset
                       format "/path/name".

        Returns: The value of the element, or None if the attribute is
                 unavailable.
        """        
        from H5radHelper import findelem, geth5attr
        e = findelem(self.info, path)
        if e is None:
            return None
        return geth5attr(e, self.data)


    def set(self, path, value):
        """
        Creates and sets the value of a new infoset element. 
        
        Arguments:
          string path: This is given in the infoset format, ie. "/path/name".
                       If the payload already exists, it will be modified with
                       the new value. If not, it will be created. Groups are
                       created automatically.

          value: The value to set. This must be an int, float, string, list
                 of strings, or a numpy array.

        Returns: Nothing if successful, a ValueError exception if the data
                 type is invalid or the payload is illegal.
        """        
        from H5radHelper import h5type, findelem, seth5attr, addelem
        from xml.etree.ElementTree import Element

        h5typ = h5type(value)
        if not h5typ:
            raise ValueError("Unsupported type %s" % type(value))
        if self.info is None:
            self.info = Element("h5rad", version=H5RAD_VERSION)
        e = findelem(self.info, path)
        if e is None:
            e = addelem(self.info, path)
        seth5attr(e, self.data, h5typ, path, value)


    def delete(self, path):
        """
        Deletes the infoset element specified by "path".

        Arguments:
        string path: The element to delete, given in the infoset format, ie.
                     "/path/name".

        Returns: Nothing if successful, otherwise a ValueError exception if
                 the data type is invalid or the payload is illegal.
        """
        from H5radHelper import type_val
        typ, val = type_val(self.info.find(path[1:]))
        self.info.delete(path)
        if typ is 'dataset':
            del(self.data[val])


    def getattribute(self, path, key):
        """
        Gets the value of 'key' from Element 'path'.
        Same as self.info.find(path).get(key)

        Arguments:
          string path: the attribute's path
          string key: the attribute's key to query

        Returns: the item of the key describing attribute 'path'
        """
        return self.info.find(path[1:]).get(key)


    def putattribute(self, path, value, typ):
        """
        This is just a mapping to the same method in rave_info.INFO
        Changes the content of attribute "path" to contain "value" of type
        "typ".

        Arguments:
          string path: the attribute's path. If this path doesn't exist in this
                       INFO object, then an AttributeError exception is raised.
          value: the thing to put, can be string, int, float, list, or tuple.
          string typ: option to explicitely specify the type attribute. Can be
                      any of "string", "int", "float", or "sequence".

        Returns: nothing. The attribute's value is simply replaced.
        """
        self.info.put(path[1:], value, typ)


    def setattribute(self, path, key, value):
        """
        Sets the Element (at 'path') attribute 'key' to 'value'.
        Same as self.info.find(path).set(key, value)
        
        Arguments:
            string path: path to Element
            string key: key of Element's attribute
            string value: item to be the value of this Element's attribute
            
        Returns: nothing, the Element's attribute is set.
        """
        self.info.find(path[1:]).set(key, value)


    def eval(self, path=None):
        """
        This is just a mapping to the same method in rave_info.INFO
        Evaluates the attribute given by "path" and returns it as its proper
        type.
        
        Arguments:
          string path: the attribute's path. If this path doesn't exist in this
                       INFO object, then an AttributeError exception is raised.

        Returns: a Python int, float, string, or sequence, depending on
                 the contents of the given attribute.
        """
        return self.info.eval(path)


    def open(self, filename):
        """
        Opens a RAVE object from file.
        
        Arguments:

        Returns:
        """
        self.info, self.data, self._h5nodes = rave_IO.open(filename)
        self.__file__ = filename


    def read_metadata(self, filename):
        """
        Reads metadata from file and returns it in a properly-formatted RAVE
        object.

        Arguments:
          string filename: file string

        Returns: a RAVE object with empty .data
        """
        self.info, self._h5nodes = rave_IO.get_metadataRAVE(filename)
        self.data = {}
        self.__file__ = filename


    def read_dataset(self, index=1):
        """
        Reads dataset with the given index into the existing RAVE object.

        Arguments:
          int index: the index of the dataset to read

        Returns: nothing, the dataset is read into self.
        """
        if index > self.eval('/what/sets'):
            raise IndexError("Don't have that many datasets.")

        a = _pyhl.read_nodelist(self.__file__)
        counter = 1

        for k, i in a.getNodeNames().items():
            if i == _pyhl.DATASET_ID:
                if index == counter:
                    none1, tag, none2 = k.split('/')
                    try:
                        check = self.data[tag]  # if already there, do nothing
                        return
                    except KeyError:
                        a.selectNode(k)
                        a.fetch()
                        self.data[tag] = a.getNode(k).data()
                        self.info.find(k[1:]).text = tag  # self.put doesn't work
                        self.set(k[1:], 'type', 'dataset')
                        return
                else:
                    counter += 1


    def read_datasets(self, indices=None):
        """
        Reads datasets into self.

        Arguments:
          list indices: optional list of indices of datasets to read.

        Returns: nothing, the datasets are all read into self.
        """
        if indices is None:
            counter = 1
            for s in range(self.eval('/what/sets')):
                self.read_dataset(counter)
                counter += 1
        else:
            for i in indices:
                self.read_dataset(i)


    def set_fcp(self, userblock=0, sizes=(4,4),
                sym_k=(1,1), istore_k=1,
                meta_block_size=0):
        """
        Optimizes default HDF5 file-creation properties when writing new files.
        
        Arguments:
          int userblock:
          tuple of two ints sizes:
          tuple of two ints sym_k:
          int istore_k:
          int meta_block_size:

        Returns: nothing, a _pyhl.filecreationproperty() is initialized
                 to self._fcp
        """
        self._fcp = _pyhl.filecreationproperty()
        if userblock is not None:
            self._fcp.userblock = userblock
        if sizes is not None:
            self._fcp.sizes = sizes
        if sym_k is not None:
            self._fcp.sym_k = sym_k
        if istore_k is not None:
            self._fcp.istore_k = istore_k
        if meta_block_size is not None:
            self._fcp.meta_block_size = meta_block_size


    def save(self, filename):
        """
        Writes the RAVE object to file.
        
        Arguments:

        Returns:
        """
        ID = ""
        a = _pyhl.nodelist() # top level
        rave_IO.traverse_save(self.info, a, ID, self.data)
        if self._fcp is None:
            self.set_fcp()
        a.write(filename, self._fcp)


    def asXML(self, filename=None):
        """
        Outputs a metadata representation of a RAVE object to XML
        with line breaks, either to file or to stdout.
        
        Arguments:

        Returns:
        """
        self.info.asXML(filename)


    def pureXML(self, filename, encoding=ENCODING):
        """
        Outputs a metadata representation of a RAVE object to XML
        without line breaks, either to file or to stdout.

        Arguments:

        Returns:
        """
        self.info.pureXML(filename)


    def ql(self, index=0, pal=None):
        """
        QuickLook visualization of a dataset given by its index.
        """
        import subprocess
        import rave_ql, rave_tempfile
        
        keys = self.data.keys()
        if len(keys) < index:
            raise IndexError("Don't have that many datasets.")

        keys.sort()
#        prefix = keys[0][:-1]  # Determines 'image', 'scan', or 'profile'.
        prefix = keys[0]  # Determines 'image', 'scan', or 'profile'.

        title = "QuickLook"
        if self.__file__ is not None:
            title = '%s of %s' % (title, self.__file__)
#        key = prefix + str(index)
        key = prefix
        title = "%s : %s" % (title, key)

        if pal is None:
            import rave_win_colors

            try:
                quant = self.eval(key + '/what/quantity')
                if quant == 'DBZ': pal = rave_win_colors.continuous_dBZ
                elif quant == 'VRAD': pal = rave_win_colors.continuous_MS
                # Add more palettes for different variables.
                else: raise KeyError
            except:
                pal = rave_ql.palette

        tmp = rave_IO.Array2Tempfile(self.data[key])

        command = RAVEBIN + '/show -i'
        if sys.platform == "darwin":
            # on darwin command returns immediately resulting in the tmp
            # file removal while the app is opening. Thanks Fredrik Lundh...
            command = "(%s %s; sleep 20; rm -f %s)&" % (command, tmp, tmp)
        else:
            command = "(%s %s; rm -f %s)&" % (command, tmp, tmp)

        try:
            retcode = subprocess.call(command, shell=True)
            if retcode < 0:
                sys.stderr.write("Child was terminated by signal %d"%retcode)
        except OSError as e:
            sys.stderr.write("Could not show: %s"%e.__str__())


    def MakeExtentFromCorners(self):
        """
        Derives the four-tuple area extent for a given 2-D Cartesian
        image. These four floats are expressed in surface (UCS) coordinates
        according to the PROJ.4 library, and represent, in order, the
        lower-left longitude, lower-left latitude, upper-right longitude,
        and upper-right latitude coordinates. Specifically, these are the
        lower-left corners of these pixels, ie. NOT the outer corner of the
        upper-right pixel!

        The /where/LL_lon, LL_lat, UR_lon, and UR_right attributes must
        exist.
        """
        import Proj

        projdef = self.get('/where/projdef')
        p = Proj.Proj(projdef.split())
        LL = self.get('/where/LL_lon'), self.get('/where/LL_lat')
        if LL == ('n/a', 'n/a'):
            raise AttributeError("Lon/lat corner attributes must exist")
        UR = self.get('/where/UR_lon'), self.get('/where/UR_lat')
        LLs = p.proj(Proj.d2r(LL))
        URs = p.proj(Proj.d2r(UR))

        xscale, yscale = self.get('/where/xscale'), self.get('/where/yscale')
        self.set('/how/extent', (LLs[0], LLs[1],
                                 URs[0]-xscale, URs[1]-yscale))


    def MakeCornersFromExtent(self):
        """
        Derives /where/LL_lon, LL_lat, UR_lon, and UR_right attributes
        from /how/extent and /where/projdef
        """
        import Proj

        extent = self.get('/how/extent')
        if not extent:
            raise AttributeError("/how/extent is missing")

        projdef = self.get('/where/projdef')
        p = Proj.Proj(projdef.split())

        xscale, yscale = self.get('/where/xscale'), self.get('/where/yscale')
        LL_lon, LL_lat = Proj.r2d(p.invproj((extent[0],extent[1])))
        UR_lon, UR_lat = Proj.r2d(p.invproj((extent[2]+xscale,
                                             extent[3]+yscale)))
        self.set('/where/LL_lon', LL_lon)
        self.set('/where/LL_lat', LL_lat)
        self.set('/where/UR_lon', UR_lon)
        self.set('/where/UR_lat', UR_lat)


    def MakeCornersFromArea(self, areaid=None):
        if not areaid:
            raise AttributeError("Missing areaid argument")

        import Proj, pcs
        from types import StringType

        a = area.area(areaid)
        if type(a.pcs) is StringType:
            p = pcs.pcs(a.pcs)
        else:
            p = a.pcs

        LL = Proj.r2d(p.invproj((a.extent[0],a.extent[1])))
        UR = Proj.r2d(p.invproj((a.extent[2]+a.xscale,
                                 a.extent[3]+a.yscale)))
        self.set('/where/LL_lon', LL[0])
        self.set('/where/LL_lat', LL[1])
        self.set('/where/UR_lon', UR[0])
        self.set('/where/UR_lat', UR[1])


    def transform(self, areaid, method=rave_transform.NEAREST, radius=None):
        """
        Simple wrapper to rave_transform.transform() for
        cartesian-to-cartesian transformation.
        """
        return rave_transform.transform(self, areaid, method=method,
                                        radius=radius)


#
# FUNDAMENTAL FUNCTIONS
#


def open(filename=None):
    """
    Opens a file and returns its content as a RAVE object.
    Files must be in a recognized format, and file content must be organized
    according to a known information model.

    Arguments:

    Returns:
    """
    if filename is None:
        print("No file given")
    elif os.path.isfile(filename):
        this = RAVE()
        this.open(filename)
        return this
    else:
        print("%s is not a valid file" % str(filename))


def get_metadata(filename=None):
    """
    Same as the open() function, but only reads metadata.
    Returns only INFO.
    Much faster and more convenient for managing metadata.
    This function calls rave_IO.get_metadata().

    Arguments:
      string filename: file string

    Returns: an INFO object
    """
    if filename is None:
        print("No file given")
    elif os.path.isfile(filename):
        return rave_IO.get_metadata(filename)
    else:
        print("%s is not a valid file" % str(filename))


def get_metadataXML(filename=None):
    """
    Like get_metadata(), but returns the metadata as a single XML string.
    Suitable for formatting metadata prior to communicating through a socket.
    This function calls rave_IO.MetadataAsXML().

    Arguments:
      string filename: file string

    Returns: a Python string containing a pure XML representation of metadata.
    """
    if filename is None:
        print("No file given")
    elif os.path.isfile(filename):
        element = rave_IO.get_metadata(filename)
        return rave_IO.MetadataAsXMLstring(element)
    else:
        print("%s is not a valid file" % str(filename))


def get_metadataRAVE(filename=None):
    """
    Like get_metadata(), but returns a RAVE object containing the metadata
    and a complete list of _h5nodes.
    Suitable for reading metadata first and then deciding later which datasets
    to read.
    This function calls rave_IO.get_metadataRAVE().

    Arguments:
      string filename: file string

    Returns: a RAVE object
    """
    if filename is None:
        print("No file given")
    elif os.path.isfile(filename):
        this = RAVE()
        this.read_metadata(filename)
        return this
    else:
        print("%s is not a valid file" % str(filename))



__all__ = ['RAVE','open','get_metadata','get_metadataXML', 'get_metadataRAVE']


if __name__ == "__main__":
    print(__doc__)
