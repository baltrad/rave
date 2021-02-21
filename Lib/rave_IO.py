#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_IO.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2005-
#                All rights reserved.
#
# $Log: rave_IO.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_IO.py - 
"""
import os
import _pyhl
from xml.etree.ElementTree import SubElement, ElementTree
import rave_info
from H5radHelper import geth5attr, h5type
from rave_defines import *


def open(filename):
    if os.path.isfile(filename):
        if os.path.getsize(filename):
            if _pyhl.is_file_hdf5(filename):

                try:

                    info, data, _h5nodes = open_hdf5(filename)
                    #if info.find("what/version").text != H5RAD_VERSION:
                    if info.find("what/version").text not in H5RAD_VERSIONS:
                        raise IOError("Contents of file %s not organized according to %s or earlier." % (filename, H5RAD_VERSION))
                    return info, data, _h5nodes

                except:
                    raise IOError("Failed to read file %s" % filename)
            else:
                raise IOError("%s is not an HDF5 file" % filename)
        else:
            raise IOError("%s is zero-length" % filename)
    else:
        raise IOError("%s is not a regular file" % filename)


def open_hdf5(filename):
    datadict = {}

    a = _pyhl.read_nodelist(filename)
    a.selectAll()
    a.fetch()
    node_names = a.getNodeNames()

    items = []
    for nodename in node_names.keys():
        b = a.getNode(nodename)
        items.append((nodename, b, b.type()))

    items.sort() # guarantees groups before attributes

    h5rad = rave_info.INFO("h5rad", version=H5RAD_VERSION)

    groupmapping = {"" : h5rad}
    for nodename, node, typ in items:
        index = nodename.rindex("/")
        parentname, tag = nodename[:index], nodename[index+1:]
        # Deal with (ignore) H5IM stubs
        #if tag in ["CLASS", "IMAGE_VERSION"]:
        if os.path.split(parentname)[1] == 'data':
            continue
        e = SubElement(groupmapping[parentname], tag)
        if typ==1:
            groupmapping[nodename] = e # save the nodename to element mapping
        elif typ==0:
            t = h5type(node.data())
            if t == "sequence":
                # convert list to string
                nodes = []
                for n in node.data():
                    node = n.strip()
                    node = remove_nulls(str(node))
                    nodes.append(("'"+node+"'"))
                e.text = ", ".join(nodes)
            else:
                e.text = remove_nulls(str(node.data()))
            if t != "string":
                e.attrib["type"] = t
        elif typ==2:
            datadict[nodename] = node.data()
            e.attrib["type"] = "dataset"
            e.text=nodename            
##             label = string.replace(parentname, "/", "")
##             print parentname, label
##             if label.startswith("profile"):  # relic from 717 ...
##                 label = label + "_" + tag
##             datadict[label] = node.data()
##             e.attrib["type"] = "dataset"
##             e.text=label

    return h5rad, datadict, items


def get_metadata(filename):

    a = _pyhl.read_nodelist(filename)
    a.selectMetadata()
    a.fetch()
    node_names = a.getNodeNames()

    items = []
    for nodename in node_names.keys():
        b = a.getNode(nodename)
        items.append((nodename, b, b.type()))

    items.sort() # guarantees groups before attributes

    h5rad = rave_info.INFO("h5rad", version=H5RAD_VERSION)

    groupmapping = {"" : h5rad}
    for nodename, node, typ in items:
        index = nodename.rindex("/")
        parentname, tag = nodename[:index], nodename[index+1:]
        e = SubElement(groupmapping[parentname], tag)
        if typ==1:
            groupmapping[nodename] = e # save the nodename to element mapping
        elif typ==0:
            t = h5type(node.data())
            if t == "sequence":
                # convert list to string
                nodes = []
                for n in node.data():
                    node = n.strip()
                    node = remove_nulls(str(node))
                    nodes.append(("'"+node+"'"))
                e.text = ", ".join(nodes)
            else:
                e.text = remove_nulls(str(node.data()))
            if t != "string":
                e.attrib["type"] = t
        # Skip typ==2, dataset array

    # return only h5rad
    return h5rad


def get_metadataRAVE(filename):
    a = _pyhl.read_nodelist(filename)
    a.selectMetadata()
    a.fetch()
    node_names = a.getNodeNames()

    items = []
    for nodename in node_names.keys():
        b = a.getNode(nodename)
        items.append((nodename, b, b.type()))

    items.sort() # guarantees groups before attributes

    h5rad = rave_info.INFO("h5rad", version=H5RAD_VERSION)

    groupmapping = {"" : h5rad}
    for nodename, node, typ in items:
        index = nodename.rindex("/")
        parentname, tag = nodename[:index], nodename[index+1:]
        e = SubElement(groupmapping[parentname], tag)
        if typ==1:
            groupmapping[nodename] = e # save the nodename to element mapping
        elif typ==0:
            t = h5type(node.data())
            if t == "sequence":
                # convert list to string
                nodes = []
                for n in node.data():
                    node = n.strip()
                    node = remove_nulls(str(node))
                    nodes.append(("'"+node+"'"))
                e.text = ", ".join(nodes)
            else:
                e.text = remove_nulls(str(node.data()))
            if t != "string":
                e.attrib["type"] = t
        # Skip typ==2, dataset array

    # return only h5rad and nodelist
    return h5rad, items


# Add stub H5IM attributes.
def add_H5IM_attributes(a, IDA):
    b=_pyhl.node(_pyhl.ATTRIBUTE_ID, IDA+"/CLASS")
    b.setScalarValue(-1, "IMAGE", "string", -1)
    a.addNode(b)
    b=_pyhl.node(_pyhl.ATTRIBUTE_ID, IDA+"/IMAGE_VERSION")
    b.setScalarValue(-1, "1.2", "string", -1)
    a.addNode(b)


def traverse_save(e, a, ID, datadict):
    for i in list(e):
        if list(i):
            IDA = ID + "/"+ i.tag
            b =_pyhl.node(_pyhl.GROUP_ID, IDA)
            a.addNode(b)
            traverse_save(i, a, IDA, datadict)
        else:
            typ = i.get("type", "string")
            value = geth5attr(i, datadict)
            IDA = ID+"/"+i.tag
            h5typ = None
            if typ == "dataset":
                if COMPRESSION == "zlib":
                    comp = _pyhl.compression(_pyhl.COMPRESSION_ZLIB)
                    comp.level = COMPRESSION_ZLIB_LEVEL
                elif COMPRESSION == "szip":
                    comp = _pyhl.compression(_pyhl.COMPRESSION_SZLIB)
                    comp.szlib_px_per_block = 10
                else:
                    comp = None
                if comp is not None:
                    b = _pyhl.node(_pyhl.DATASET_ID, IDA, comp)
                else:
                    b = _pyhl.node(_pyhl.DATASET_ID, IDA)
                h5typ = ARRAYTYPES[value.dtype.char]
                #h5typ = ARRAYTYPES[value.typecode()]  # relic from Numeric
                b.setArrayValue(-1, list(value.shape), value, h5typ, -1)
            elif typ == "sequence":
                b = _pyhl.node(_pyhl.ATTRIBUTE_ID, IDA)
                if type(value[0]) in [int, float]:
                    v = []
                    for val in value:
                        v.append(str(val))
                    value = v
                b.setArrayValue(-1, [len(value)], value, "string", -1)
            else:
                b =_pyhl.node(_pyhl.ATTRIBUTE_ID, IDA)
                b.setScalarValue(-1, value, typ, -1)
            a.addNode(b)

            # Workaround. For 8-bit uchar datasets, add H5IM attributes.
            if typ == "dataset" and h5typ == 'uchar':
                add_H5IM_attributes(a, IDA)
                


# Don't know if writing to tempfile is less efficient than doing everything
# in memory, since tempfile IS memory these days...
def MetadataAsXMLstring(element, encoding=ENCODING):
    import __builtin__, os
    import rave_tempfile

    # Write to tempfile
    fid, fstr = rave_tempfile.mktemp()
    os.close(fid)
    e = ElementTree(element)
    e.write(fstr, encoding=encoding)

    # Read tempfile into string and return it
    fd = __builtin__.open(fstr)
    rets = fd.read()
    fd.close()
    os.remove(fstr)
    return rets


def prettyprint(element, encoding=None, indent=""):
    if element is None:
        return
    if isinstance(element, ElementTree):
        element = element.getroot()
    if encoding is None:
        import locale
        _, encoding = locale.getdefaultlocale()
        if not encoding or encoding == "utf":
            encoding = ENCODING
        if sys.platform == "win32" and encoding == "cp1252":
            encoding = "cp850" # FIXME: bug in getdefaultlocale?
    start_tag = "<%s" % element.tag
    if element.attrib:
        items = element.items()
        items.sort()
        for k, v in items:
            start_tag = start_tag + (" %s=\"%s\"" % (k, v))
    start_tag = (start_tag + ">").encode(encoding, "replace")
    end_tag = ("</%s>" % element.tag).encode(encoding, "replace")
    if element:
        subindent = indent + "  "
        print(indent + start_tag)
        for subelement in element:
            prettyprint(subelement, encoding, subindent)
        print(indent + end_tag)
    else:
        print(indent + "%s%s%s" % (
            start_tag,
            (element.text or "").encode(encoding, "replace"),
            end_tag
            ))


def Array2Tempfile(value):
    import rave_tempfile

    _, fstr = rave_tempfile.mktemp()
    a = _pyhl.nodelist()
    b = _pyhl.node(_pyhl.GROUP_ID, "/dataset1")
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID, "/dataset1/data")
    h5typ = ARRAYTYPES[value.dtype.char]
    b.setArrayValue(-1, list(value.shape), value, h5typ, -1)
    a.addNode(b)
    a.write(fstr)
    return fstr


def remove_nulls(s):
    if s == '\x00' or s == '':
        return ''
    while s != '' and s[-1] == '\x00':
        s = s[:-1]
    return s



__all__ = ['prettyprint','MetadataAsXMLstring','traverse_save','get_metadata',
           'open_hdf5','Array2Tempfile']

if __name__ == "__main__":
    print(__doc__)
