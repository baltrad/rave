#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id:  $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2009-
#                All rights reserved.
#
# $Log:  $
#
#
"""
rave_odimh5.py - Functions for initializing new RAVE objects, organized
                 according to the ODIM_H5 information model.

The information model followed when representing data and products is that
developed in EUMETNET OPERA for use with HDF5. References to the tables in the
information model specification are made throughout this module.

Michelson D.B., Lewandowski R., Szewczykowski M., and Beekhuis H., 2009:
EUMETNET OPERA weather radar information model for implementation with the HDF5
file format.
OPERA Working Document WD_2008_03.
Available at http://www.knmi.nl/opera/opera3/OPERA_2008_03_WP2.1a_InformationModel_UML_2.0.pdf
"""
from xml.etree.ElementTree import  Element, SubElement
import rave_info
from rave_defines import H5RAD_VERSION


def Root(tag='h5rad', attrib={"version":H5RAD_VERSION}):
    """
    This is the root element: an INFO object with the right version
    attribute and with a UNIDATA Conventions attribute.

    Arguments:
      string tag: should always be "h5rad"
      dict attrib: should always be the h5rad version

    Returns:
      an INFO object
    """
    this = rave_info.INFO(tag=tag, attrib=attrib)
    this.set('/Conventions', rave_defines.ODIM_CONVENTIONS)
    return this


def Dataset(tag="data", attrib={"type":"dataset"}, prefix='image', set=1):
    """
    Simple dataset representation in INFO.

    Arguments:
      string tag: Should always be "data".
      dict attrib: Should always be of type "dataset" for datasets.
      string prefix: Can be any of: "image", "scan", "profile"
      int set: Denotes the index of the dataset among other datasets.

    Returns:
      an Element instance denoting the presence of a dataset, eg.
      <data type="dataset">image1</data>
    """
    if prefix in ["image","scan","profile"]:
        E = Element(tag=tag, attrib=attrib)
        E.text = prefix + str(set)
        return E
    else:
        raise AttributeError('Invalid "prefix" argument: %s' % prefix)


def DatasetGroup(prefix='image', set=1, **args):
    """
    A complete Group representing a dataset.
    Contains the dataset together with corresponding "what" attributes.
    Additional metadata must be added seperately.

    Arguments:
      string prefix: Can be any of: "image", "scan", "profile"
      int set: Denotes the index of the dataset among other datasets.
      Additional agruments:
        Any attributes in dataset-specific "what" can be specified here.
        They will otherwise be initialized to "n/a".
        These can be any of the following, specified in Table 13:
          string product: Table 14
          float prodpar: Product parameter
          string quantity: Table 15
          string startdate: YYYYMMDD
          string starttime: HHmmss
          string enddate: YYYYMMDD
          string endtime: HHmmss
          float gain: Coefficient 'a' in y=ax+b used to convert int to float
          float offset: Coefficient 'b' in y=ax+b used to convert int to float
          float nodata: Raw value used to denote areas void of data
          float undetect: Raw value used to denote areas below the measurement
                          detection threshold (radiated but nothing detected).

    Returns: An Element containing a complete dataset Group including metadata
             structures.
    """
    TAG = prefix + str(set)
    E = Element(tag=TAG)
    E.append(Dataset(prefix=prefix, set=set))
    E.append(DatasetWhat(**args))
    return E


def DatasetArray(xsize=None, ysize=None, typecode='B', initval=None):
    """
    Creates an empty numpy array, containing zeros, based on the
    input arguments.
    Can be appended to a RAVE object's "data" dictionary.

    Arguments:
      int xsize: the number of columns in the array (horizontal dimension)
      int ysize: the number of rows in the array (vertical dimension)
      string typecode: the typecode of the array. Can be any of those
                       supported in numpy, although B,s,i,l,f,d are safest.
      int or float initval: value with which to initialize the whole array

    Returns: a numpy array
    """
    import numpy

    if xsize is None or ysize is None:
        raise AttributeError("Missing xsize or ysize when creating dataset array.")

    else:
        if initval:
            return (numpy.zeros((ysize, xsize))+initval).astype(typecode)
        else:
            return numpy.zeros((ysize, xsize), typecode)



def TopLevelWhat(tag='what', **args):
    """
    Representation of top-level "what" Group.

    Arguments:
      string tag: Should always be "what".
      Additional arguments:
        Any attributes in top-level "what" can be specified here.
        They will otherwise be initialized to "n/a".
        They can be any of the following:
          string object: Table 2
          int sets: The number of datasets in this file.
          string version: Presently "H5rad 1.2".
          string date: YYYYMMDD
          string time: HHmmss

    Returns: An Element containing top-level "what" attributes.
    """
    STRINGS = ['obj','version','date','time']
    INTS = ['sets']

    E = Element(tag=tag)

    if len(args) != 0:
        for k, i in args.items():
            if k in STRINGS:
                SubElement(E, k, attrib={}).text = i
                del(STRINGS[STRINGS.index(k)])
            elif k is 'sets':
                SubElement(E, k, attrib={"type":"int"}).text = str(i)
                INTS = []
            else:
                raise KeyError('Illegal key: "%s" for top-level what.'%str(k))

    for k in STRINGS + INTS:
        SubElement(E, k, attrib={}).text = 'n/a'

    return E


def DatasetWhat(tag='what', **args):
    """
    Dataset-specific "what" attributes.

    Arguments:
      string tag: Should always be "what".
      Additional arguments:
        Any arguments in dataset-specific "what" can be specified here.
        They will otherwise be initialized to "n/a".
        These can be any of the following, specified in Table 13:
          string product: Table 14
          float prodpar: Product parameter
          string quantity: Table 15
          string startdate: YYYYMMDD
          string starttime: HHmmss
          string enddate: YYYYMMDD
          string endtime: HHmmss
          float gain: Coefficient 'a' in y=ax+b used to convert int to float
          float offset: Coefficient 'b' in y=ax+b used to convert int to float
          float nodata: Raw value used to denote areas void of data
          float undetect: Raw value used to denote areas below the measurement
                          detection threshold (radiated but nothing detected).

    Returns: An Element containing dataset-specific "what" attributes.
    """
    STRINGS = ['product','quantity',
               'startdate','starttime','enddate','endtime']
    FLOATS = ['prodpar','gain','offset','nodata','undetect']

    E = Element(tag=tag)

    if len(args) != 0:
        for k, i in args.items():
            if k in STRINGS:
                SubElement(E, k, attrib={}).text = i
                del(STRINGS[STRINGS.index(k)])
            elif k in FLOATS:
                SubElement(E, k, attrib={"type":"float"}).text = str(i)
                del(FLOATS[FLOATS.index(k)])
            else:
                raise KeyError('Illegal key: "%s" for dataset-specific what.'%str(k))

    for k in STRINGS + FLOATS:
        SubElement(E, k, attrib={}).text = 'n/a'

    return E


def Where(tag='where', **args):
    """
    Top-level "where" attributes for all object types.

    Arguments:
      string tag: Should always be "where".
      string object: Table 2
      Additional arguments (Tables 4-7):
        Any arguments in dataset-specific "what" can be specified here.
        They will otherwise be initialized to "n/a".
        They can be any of the following (this can be a bit messy...):
          For "scan" and "pvol" objects:
            int xsize: The number of range bins along a ray.
            int ysize: The number of azimuth gates in a sweep/scan.
            float lon: The radar's longitude, using decimal degrees.
            float lat: The radar's latitude, using decimal degrees.
            float height: The radar's height in m a s l.
            float angle: The elevation angle in degrees.
            float xscale: The range bin spacing in meters.
          For "image" and "comp" objects:
            int xsize: The number of pixels in the horizontal (E-W) direction.
            int ysize: The number of pixels in the vertical (N-S) direction.
            float xscale: Horizontal resolution in meters in the E-W direction.
            float yscale: Horizontal resolution in meters in the N-S direction.
            float LL_lon: Longitude of the lower-left image corner.
            float LL_lat: Latitude of the lower-left image corner.
            float UR_lon: Longitude of the upper-right image corner.
            float LR_lat: Latitude of the upper-right image corner.
          For "xsect" objects:
            Common attributes:
              int xsize: pixels in horizontal dimension.
              int ysize: pixels in vertical dimension.
              float xscale: horisontal resolution in meters.
              float yscale: vertical resolution in meters.
            RHI-specific:
              float lon: Longitude of radar antenna (degrees).
              float lat: Latitude of radar antenna (degrees).
              float az_angle: Azimuth angle (degrees).
              float range: Maximum range in km.
            Cross section and side-panel specific:
              float start_lon: Start position's longitude.
              float start_lat: Start position's latitude.
              float stop_lon: Stop position's longitude.
              float stop_lat: Stop position's latitude.
          For "vp" objects:
            float lon: Longitude of radar antenna (degrees).
            float lat: Latitude of radar antenna (degrees).
            float height: Height of the feed horn with antenna at the horizon
                          in m a s l.
            int levels: Number of points in the profile.
            float interval: Vertical distance (m) between height intervals,
                            or 0.0 if unavailable.
          For "THVP" objects:
            float lon: Longitude of radar antenna (degrees).
            float lat: Latitude of radar antenna (degrees).
            float height: Height of the feed horn with antenna at the horizon
                          in m a s l.
            int xsize: Number of pixels on the time axis.
            int ysize: Number of pixels in the vertical dimension.
            float xscale: Time resolution in seconds.
            float yscale: Vertical resolution in meters.
            float xoffset: Time (UTC) of the first profile in seconds after
                           00:00 UTC on the day defined by the "date"
                           Attribute in the "what" Group.
            float yoffset: Height of the first level of profile in meters.
            
    Returns: An Element containing top-level "where" for the given object.

    """
    INTS = ['xsize','ysize']
    SCAN_FLOATS = ['lon','lat','height','angle','xscale']
    IMAGE_FLOATS = ['xscale','yscale','LL_lon','LL_lat','UR_lon','UR_lat']
    STRINGS = ['projdef']
    XSECT_FLOATS = ['xscale','yscale','lon','lat','az_angle','range',
                    'start_lon','start_lat','stop_lon','stop_lat']
    VP_FLOATS = ['lon','lat','height','interval']
    VP_INT = ['levels']
    THVP_FLOATS = ['lon','lat','height','xscale','yscale','xoffset','yoffset']
    errmesg = 'Need to know object type to initialize "where" attributes.'

    E = Element(tag=tag)

    if len(args) == 0:
        raise IOError(errmesg)

    elif len(args) > 0:
        obj = args['obj'].lower()
        del(args['obj'])

        # Polar scans and polar volumes
        if obj in ['scan','pvol']:
            for k, i in args.items():
                if k in SCAN_FLOATS:
                    SubElement(E, k, attrib={"type":"float"}).text = str(i)
                    del(SCAN_FLOATS[SCAN_FLOATS.index(k)])
                elif k in INTS:
                    SubElement(E, k, attrib={"type":"int"}).text = str(i)
                    del(INTS[INTS.index(k)])
                else:
                    raise KeyError('Illegal key: "%s" for where' % str(k))

        # Cartesian images, volumes, and composites (mosaics)
        elif obj in ['image','comp','cvol']:
            for k, i in args.items():
                if k in STRINGS:
                    SubElement(E, k, attrib={}).text = i
                    STRINGS = []
                elif k in IMAGE_FLOATS:
                    SubElement(E, k, attrib={"type":"float"}).text = str(i)
                    del(IMAGE_FLOATS[IMAGE_FLOATS.index(k)])
                elif k in INTS:
                    SubElement(E, k, attrib={"type":"int"}).text = str(i)
                    del(INTS[INTS.index(k)])
                else:
                    raise KeyError('Illegal key: "%s" for where' % str(k))

        # Cross sections, including RHIs, and side panels
        elif obj is 'xsect':
            for k, i in args.items():
                if k in INTS:
                    SubElement(E, k, attrib={"type":"int"}).text = str(i)
                    del(INTS[INTS.index(k)])
                elif k in XSECT_FLOATS:
                    SubElement(E, k, attrib={"type":"float"}).text = str(i)
                    del(XSECT_FLOATS[XSECT_FLOATS.index(k)])

        # Vertical profiles
        elif obj is 'vp':
            for k, i in args.items():
                if k in VP_INTS:
                    SubElement(E, k, attrib={"type":"int"}).text = str(i)
                    VP_INTS = []
                elif k in VP_FLOATS:
                    SubElement(E, k, attrib={"type":"float"}).text = str(i)
                    del(VP_FLOATS[VP_FLOATS.index(k)])

        elif obj is 'thvp':
            for k, i in args.items():
                if k in INTS:
                    SubElement(E, k, attrib={"type":"int"}).text = str(i)
                    del(INTS[INTS.index(k)])
                elif k in THVP_FLOATS:
                    SubElement(E, k, attrib={"type":"float"}).text = str(i)
                    del(THVP_FLOATS[THVP_FLOATS.index(k)])

        else:
            raise IOError(errmesg)

        # Fill in the blanks

        if obj in ['scan','pvol']:
            for k in SCAN_FLOATS:
                SubElement(E, k, attrib={"type":"float"}).text = 'n/a'

        elif obj in ['image','comp','cvol']:
            for k in STRINGS + IMAGE_FLOATS:
                SubElement(E, k, attrib={}).text = 'n/a'

        elif obj is 'xsect':
            for k in XSECT_FLOATS:
                SubElement(E, k, attrib={"type":"float"}).text = 'n/a'

        elif obj is 'vp':
            for k in VP_INTS + VP_FLOATS:
                SubElement(E, k, attrib={}).text = str(0)

        elif obj is 'thvp':
            for k in THVP_FLOATS:
                SubElement(E, k, attrib={"type":"float"}).text = 'n/a'

        if obj is not 'vp':
            for k in INTS:
                SubElement(E, k, attrib={"type":"int"}).text = str(1)

    return E


__all__ = ['Root', 'Dataset','DatasetGroup','TopLevelWhat','DatasetWhat',
           'Where']


if __name__ == "__main__":
    print __doc__
