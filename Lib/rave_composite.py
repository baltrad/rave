#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_composite.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2006-
#                All rights reserved.
#
# $Log: rave_composite.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_composite.py

Functionality for compositing in RAVE.
"""
import os
import rave, rave_IO, rave_defines, radar, area
import _composite, _helpers


# Composite algorithm constants
NEAREST = 0  # nearest radar
LOWEST  = 1  # lowest pixel

# Other constants
NODATA = 255.0
UNDETECT = 0.0
TYPECODE = 'b'

def composite(sources, areaid, method=NEAREST, bitmap=1,
              gain=rave_defines.GAIN, offset=rave_defines.OFFSET,
              nodata=NODATA, undetect=UNDETECT, typecode=TYPECODE, qc=0):
    """
    """
    nodes = []

    # Initialize a reference object containing output area's characteristics
    projdef = area.area(areaid).pcs.definition

    # Loop through input images
    for image in sources:

        # Does this image have the right projection?
        pj = image.get('/where/projdef').split()
        for p in pj:
            if p not in projdef:
                raise AttributeError, "Incorrect input image projection"

        # Common gain and offset!
        if image.get('/image1/what/gain') != gain or \
           image.get('/image1/what/offset') != offset:
            _helpers.CommonGainOffset(image, 1, gain, offset)

        # Correct value of 'nodata'? Pretty likely, but you never know...
        if image.get('/image1/what/nodata') != nodata:
            NewNodata(image, nodata)

        # Do all images have the same depth? If not, convert.
        if image.typecode() != typecode:
            image.data['image1'] = image.data['image1'].astype(typecode)

        image.MakeExtentFromCorners()
        imagearea = image.get('/how/area')
        r = radar.radar(imagearea[:3])
        image.set('/how/lon_0', r.lon)
        image.set('/how/lat_0', r.lat)
        nodes.append(r.Id)

        # If LOWEST algorithm: load radar_height lookup-table for each input
        # and add it as /image2/data
        if method == LOWEST:
            a = area.area(imagearea)
            p = a.pcs.id
            radar_height = "radar_height/radar_height.%s_%s.h5" % (imagearea[:3], p)
            radar_height = os.path.join(rave_defines.RAVEDB, radar_height)
            this = rave.RAVE()  # temporary container object
            t = rave_IO.open_hdf5(radar_height)
            this.info, this.data = t[0], t[1]
            image.set('/image2/data', this.get('/image1/data'))

    dest = initDest(image, areaid)
    dest.set('/how/nodes', nodes)

    # Quality control using information from overlapping radar,
    # as performed in MESAN.
    if qc:
        print "Warning: quality control is not available yet; compositing anyway..."

    # compositing using the shortest distance to a given radar
    if method == NEAREST:
        _composite.nearest(sources, dest, bitmap)

    # compositing using the shortest distance to the Earth's surface
    elif method == LOWEST:
        _composite.lowest(sources, dest, bitmap)

    else:
        raise AttributeError, "Only NEAREST and LOWEST algorithms supported"

    return dest


def NewNodata(image, nodata):
    """
    """
    from numpy import where, equal

    data = image.data['image1']
    data = where(equal(data, image.get('/image1/what/nodata')),
                 nodata, data)
    image.data['image1'] = data.astype(image.typecode())
    image.set('/image1/what/nodata', nodata)


def initDest(image, areaid):
    """
    """
    nodata = image.get('/image1/what/nodata')
    undetect = image.get('/image1/what/undetect')

    out = rave.RAVE(area=areaid, nodata=nodata)
    out.addDataset(initval=0.0)
    out.MakeCornersFromArea(areaid)
    out.MakeExtentFromCorners()

    DATE, TIME = image.get('/what/date'), image.get('/what/time')
    out.set('/what/date', DATE)
    out.set('/what/time', TIME)

    out.set('/image1/what/nodata', nodata)
    out.set('/image1/what/undetect', undetect)
    out.set('/image1/what/gain', image.get('/image1/what/gain'))
    out.set('/image1/what/offset', image.get('/image1/what/offset'))
    out.set('/image1/what/product', image.get('/image1/what/product'))
    out.set('/image1/what/quantity', image.get('/image1/what/quantity'))
    out.set('/image1/what/prodpar', image.get('/image1/what/prodpar'))
    out.set('/image1/what/startdate', DATE)
    out.set('/image1/what/starttime', TIME)
    out.set('/image1/what/enddate', DATE)
    out.set('/image1/what/endtime', TIME)

    out.set('/image2/what/nodata', nodata)
    out.set('/image2/what/undetect', undetect)
    out.set('/image2/what/gain', 1.0)
    out.set('/image2/what/offset', 0.0)
    out.set('/image2/what/product', 'COMP')
    out.set('/image2/what/quantity', 'BRDR')
    out.set('/image2/what/prodpar', image.get('/image1/what/prodpar'))
    out.set('/image2/what/startdate', DATE)
    out.set('/image2/what/starttime', TIME)
    out.set('/image2/what/enddate', DATE)
    out.set('/image2/what/endtime', TIME)

    out.set('/what/object', 'COMP')
    return out



__all__ = ["composite"]

if __name__ == "__main__":
    print __doc__
