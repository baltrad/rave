#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_transform.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2006-
#                All rights reserved.
#
# $Log: rave_transform.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_transform.py

Functionality for performing transformations between cartesian surfaces.
"""
import rave
import _ctoc
#import _h5rad

# transform algorithm constants
NEAREST = 1
BILINEAR = 2
CUBIC = 3
CRESSMAN = 4
UNIFORM = 5
INVERSE = 6


def transform(image, areaid, method=NEAREST, radius=None):
    """
    """
    # Initialize 'new' output image
    new = rave.RAVE(area=areaid, nodata=image.get('/image1/what/nodata'))
    image.info.CopyDatasetAttributes(new.info, ipath='/image1/what', oset=1)
    new.set('/what/date', image.get('/what/date'))
    new.set('/what/time', image.get('/what/time'))

    image.set('/how/i_method', method)
    if radius: image.set('/how/cressman_xy', float(radius))

    sets = image.get('/what/sets')
    set = 2

    # Initialize corresponding empty datasets beyond the first one
    while set <= sets:
        typecode = image.typecode(set)
        xsize, ysize = new.get('/where/xsize'), new.get('/where/ysize')
        initval = image.get('/image%i/what/nodata' % set)

        new.addDataset(set=set, typecode=typecode, xsize=xsize,
                       ysize=ysize, initval=initval)
        image.info.CopyDatasetAttributes(new.info,
                                         ipath='/image%i/what' % set,
                                         oset=set)
        set += 1

    image.MakeExtentFromCorners()
    new.MakeCornersFromArea(areaid)
    new.MakeExtentFromCorners()

#    _h5rad.read_h5rad(image, new)
    _ctoc.transform(image, new)

    # Clean up
    image.delete('/how/extent')
    new.delete('/how/extent')
    image.delete('/how/i_method')
    if radius: image.delete('/how/cressman_xy')

    return new


__all__ = ["transform"]

if __name__ == "__main__":
    print(__doc__)
