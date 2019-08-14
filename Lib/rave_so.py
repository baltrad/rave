#!/usr/bin/env python
# -*- coding: latin-1 -*-
# $Id$
# Author(s): Daniel Michelson and Günther Haase
# Copyright: SMHI, 1999, 2000-
# History: 
#	  1999-12-?? dmichels first version with single volumes
#	  2000-04-26 dmichels modified for use with new _ptop which uses
#			      wind and reflectivity volumes.
#         2008-03-04 ghaase   adjustments to handle HDF5 file format
#         2008-11-04 dmichels further tweaks prior to release
"""
rave_so.py - RAVE super-observations (SO). Defines new, generalized, pvol
             instances on the fly and feeds them to ptop for transformation.
             Output is to ASCII so modellers can manage...

             Input must be polar volume or scan files containing both Z and V.
"""
import sys, os
import rave, rave_tempfile
import _ptop
from rave_h5rad import DatasetArray
from numpy import *


USE_SINGLE_ELEV = 1
USE_MANY_ELEV = 2

NEAREST = 1
BILINEAR = 2
CUBIC = 3
CRESSMAN = 4
UNIFORM = 5
INVERSE = 6

ALL_WEIGHTS = 1
NO_ZERO_WEIGHTS = 2


def newSO(ipvol, opvol, aavg, ravg, maxelev):
    """
    Generates a new SO: Selects non-overlapping scans. Prepares output pvol
    and adds required attributes.

    Arguments:
      pyobject ipvol: input pvol
      pyobject opvol: output pvol (SO)
      int aavg: Azimuthal resolution [# azimuth gates] of the SO. Note: the
                total number of azimuth gates per scan (ysize) must be
                divisible by aavg.
      int ravg: Radial resolution [m] of the SO. Note: ravg must be divisible
                by the distance [m] betweeen two successive range bins
                (xscale). 
      int maxelev: Maximum elevation angle [degrees] used for SO production.

    Returns:
      pyobject ipvol: input pvol
      pyobject opvol: output pvol (SO)
    """
    
    if (ravg % ipvol.get('/where/xscale') == 0.0):
        ravg = int(ravg / ipvol.get('/where/xscale'))
    else:
        raise IOError("Invalid radial integration length.")

    if (ipvol.get('/where/ysize') % float(aavg) == 0.0) and \
       (ipvol.get('/where/xsize') % float(ravg) == 0.0):   
        beamwidth = ipvol.get('/how/beamwidth')
        oscan = []
        
        # Only select non-overlapping scans.
        iscan = ipvol.get('/how/scan')
        if ipvol.get('/scan%s/where/angle' % iscan[0]) > maxelev:
          raise IOError("Invalid elevation angles.")
        for s in range(len(iscan)):
          if iscan[s] == iscan[0]:
            oscan.append(iscan[s])
          elif abs(ipvol.get('/scan%s/where/angle' % iscan[s])-
                   ipvol.get('/scan%s/where/angle'%oscan[-1])) >= beamwidth/2.0 \
                   and ipvol.get('/scan%s/where/angle' % iscan[s]) <= maxelev:
            oscan.append(iscan[s])
        #oscan.sort()
        ipvol.set('/how/scan', oscan)

        # Prepare output volume.
        opvol.set('/where/ysize', ipvol.get('/where/ysize') / aavg)
        opvol.set('/where/xsize', ipvol.get('/where/xsize') / ravg)
        opvol.set('/where/xscale', ipvol.get('/where/xscale') * ravg) # must be in m!    

        # Add required info attributes
        opvol.set('/how/transform_weighting', NO_ZERO_WEIGHTS)    
        opvol.set('/how/i_method', UNIFORM) # imethod
        opvol.set('/how/scan', oscan)
        opvol.set('/how/elev_usage', USE_SINGLE_ELEV) # elevUsage
        opvol.set('/how/rs', [0.0] * opvol.get('/where/xsize')) # search radii

        A = DatasetArray(xsize=opvol.get('/where/xsize'),
                         ysize=opvol.get('/where/ysize'), \
                         typecode='d', initval=None)
        for i in oscan:
            opvol.set('/scan%s/data' % str(i), A)

        tmpfile = rave_tempfile.mktemp()
        os.close(tmpfile[0])  # tempfile.mkstemp() opens the file for us
        opvol.set('/how/tmpfile', tmpfile[1])

        return ipvol, opvol
    else:
        raise IOError("Invalid integration lengths.")


# New version of ptop which can manage both volumes at once.
def transform(iw, ow, iz, oz, aavg, ravg, maxelev):
    """
    The default transformation is UNIFORM (simple average). Other optional
    methods are NEAREST (nearest neighbour), BILINEAR, CUBIC, INVERSE
    (inverse-distance) and CRESSMAN (weights proximate bins more and distant
    bins less than INVERSE). The actual choice of two or three interpolation
    dimensions is regulated through the use of USE_SINGLE_ELEV (default) or
    USE_MANY_ELEV, respectively. Additionally, there is a choice between
    using NO_ZERO_WEIGHTS (default) and ALL_WEIGHTS.

    Arguments:
      pyobject iw: input pvol (radial wind)
      pyobject ow: output pvol (radial wind) -> SO.
      pyobject iz: input pvol (reflectivity)
      pyobject oz: output pvol (reflectivity) -> SO.
      int aavg: Azimuthal resolution [# azimuth gates] of the SO. Note: the
                total number of azimuth gates per scan (ysize) must be
                divisible by aavg.
      int ravg: Radial resolution [m] of the SO. Note: ravg must be divisible
                by the distance [m] betweeen two successive range bins
                (xscale). 
      int maxelev: Maximum elevation angle [degrees] used for SO production.

    Returns:
      pyobject ow: output pvol (radial wind) -> SO.
      pyobject oz: output pvol (reflectivity) -> SO.
    """
    
    iw, ow = newSO(iw, ow, aavg, ravg, maxelev)
    if iz != None:
        iz, oz = newSO(iz, oz, aavg, ravg, maxelev)
        _ptop.transform(iw, ow, iz, oz)
        return ow, oz
    else:
      _ptop.transform(iw, ow, None, None)

    return ow, None


# -----------------------------------------------------------------------------
# HELPER FUNCTION

def makeSO(fstr, ofstr, aavg, ravg, maxelev):

    """
    Prepares a SO: Opens the SO file and writes the main header. Extracts wind 
    and reflectivity scans from the HDF5 files and converts data quantity from
    dBZ to Z. 

    Arguments:
      string fstr: String of the HDF5 file to be used for SO production.
      string ofstr: SO output file name.
      int aavg: Azimuthal resolution [# azimuth gates] of the SO. Note: the
                total number of azimuth gates per scan (ysize) must be
                divisible by aavg.
      int ravg: Radial resolution [m] of the SO. Note: ravg must be divisible
                by the distance [m] betweeen two successive range bins
                (xscale). 
      int maxelev: Maximum elevation angle [degrees] used for SO production.

    Returns: Nothing if successful.
    """
    this = rave.open(fstr)

    # Extract wind and reflectivity scans
    DATE, TIME = this.get('/what/date'), this.get('/what/time')

    # Open the output superob file and write the main header.
    fd = open(ofstr, 'w')
    fd.write("SUPEROB %s %s\n" % (DATE, TIME[:4]))

    sets = this.get('/what/sets')
    iwscan, izscan = [], []

    # Read the same data four times, for input and output, and for Z and V
    iw = rave.open(fstr)
    ow = rave.open(fstr)
    iz = rave.open(fstr)
    oz = rave.open(fstr)

    for s in range(sets):

        # Ignore spectral width or whatever else may be there except these:
        if this.get('/scan%s/what/quantity' % (s+1)) == 'VRAD':
            iwscan.append(s+1)
        if this.get('/scan%s/what/quantity' % (s+1)) == 'DBZ':
            izscan.append(s+1)

    # Sort scan lists (low -> high elevation angles)
    iwangle, izangle, iwscans, izscans = [], [], [], []
    for s in range(len(iwscan)):
	    iwangle.append(this.get('/scan%s/where/angle' % iwscan[s]))
	    izangle.append(this.get('/scan%s/where/angle' % izscan[s]))
    iwangle.sort()
    izangle.sort()
    for a in range(len(iwangle)):
      for s in range(len(iwscan)):
        if iwangle[a]==this.get('/scan%s/where/angle' % iwscan[s]):
          iwscans.append(iwscan[s])
        if izangle[a]==this.get('/scan%s/where/angle' % izscan[s]):
          izscans.append(izscan[s])
    iwscan = iwscans
    izscan = izscans

    #iwscan.sort()
    #izscan.sort()

    iw.set('/how/scan', iwscan)
    iz.set('/how/scan', izscan)

    # save wind data as 'f'-type
    for s in range(len(iwscan)):
        scan = iw.get('/how/scan')[s]
        arrw = iw.get('/scan%s/data' % scan)
        iw.set('/scan%s/data' % scan, arrw.astype(float))

    # convert data quantity from dBZ to Z. Modify the object instead of
    # returning a new one. The lowest possible Z value is truncated to 0.
    for s in range(len(izscan)):
        scan = iz.get('/how/scan')[s]
        offset = iz.get('/scan%s/what/offset' % scan)
        gain = iz.get('/scan%s/what/gain' % scan)

        arrz = iz.get('/scan%s/data' % scan)
        arrz = offset + arrz * gain
        minz = minimum.reduce(arrz.flat)
        arrz = where(greater(arrz, minz), 10**(arrz/10.0), 0.0)
        iz.set('/scan%s/data' % scan, arrz.astype(float))

        nodata = iz.get('/scan%s/what/nodata' % scan)            
        nodata = offset + nodata * gain
        nodata = 10**(nodata/10.0)
        iz.set('/scan%s/what/nodata' % scan, nodata)

        iz.set('/scan%s/what/quantity' % scan, 'Z')
                    
    # Perform the transform.
    #ow, oz = transform(iw, ow, None, None, aavg, ravg, maxelev)        
    ow, oz = transform(iw, ow, iz, oz, aavg, ravg, maxelev)
                 
    # Open the resulting tmpfile and append its contents to the
    # superob file. Then delete the tmpfile.
    tmpfile = ow.get('/how/tmpfile')
    if os.path.isfile(tmpfile):
        tmpfd = open(tmpfile)
        contents = tmpfd.read()
        fd.write(contents)
        tmpfd.close()
        os.remove(tmpfile)
    fd.close()
    tmpfile = oz.get('/how/tmpfile')  # already closed
    if os.path.isfile(tmpfile):
        os.remove(tmpfile)


__all__ =  ['newSO', 'transform', 'makeSO']

if __name__ == "__main__":
    print(__doc__)
