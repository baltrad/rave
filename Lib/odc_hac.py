#!/usr/bin/env python
'''
Copyright (C) 2013- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## Performs hit-accumulation clutter filtering using hit-accumulation
#  monthly "climatologies" or "counter files".
#  Added also Z-diff quality indicator.

## @file
## @author Daniel Michelson, SMHI
## @date 2013-01-14

import sys, os, time, glob, types
import multiprocessing
import _raveio, _ravefield
import _polarvolume, _polarscan
import _pyhl, _odc_hac
import rave_defines
import odim_source
from Proj import rd
from numpy import zeros, uint8, uint32
import xml.etree.ElementTree as ET


HACDATA = rave_defines.RAVEROOT + '/share/hac/data'
CONFIG_FILE = rave_defines.RAVECONFIG + '/hac_options.xml'

initialized = 0
ARGS = {}

## Initializes the ARGS dictionary by reading config from XML file
def init():
    global initialized
    if initialized: return
    
    C = ET.parse(CONFIG_FILE)
    OPTIONS = C.getroot()
    
    for site in list(OPTIONS):
        hac = HAC()
        
        for k in site.attrib.keys():
            if   k == "threshold": hac.thresh = float(site.attrib[k])

        ARGS[site.tag] = hac                
    initialized = 1


class HAC:
    def __init__(self):
        self.hac = None
        self.thresh = None


    ## Creates a HAC. Should be called only after a failed call to \ref readHac
    # @param fstr file string
    # @param nrays int number of rays in the scan
    # @param nbins int number of bins per ray
    def makeHac(self, fstr, nrays, nbins):
        if not os.path.isfile(fstr):
            self.hac = _ravefield.new()
            self.hac.addAttribute("how/count", 0)
            self.hac.setData(zeros((nrays, nbins), uint32))
        else:
            raise IOError, "HAC file already exists: %s" % fstr


    ## Reads a HAC HDF5 file and returns the dataset in it.
    # @param fstr file string
    def readHac(self, fstr):
        if os.path.isfile(fstr):
            nodelist = _pyhl.read_nodelist(fstr)
            nodelist.selectNode("/accumulation_count")
            nodelist.selectNode("/hit_accum")
            nodelist.fetch()

            self.hac = _ravefield.new()
            self.hac.addAttribute("how/count",
                                  nodelist.getNode("/accumulation_count").data())
            self.hac.setData(nodelist.getNode("/hit_accum").data())
        else:
            raise IOError, "No such HAC file: %s" % fstr


    ## Writes a HAC to HDF5.
    # @param fstr file string
    # @param compression int ZLIB compression level
    def writeHac(self, fstr, compression=0):
        nodelist = _pyhl.nodelist()

        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/accumulation_count")
        node.setScalarValue(-1,self.hac.getAttribute("how/count"),"long",-1)
        nodelist.addNode(node)

        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/validity_time_of_last_update")
        node.setScalarValue(-1,int(time.time()),"long",-1)
        nodelist.addNode(node)

        node = _pyhl.node(_pyhl.DATASET_ID, "/hit_accum")
        node.setArrayValue(-1,[self.hac.ysize, self.hac.xsize],
                           self.hac.getData(),"uint",-1)
        nodelist.addNode(node)

        fcp = _pyhl.filecreationproperty()
        fcp.userblock = 0
        fcp.sizes = (4,4)
        fcp.sym_k = (1,1)
        fcp.istore_k = 1
        fcp.meta_block_size = 0

        path = os.path.split(fstr)[0]
        if not os.path.isdir(path):
            os.makedirs(path)
        nodelist.write(fstr, compression, fcp)


    ## Performs the filtering
    # @param scan input SCAN object
    # @param param string of the quantity to filter
    # @param enough int lower threshold of the number of hits to accept in order to process
    def hacFilter(self, scan, quant="DBZH", enough=100):
        NOD = odim_source.NODfromSource(scan)

        # If HAC files are missing, then this method will passively fail.
        try:
            self.readHac(hacFile(scan, lastmonth=True))

            if self.hac.getAttribute("how/count") < enough:
                raise ValueError, "Not enough hits in climatology for %s" % NOD

            hac_data = self.hac.getData()
            if hac_data.shape != (scan.nrays, scan.nbins):
                print hac_data.shape, (scan.nrays, scan.nbins)
                raise IOError, "Scan and HAC have different geometries for %s" % NOD

            ## Get site-specific threshold!
            try:
                self.thresh = ARGS[NOD].thresh
            except KeyError:
                self.thresh = ARGS["default"].thresh
            ## Got site-specific threshold?

            qind = _ravefield.new()
            qind.setData(zeros(hac_data.shape, uint8))
            qind.addAttribute("how/task", "eu.opera.odc.hac")
            qind.addAttribute("how/task_args", self.thresh)
            scan.addQualityField(qind)

            _odc_hac.hacFilter(scan, self.hac, quant)
        except IOError:
            pass


    ## Increments the HAC with the hits in the current scan.
    # @param scan input SCAN object
    # @param param string of the quantity to filter
    def hacIncrement(self, scan, quant="DBZH"):
        NOD = odim_source.NODfromSource(scan)
        hacfile = hacFile(scan)

        try:
            try:
                self.readHac(hacfile)
            except IOError:
                self.makeHac(hacfile, scan.nrays, scan.nbins)

            hac_data = self.hac.getData()
            if hac_data.shape != (scan.nrays, scan.nbins):
                print hac_data.shape, (scan.nrays, scan.nbins)
                raise IOError, "Scan and HAC have different geometries for %s" % NOD

            _odc_hac.hacIncrement(scan, self.hac, quant)

            self.writeHac(hacfile)
        except IOError:
            pass


## Convenience functions

## Takes a year-month string and returns the previous month's equivalent string.
# @param YYYYMM year-month string
# @returns year-month string
def lastMonth(YYYYMM):
    tt = (int(YYYYMM[:4]), int(YYYYMM[4:6])-1, 1,0,0,0,0,0,-1)
    newtt = time.localtime(time.mktime(tt))
    return time.strftime("%Y%m", newtt)
    

## Derives a file string from the input object.
# @param scan that must be an individual SCAN. This SCAN's
# /what/source must contain a valid NOD identifier.
# @param lastmonth boolean specifying whether to read the previous month's file.
# @returns string file string
def hacFile(scan, lastmonth=False):
    NOD = odim_source.NODfromSource(scan)
    CCCC = odim_source.CCCC[NOD]
    RAD = odim_source.RAD[NOD][2:]
    elangle = str(int(round(scan.elangle * rd * 10)*10)).zfill(5)
    rays = str(scan.nrays).zfill(4)
    bins = str(scan.nbins).zfill(4)

    YYYYMM = scan.date[:6]
    if lastmonth == True:
        YYYYMM = lastMonth(YYYYMM)

    return HACDATA + "/%s_%s_%s_%s_%sx%s_hit-accum.hdf" % (YYYYMM, CCCC,
                                                           RAD, elangle,
                                                           rays, bins)


## Increments the HAC file(s) for the given object
# @param obj input SCAN or PVOL, can also be a file string
def hacIncrement(obj, quant="DBZH"):
    if _polarvolume.isPolarVolume(obj):
        incrementPvol(obj, quant)
    elif _polarscan.isPolarScan(obj):
        incrementScan(obj, quant)
    elif type(obj) == types.StringType:
        if os.path.isfile(obj) and os.path.getsize(obj):
            obj = _raveio.open(obj).object
            hacIncrement(obj)
        else:
            raise TypeError, "HAC incrementor received a string without a matching file, or file is empty"
    else:
        raise TypeError, "HAC incrementor received neither SCAN nor PVOL as input object"
    

## Increments the HAC file for this scan. We will assume we only want to deal with DBZH.
# @param scan polar scan object
def incrementScan(scan, quant="DBZH"):
    hac = HAC()
    hac.hacIncrement(scan, quant)


## Increments all the HAC files for the scans in a volume, assuming we only wanty to deal with DBZH.
# @param pvol polar volume object
def incrementPvol(pvol, quant="DBZH"):
    for i in range(pvol.getNumberOfScans()):
        scan = pvol.getScan(i)
        incrementScan(scan, quant)


## Filters the given object
# @param obj input SCAN or PVOL
def hacFilter(obj, quant="DBZH"):
    if _polarvolume.isPolarVolume(obj):
        filterPvol(obj, quant)
    elif _polarscan.isPolarScan(obj):
        filterScan(obj, quant)
    else:
        raise TypeError, "HAC filter received neither SCAN nor PVOL as input"


## Filters this scan. We will assume we only want to deal with DBZH.
# @param scan polar scan object
def filterScan(scan, quant="DBZH"):
    hac = HAC()
    hac.hacFilter(scan, quant)


## Filters this scan. We will assume we only want to deal with DBZH.
# @param scan polar scan object
def filterPvol(pvol, quant="DBZH"):
    hac = HAC()
    for i in range(pvol.getNumberOfScans()):
        scan = pvol.getScan(i)
        hac.hacFilter(scan, quant)


## Multiprocesses the incrementation
# @param fstrs list of input file strings
# @param procs int number of concurrent processes, defaults to the max allowed
# @return list of returned tuples from \ref hacIncrement
def multi_increment(fstrs, procs=None):
    pool = multiprocessing.Pool(procs)

    results = []
    r = pool.map_async(hacIncrement, fstrs, chunksize=1)
    r.wait()


## Odds and ends below

## Z-diff quality indicator. Takes the difference between uncorrected and corrected reflectivities
#  and derives a quality indicator out of it. The threshold is the maximum difference in dBZ
#  giving the equivalent of zero quality.
# @param scan Polar scan 
# @param thresh float maximum Z-diff allowed 
def zdiffScan(scan, thresh=40.0):
    if _polarscan.isPolarScan(scan):
        if not scan.hasParameter("DBZH") or not scan.hasParameter("TH"):
          return 
        qind = _ravefield.new()
        qind.setData(zeros((scan.nrays,scan.nbins), uint8))
        qind.addAttribute("how/task", "eu.opera.odc.zdiff")
        qind.addAttribute("how/task_args", thresh)
        qind.addAttribute("what/gain", 1/255.0)
        qind.addAttribute("what/offset", 0.0)
        scan.addQualityField(qind)

        ret = _odc_hac.zdiff(scan, thresh)
    else:
        raise TypeError, "Input is expected to be a polar scan. Got something else."

    
def zdiffPvol(pvol, thresh=40.0):
    if _polarvolume.isPolarVolume(pvol):
        for i in range(pvol.getNumberOfScans()):
            scan = pvol.getScan(i)
            zdiffScan(scan, thresh)
    else:
        raise TypeError, "Input is expected to be a polar volume. Got something else."

def zdiff(obj, thresh=40.0):
  if _polarscan.isPolarScan(obj):
    zdiffScan(obj, thresh)
  elif _polarvolume.isPolarVolume(obj):
    zdiffPvol(obj, thresh)
  else:
    raise TypeError, "Input is expected to be a polar volume or scan" 

## Initialize
init()


if __name__ == "__main__":
    pass
