'''
Copyright (C) 2019 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE and this software are distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
##
#  Python-level functionality for performing DRQC


## 
# @file
# @author Daniel Michelson, Environment and Climate Change Canada
# @date 2019-04-26

import copy
import _rave, _raveio
import _polarvolume, _polarscan
import _dr_qc
import ec_temperatureProfile

TASK = "ca.mcgill.qc.depolarization_ratio"
targsfmt = "param_name=%s zdr_offset=%2.2f kernely=%i kernelx=%i param_thresh=%2.1f dr_thresh=%2.1f"


## 
#  Angles are in radians
def getBeamTopBottomHeights(scan):
  elangle = copy.deepcopy(scan.elangle)
  half_bw = scan.beamwidth / 2.0
  scan.elangle = elangle + half_bw
  beamtop = scan.getHeightField().getData()[0]
  scan.elangle = elangle - half_bw
  beambot = scan.getHeightField().getData()[0]
  scan.elangle = elangle  # revert
  scan.addAttribute('how/beamtop', beamtop)
  scan.addAttribute('how/beambot', beambot)


## Performs depolarization ratio based quality control on a polar volume or
#  polar scan. This involves deriving the depolarization ratio (DR) parameter
#  if it doesn't already exist, and then speckle filtering the radar observable
#  with it. DR is based on RHOHV and ZDR. The speckle filter derives the 
#  relative proportions of "precipitation", "non-meteorological", and "no echo" 
#  (undetect), and the centre bin in the polar kernel is assigned the class that
#  is in majority. If that class is "precipitation", then the original radar 
#  observable is kept, otherwise it is assigned the "undetect" value.
# @param PolarScanCore object
# @param string radar quantity name, defaults to DBZH
# @param Float ZDR offset in dB. Defaults to 0.0.
# @param int number of polar azimuth rays around the centre bin in the kernel 
# used with the speckle filter. Defaults to 2.
# @param int number of polar range bins around the centre bin in the kernel 
# used with the speckle filter. Defaults to 2.
# @param Float threshold above which the radar quantity will not be touched. 
# Defaults to 35 assuming the quantity is dBZ.
# @param Float threshold below which DR is considered to represent 
# precipitation. Defaults to -12 dB.
# @param boolean whether to use Tw=True or Td=False
# @param boolean whether (True) or not (False) to keep the derived DR parameter
def drQCscan(scan, profile=None, param_name="DBZH", zdr_offset=0.0,
             kernely=2, kernelx=2,
             param_thresh=35.0, dr_thresh=-12.0, Tw=True, keepDR=True):

    # If we have a temperature profile, set some attributes. Note that we can't
    # remove them later at the Python level, but we can at the C level.
    if profile:
        rtempc = ec_temperatureProfile.getTempcProfile(scan, profile, Tw)
        scan.addAttribute('how/tempc', rtempc)
        if Tw: bb_topT, bb_botT = BB["Tw"]
        else:  bb_botT, bb_botT = BB["Td"]
        scan.addAttribute('how/bb_topT', bb_top)
        scan.addAttribute('how/bb_botT', bb_bot)
        getBeamTopBottomHeights(scan)
    
    # Create DR parameter if it's not already there
    if not scan.hasParameter("DR"):
        try:
            _dr_qc.drDeriveParameter(scan, zdr_offset)
            DR = scan.getParameter("DR")
            if keepDR:
                DR.addAttribute('how/task', TASK)
                targs = targsfmt % (param_name, zdr_offset, kernely, kernelx, 
                                    param_thresh, dr_thresh)
                DR.addAttribute('how/task_args', targs)
        except AttributeError:
            pass

    # Then speckle filter the parameter with it
    try:
        _dr_qc.drSpeckleFilter(scan, param_name, kernely, kernelx, 
                               param_thresh, dr_thresh)
    except AttributeError:
        pass

    if not keepDR: scan.removeParameter("DR")      


## Manages depolarization ratio based quality control. A single scan object is
#  sent to the \ref drQCscan function. The scans comprising a polar volume are
#  sent to the same function individually.
# @param PolarVolumeCore or PolarScanCore object
# @param string radar quantity name, defaults to DBZH
# @param Float ZDR offset in dB. Defaults to 0.0.
# @param int number of polar azimuth rays around the centre bin in the kernel 
# used with the speckle filter. Defaults to 2.
# @param int number of polar range bins around the centre bin in the kernel 
# used with the speckle filter. Defaults to 2.
# @param Float threshold above which the radar quantity will not be touched. 
# Defaults to 35 assuming the quantity is dBZ.
# @param Float threshold below which DR is considered to represent 
# precipitation. Defaults to -12 dB.
# @param boolean whether to use Tw=True or Td=False
# @param boolean whether (True) or not (False) to keep the derived DR parameter
def drQC(pobject, profile_fstr=None, param_name="DBZH",
         zdr_offset=0.0,
         kernely=2, kernelx=2, 
         param_thresh=35.0, dr_thresh=-12.0, Tw=True, keepDR=True):

    if profile_fstr: profile = ec_temperatureProfile.readProfile(profile_fstr)
    else: profile = None
    
    if _polarvolume.isPolarVolume(pobject):
        nscans = pobject.getNumberOfScans(pobject)
        for n in range(nscans):
            scan = pobject.getScan(n)
            drQCscan(scan, profile, param_name, zdr_offset, kernely, kernelx, 
                     param_thresh, dr_thresh, Tw, keepDR)

    elif _polarscan.isPolarScan(pobject):
        drQCscan(pobject, profile, param_name, zdr_offset, kernely, kernelx, 
                 param_thresh, dr_thresh, Tw, keepDR)

    else:
        raise IOError("Input object is neither polar volume nor scan")
        


if __name__=="__main__":
    pass
