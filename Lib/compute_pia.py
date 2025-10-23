#!/usr/bin/env python
'''
Copyright (C) 2025- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

Created on Mon Aug 25 14:12:08 2025

@author: Remco van de beek
@author: Yngve Einarsson SMHI
'''

import numpy as np
import os
import _raveio
import _polarscan, _polarscanparam, _polarvolume, _ravefield

from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY


######################################
#settings
######################################
co_zk_power         = 7.34e5;                                # Coefficient of Z-k power law of C-band
exp_zk_power         = 1.344;                                 # Exponent of Z-k power law of C-band

dBZmin    = 10;                                    # Minimum radar reflectivity [dBZ]
dBZmax    = 55;                                    # Maximum radar reflectivity [dBZ]
Rmin      = 0.5; #((10^(0.1*dBZmin))/a)^(1/b);     # Minimum rain rate [mm/h]
Rmax      = 50;                                    # Maximum rain rate [mm/h]
PIAmin    = 0.1;                                   # Minimum path-integrated attenuation (PIA) [dB]
PIAmax    = 10;                                    # Maximum PIA in Hitschfeld-Bordan algorithm [dB]
dtheta    = 1.875 * np.pi / 180;                   # Azimuth resolution [rad]
dr        = 2;                                     # Range resolution [km]

TASK = "se.smhi.qc.hitschfeld-bordan"
targsfmt = "param_name=%s c_ZK=%2.2f d_ZK=%2.2f dBZmin=%i dbZmax=%i Rmin=%2.1f Rmax=%2.1f PIAmin=%2.1f PIAmax=%2.1f dtheta=%2.1f dr=%i"

# Computes PIA based on Hitschfeld-Bordan algorithm. A single scan object is
#  sent to the PIADeriveParameter function.
# @param PolarScanCore object
# @param string radar quantity name, defaults to DBZH
# @param boolean reprocess_quality_flag
# @param enum quality_control_mode

def PIADeriveParameter(scan, param_name,reprocess_quality_flag, quality_control_mode):

    DBZH = scan.getParameter(param_name)
    gain = DBZH.gain
    offset = DBZH.offset
    nodata = -32767.0
    # FIXME: Default error handling used for numpy?
    # np.seterr(all='raise')
    raw_data = np.array(DBZH.getData())
    raw_data=raw_data.astype(float) #convert from int to float, so nan is valid value
    # raw_data = np.where(raw_data != nodata, offset+gain*raw_data, -327.67)
    # No need to take special care for nodata vill be converted back to -32767.0 later on.
    raw_data=offset+gain*raw_data #convert to dBZ
    dBZ=(raw_data)
    ka        = ((10.0 ** (0.1*raw_data)) / co_zk_power) ** (1.0 / exp_zk_power)   # Apparent specific attenuation [dB/km]
    # Check these to aviod RuntimeWarning: invalid value encountered in log10.
    log10_in  = 1.0 - 0.2 * np.log (10.0) / exp_zk_power * np.cumsum (ka, 1) * dr
    # replace invalid values for log10_in
    log10_in = np.where(log10_in < 0.1, 0.1, log10_in)
    PIA       = -10.0 * exp_zk_power * np.log10 (log10_in)

    #Make certain no values above PIAmax are present and set nans due to exploding algorithm to PIAmax as well.
    PIA=np.nan_to_num(PIA,nan=10)
    PIA[PIA>PIAmax]=PIAmax
    # Convert back
    PIA_DBZH = PIA/gain - offset
    param = _polarscanparam.new()
    param.quantity = "PIA"
    param.gain = gain
    param.offset = offset
    param.nodata = DBZH.nodata
    param.undetect = DBZH.undetect
    param.setData(PIA_DBZH)
    scan.addParameter(param)

    if reprocess_quality_flag or not scan.findQualityFieldByHowTask(TASK):
        piaf = _ravefield.new()
        piaf.setData(PIA_DBZH)
        piaf.addAttribute("how/task", TASK)
        targs = targsfmt % ("PIA", co_zk_power, exp_zk_power, dBZmin, dBZmax, Rmin, Rmax, PIAmin, PIAmax, dtheta, dr)
        piaf.addAttribute("how/task_args", targs)
        piaf.addAttribute("what/gain", gain)
        piaf.addAttribute("what/offset", offset)
        scan.addOrReplaceQualityField(piaf)

    #apply PIA
    if quality_control_mode==QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY:
        dBZ_pia=PIA+dBZ
        param_pia = _polarscanparam.new()
        param_pia.quantity = "DBZH"
        param_pia.gain = gain
        param_pia.offset = offset
        param_pia.nodata = DBZH.nodata
        param_pia.undetect = DBZH.undetect
        # Convert back to DBZH
        dbzh_pia = dBZ_pia/gain - offset
        param_pia.setData(dbzh_pia)
        scan.removeParameter("DBZH")
        scan.addParameter(param_pia)

## Computes PIA based quality control. A single scan object is
#  sent to the ComputePIAscan function. The scans comprising a polar volume are
#  sent to the same function individually.
# @param PolarScanCore object
# @param string radar quantity name, defaults to DBZH
# @param boolean whether (True) or not (False) to keep the derived PIA parameter
# @param boolean reprocess_quality_flag
# @param enum quality_control_mode


def ComputePIAscan(scan, param_name, keepPIA, reprocess_quality_flag, quality_control_mode):


    # Create PIA parameter if it's not already there
    if not scan.hasParameter("PIA"):
        try:
            PIADeriveParameter(scan,param_name,reprocess_quality_flag,quality_control_mode)
            PIA = scan.getParameter("PIA")
            if keepPIA:
                PIA.addAttribute("how/task", TASK)
                targs = targsfmt % ("PIA", co_zk_power, exp_zk_power, dBZmin, dBZmax, Rmin, Rmax, PIAmin, PIAmax, dtheta, dr)
                PIA.addAttribute("how/task_args", targs)
        except AttributeError:
            pass

    if not keepPIA:
        if scan.hasParameter("PIA"):
            scan.removeParameter("PIA")

## Computes PIA based quality control. A single scan object is
#  sent to the ComputePIA function. The scans comprising a polar volume are
#  sent to the same function individually.
# @param PolarVolumeCore or PolarScanCore object
# @param string radar quantity name, defaults to DBZH
# @param boolean whether (True) or not (False) to keep the derived PIA parameter
# @param boolean reprocess_quality_flag
# @param enum quality_control_mode
def ComputePIA(pobject, param_name, keepPIA, reprocess_quality_flag, quality_control_mode):

    if _polarvolume.isPolarVolume(pobject):
        nscans = pobject.getNumberOfScans(pobject)
        for n in range(nscans):
            scan = pobject.getScan(n)
            ComputePIAscan(scan, param_name, keepPIA, reprocess_quality_flag, quality_control_mode)

    elif _polarscan.isPolarScan(pobject):
        ComputePIAscan(pobject, param_name, keepPIA, reprocess_quality_flag, quality_control_mode)

    else:
        raise IOError("Input object is neither polar volume nor scan")


## Computes PIA and performs quality control with it.
# @param object containing command-line arguments
def main(options):
    rio = _raveio.open(options.ifile)
    ComputePIA(rio.object, options.param_name, options.keepPIA, options.reprocess_quality_fields,
        options.quality_control_mode)
    rio.save(options.ofile)

if __name__=="__main__":
    from optparse import OptionParser

    description = "Computes PIA and uses it to quality control radar observations"

    usage = "usage: %prog -i <input file> -o <output file> [-p <parameter> -k <store PIA >] [h]"

    parser = OptionParser(usage=usage, description=description)

    parser.add_option("-i", "--infile", dest="ifile",
                      help="Name of input file to quality control. Must contain a polar volume or scan with ZDR and RHOHV besides the parameter to QC.")

    parser.add_option("-o", "--outfile", dest="ofile",
                      help="Name of output file to write.")

    parser.add_option("-p", "--parameter", dest="param_name", default="DBZH",
                      help="ODIM quantity/parameter name to QC. Defaults to DBZH.")

    parser.add_option("-k", "--keepPIA", dest="keepPIA", default="store_true",
                      help="Whether (True) or not (False) to keep and store the PIA parameter. Defaults to True.")

    parser.add_option(
        "--reprocess_quality_fields",
        action="store_true",
        dest="reprocess_quality_fields",
        help="Reprocessed the quality fields even if they already exist in the object.",
    )

    parser.add_option(
        "--quality_control_mode",
        dest="quality_control_mode",
        default=QUALITY_CONTROL_MODE_ANALYZE,
        help="Quality_control_mode, default QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY ."
    )

    (options, args) = parser.parse_args()

    if not options.ifile or not options.ofile:
        parser.print_help()
        sys.exit()


    main(options)
