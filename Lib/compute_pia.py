#!/usr/bin/env python3
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

''' Original settings
######################################
#settings
######################################
a         = 171;                                   # Coefficient of Z-R power law of X-band
b         = 1.73;                                  # Exponent of Z-R power law of X-band
c         = 9.25e4;                                # Coefficient of Z-k power law of X-band
d         = 1.25;                                  # Exponent of Z-k power law of X-band


c         = 7.34e5;                                # Coefficient of Z-k power law of C-band
d         = 1.344;                                 # Exponent of Z-k power law of C-band

dBZmin    = 10;                                    # Minimum radar reflectivity [dBZ]
dBZmax    = 55;                                    # Maximum radar reflectivity [dBZ]
Rmin      = 0.5; #((10^(0.1*dBZmin))/a)^(1/b);     # Minimum rain rate [mm/h]
Rmax      = 50;                                    # Maximum rain rate [mm/h]
PIAmin    = 0.1;                                   # Minimum path-integrated attenuation (PIA) [dB]
PIAmax    = 10;                                    # Maximum PIA in Hitschfeld-Bordan algorithm [dB]
dtheta    = 1.875 * np.pi / 180;                   # Azimuth resolution [rad]
dr        = 2;

'''

import numpy as np
import os
import _raveio
import _polarscan, _polarscanparam, _polarvolume, _ravefield

from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

######################################
#settings
######################################
CBAND_CO_ZK_POWER = 7.34e5                                   # Coefficient of Z-k power law of C-band
CBAND_EXP_ZK_POWER = 1.344                                   # Exponent of Z-k power law of C-band

XBAND_CO_ZK_POWER = 9.25e4                                   # Coefficient of Z-k power law of C-band
XBAND_EXP_ZK_POWER = 1.25                                   # Exponent of Z-k power law of C-band

PIA_MAX = 10.0;                                    # Maximum PIA in Hitschfeld-Bordan algorithm [dB]
DEFAULT_RANGE_RESOLUTION = 2.0;                                     # Range resolution [km]

TASK = "se.smhi.qc.hitschfeld-bordan"

UNDETECT_TYPE=0
NODATA_TYPE=1
DATA_TYPE=2

class PIAHitschfeldBordan:
    def __init__(self, param_name="DBZH", co_zk_power=CBAND_CO_ZK_POWER, exp_zk_power=CBAND_EXP_ZK_POWER, PIAMax = PIA_MAX):
        """Constructor
        :param_name: the quantity to process in the scan. Only allows DBZH and TH
        :param co_zk_power: The coefficient of the Z-K power law. Default to C_BAND value
        :param exp_zk_power: The exponent of the Z-K power law. Default to C_BAND value
        :param PIAMax: Maximum PIA adjustment. Defaults to PIA_MAX.
        """
        if param_name not in ["DBZH", "TH"]:
            raise AttributeError("Method only supports DBZH and/or TH when deriving PIA")
        self._param_name = param_name
        self._co_zk_power = co_zk_power
        self._exp_zk_power = exp_zk_power
        self._piamax = PIAMax

    def createPIAField(self, scan):
        """ Calculates the PIA field according to the Hitschfeld-Bordan algorithm.
        The range resolution is taken from the scans rscale and converted to km. The algorithm limits the
        adjustment by ensuring that the log10 call never will be called with a negative value by limiting the cumulative
        values by the inverse of the adjustment factor.
        :param scan: Scan to operate on
        :return: a tuple of PIA field, datatype field (0 = Undetect, 1 = Nodata and 2 = Data) and the range resolution used
        """
        if not scan.hasParameter(self._param_name):
            raise AttributeError("Scan does not contain wanted parameter %s"%self._param_name)

        dr = scan.rscale / 1000.0  # m -> km
        if dr == 0.0:
            dr = DEFAULT_RANGE_RESOLUTION

        kacumsum_factor = 0.2 * dr  * np.log (10) / self._exp_zk_power
        kacumsum_limit = 1 / kacumsum_factor   # To avoid NaN when calculating PIA (i.e. not any calls to log10 with negative values)

        dbzh_param = scan.getParameter(self._param_name)
        gain = dbzh_param.gain
        offset = dbzh_param.offset
        nodata = dbzh_param.nodata
        undetect = dbzh_param.undetect

        # Get dbzh and ensure we have control over where undetect and nodata are
        raw_data = dbzh_param.getData().astype(np.float64)
        datatypes = np.where(raw_data==undetect, UNDETECT_TYPE, DATA_TYPE)   # Datatypes, 0 = Undetect, 1 = Nodata and 2 = Data
        datatypes[raw_data==nodata] = NODATA_TYPE

        dbzh_data = raw_data * gain + offset  # Don't care about including undetect and nodata, it will be handled later on.

        ##  ka        = ((10 ** (0.1 * raw_data)) / c) ** (1 / d)   # Apparent specific attenuation [dB/km]
        ##  PIA       = -10 * d * np.log10 (1 - 0.2 * np.log (10) / d * np.cumsum (ka, 1) * dr)
        ka = np.where(datatypes==DATA_TYPE, ((10**(0.1 * (dbzh_data))) / self._co_zk_power) ** (1 / self._exp_zk_power), 0)    # Apparent specific attenuation [dB/km]
        kacumsum = np.cumsum(ka, 1)
        kacumsum[kacumsum > kacumsum_limit] = kacumsum_limit-(1e-06)  # Ensure that PIA calculate does not explode

        PIA = -10 * self._exp_zk_power * np.log10(1 - kacumsum_factor * kacumsum)

        PIA[PIA > self._piamax] = self._piamax

        return PIA, datatypes, dr        

    def createPIAParameter(self, scanparam, PIA_field):
        """ Creates a PIA parameter from the PIA field and the datatypes
        :param scanparam: The parameter that was used when creating the PIA field
        :param PIA_field: the PIA field without gain/offset applied
        :return the PIA parameter
        """
        result = _polarscanparam.new()
        result.quantity = "PIA"
        result.gain = scanparam.gain
        result.offset = scanparam.offset
        result.nodata = scanparam.nodata                # Might be better to set both nodata and undetect to 0.0 since the PIA field always is calculated and if don't
                                                        # know what to do with a value with should probably add 0.0 anyway.
        result.undetect = scanparam.undetect
        data = PIA_field / result.gain - result.offset
        result.setData(data)
        return result

    def createPIAQualityField(self, pia_parameter, dr):
        """ Creates a PIA quality field from the PIA parameter.
        :param pia_parameter: The PIA parameter created
        :param dr: the range resolution in km
        """
        qfield = pia_parameter.toField()
        qfield.addAttribute("how/task", TASK)
        qfield.addAttribute("how/task_args", self.createHowTaskArgs(self._co_zk_power, self._exp_zk_power, self._piamax, dr))
        return qfield

    def process(self, scan, reprocess_quality_flag, quality_control_mode):
        """ Creates the PIA derived parameter and corresponding quality field in the scan. If quality_control_mode is set
        to APPLY it will also adjust the parameter referenced by param_name with the PIA field.
        :param scan: scan to change
        :reprocess_quality_flag: if quality flag should be reprocessed or not
        :quality_control_mode: how to handle the quality control mode if only analyzing or analyze and apply
        :return: N/A
        """
        if not scan.hasParameter(self._param_name):
            raise AttributeError("Scan does not contain wanted parameter %s"%self._param_name)

        PIA_field, datatypes, dr = self.createPIAField(scan)

        PIA_parameter = self.createPIAParameter(scan.getParameter(self._param_name), PIA_field)
        scan.addParameter(PIA_parameter)

        if reprocess_quality_flag or not scan.findQualityFieldByHowTask(TASK):
            qfield = self.createPIAQualityField(PIA_parameter, dr)
            scan.addOrReplaceQualityField(qfield)

        if quality_control_mode==QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY:
            parameter = scan.getParameter(self._param_name)
            data = parameter.getData()
            newdata = np.where(datatypes==DATA_TYPE, (data*parameter.gain + parameter.offset + PIA_field)/parameter.gain - parameter.offset, data)
            parameter.setData(newdata)

    def createHowTaskArgs(self, co_zk_power, exp_zk_power, PIAMax, dr):
        """ Creates the how/task_args string that should be associated with the PIA fields
        :param co_zk_power: The coefficient of the Z-K power law. Default to C_BAND value
        :param exp_zk_power: The exponent of the Z-K power law. Default to C_BAND value
        :param PIAMax: Maximum PIA adjustment. Defaults to PIA_MAX.
        :param dr: the range resolution in km
        :return: the string
        """
        return "param_name=PIA c_ZK=%2.2f d_ZK=%2.2f PIAmax=%2.1f dr=%g"%(co_zk_power, exp_zk_power, PIAMax, dr)

## Computes PIA based quality control. A single scan object is
#  sent to the ComputePIAscan function. The scans comprising a polar volume are
#  sent to the same function individually.
# @param PolarScanCore object
# @param string radar quantity name, defaults to DBZH
# @param boolean whether (True) or not (False) to keep the derived PIA parameter
# @param boolean reprocess_quality_flag
# @param enum quality_control_mode
def ComputePIAscan(scan, param_name, keepPIA, reprocess_quality_flag, quality_control_mode):
    pia_added = False
    if not scan.hasParameter("PIA"):
        try:
            PIAHitschfeldBordan(param_name).process(scan, reprocess_quality_flag, quality_control_mode)
            PIA = scan.getParameter("PIA")
            PIA.addAttribute("how/task", TASK)
            task_args = scan.findQualityFieldByHowTask(TASK).getAttribute("how/task_args")
            PIA.addAttribute("how/task_args", task_args)
            pia_added = True
        except:
            pass
    
    if not keepPIA and pia_added:
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
