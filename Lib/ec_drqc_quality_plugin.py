'''
Copyright (C) 2019 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

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

'''
##
# Plugin for performing quality control using depolarization ratio.
#

## 
# @file
# @author Daniel Michelson, Environment and Climate Change Cananda
# @date 2019-04-26

import _rave, _raveio, _polarvolume
import ec_drqc

from rave_quality_plugin import rave_quality_plugin


class ec_drqc_quality_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(ec_drqc_quality_plugin, self).__init__()
        self._option_file = None # Mostly for test purpose
    
    ##
    # @return a list containing the appropriate string
    def getQualityFields(self):
        return ["ca.mcgill.qc.depolarization_ratio"]

    ##
    # @param PolarVolumeCore or PolarScanCore object
    # @param boolean Not used here.
    # @param Not used here.
    # @return: obj - Filtered input object with associated metadata attributes
    def process(self, obj, reprocess_quality_flag=True, arguments=None):
        #_rave.setDebugLevel(_rave.Debug_RAVE_DEBUG)

        ec_drqc.drQC(obj)
    
        return obj

    ##
    # @return: placeholder
    #
    def algorithm(self):
        return None
  
  
if __name__=="__main__":
    a = ec_drqc_quality_plugin()
