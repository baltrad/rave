'''
Copyright (C) 2012- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of the bRopo extension to RAVE.

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
# A quality plugin for enabling support for dealiasing.
# The plugin will leave the original field unaffected and add a new 
# field where the dealiasing has been applied. This is done for all 
# quality control modes, i.e., both for "analyze" and "analyze & apply".

## 
# @file
# @author Daniel Michelson, SMHI
# @date 2013-10-07

from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
import rave_pgf_logger
import _polarscan
logger = rave_pgf_logger.create_logger()

# hardcoded here to True for now - make it controllable when/if needed
CREATE_SEPARATE_DEALIAS_PARAM = True

QUANTITY_CONVERSION_MAP = {"VRAD" : "VRADDH", "VRADH" : "VRADDH", "VRADV" : "VRADDV"} 

class dealias_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(dealias_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.dealias
  def getQualityFields(self):
    return ["se.smhi.detector.dealias"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @param reprocess_quality_flag: Not used
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY, arguments=None):
    try:
      import _dealias
      if CREATE_SEPARATE_DEALIAS_PARAM:
        add_dealiased_param(obj)
      else:
        _dealias.dealias(obj)
    except:
      logger.exception("Failure during dealias processing")
      
    return obj, self.getQualityFields()
  
def add_dealiased_param(obj):
  import _polarvolume
  if obj != None and _polarvolume.isPolarVolume(obj):
    for i in range(obj.getNumberOfScans()):
      scan = obj.getScan(i)
      add_dealiased_param_for_scan(scan)
  elif obj != None and _polarscan.isPolarScan(obj):
    add_dealiased_param_for_scan(obj)

def add_dealiased_param_for_scan(scan):
  try:
    import _dealias
    logger.debug("Adding dealiased parameter to scan.")
    for original_quantity in QUANTITY_CONVERSION_MAP.keys():
      if scan.hasParameter(original_quantity):
        new_quantity = QUANTITY_CONVERSION_MAP.get(original_quantity)
        param = _dealias.create_dealiased_parameter(scan, original_quantity, new_quantity)
        scan.addParameter(param)
  except:
    logger.exception("Failure during process of adding dealiased parameter")

