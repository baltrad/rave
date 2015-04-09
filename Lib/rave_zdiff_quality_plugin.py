'''
Copyright (C) 2015- Swedish Meteorological and Hydrological Institute (SMHI)

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
# A quality plugin for enabling zdiff qc support 

## 
# @file
# @author Anders Henja, SMHI
# @date 2015-04-09

from rave_quality_plugin import rave_quality_plugin
import odc_hac
import rave_pgf_logger
import _polarvolume, _polarscan

logger = rave_pgf_logger.rave_pgf_syslog_client()

class rave_zdiff_quality_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(rave_zdiff_quality_plugin, self).__init__()
  
  ##
  # @return a list containing the string eu.opera.odc.zdiff
  def getQualityFields(self):
    return ["eu.opera.odc.zdiff"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @param reprocess_quality_flag: If quality flag should be reprocessed or not
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, arguments=None):
    try:
      if _polarscan.isPolarScan(obj) or _polarvolume.isPolarVolume(obj):
        odc_hac.zdiff(obj)
    except:
      logger.exception("Failure during zdiff processing")
      
    return obj