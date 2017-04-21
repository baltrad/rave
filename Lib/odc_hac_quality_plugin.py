'''
Copyright (C) 2013- Swedish Meteorological and Hydrological Institute (SMHI)

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
'''
##
# A quality plugin for enabling ODC hit-accumulation clutter management

## 
# @file
# @author Daniel Michelson, SMHI
# @date 2013-01-24

from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

class odc_hac_increment_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(odc_hac_increment_plugin, self).__init__()
  
  ##
  # @return a list containing the corresponding string
  def getQualityFields(self):
    return ["eu.opera.odyssey.hac"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @param reprocess_quality_flag: Not used
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object. In this case, no changes will be made.
  def process(self, obj, reprocess_quality_flag=True, quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY, arguments=None):
    try:
      import odc_hac
      odc_hac.hacIncrement(obj)
    except:
      pass
    return obj, self.getQualityFields()


class odc_hac_filter_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(odc_hac_filter_plugin, self).__init__()
  
  ##
  # @return a list containing the corresponding string
  def getQualityFields(self):
    return ["eu.opera.odyssey.hac"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @param reprocess_quality_flag: Not used  
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY, arguments=None):
    try:
      import odc_hac
      odc_hac.hacFilter(obj)
    except:
      pass
    return obj, self.getQualityFields()
