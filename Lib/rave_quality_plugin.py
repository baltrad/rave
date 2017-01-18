'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

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
# Base class for all quality plugins

##
# Some quality plugins might be able to handle a directive on what to do with the
# quality control. In some cases, only an analyze should be performed. In other cases
# the actual analyzed field should be modified with the analyzed field
#
QUALITY_CONTROL_MODE_ANALYZE = "analyze"
QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY = "analyze_and_apply"
  

## 
# @file
# @author Anders Henja, SMHI
# @date 2011-11-04

class rave_quality_plugin(object):
  ##
  # Default sonstructor
  def __init__(self):
    pass
  
  ##
  # @return: The quality fields that should be added to the composite. Should be a list
  #
  def getQualityFields(self):
    return []
  
  ##
  # @param obj: A rave object that should be processed.
  # @param reprocess_quality_flag: Specifies if the quality flag should be reprocessed or not. If False, then if possible the plugin should avoid generating the quality field again.
  # @param arguments: If there are any arguments that should be passed to the processor  
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY, arguments=None):
    return obj, self.getQualityFields()
  
  ##
  # @return: The algorithm to use or None if no specifiec algorithm wanted
  #
  def algorithm(self):
    return None
