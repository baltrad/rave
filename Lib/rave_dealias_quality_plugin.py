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
# A quality plugin for enabling support for dealiasing

## 
# @file
# @author Daniel Michelson, SMHI
# @date 2013-10-07

from rave_quality_plugin import rave_quality_plugin

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
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True):
    try:
      import _dealias
      _dealias.dealias(obj)
    except:
      pass
    return obj
