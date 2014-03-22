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
# A quality plugin for enabling RADVOL-QC support 

## 
# @file
# @author Daniel Michelson, SMHI
# @date 2012-11-26

from rave_quality_plugin import rave_quality_plugin

class radvol_att_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(radvol_att_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.poo
  def getQualityFields(self):
    return ["pl.imgw.radvolqc.att"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj):
    try:
      import _radvol
      _radvol.attCorrection(obj)
    except:
      pass
    return obj


class radvol_broad_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(radvol_broad_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.poo
  def getQualityFields(self):
    return ["pl.imgw.radvolqc.broad"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj):
    try:
      import _radvol
      _radvol.broadAssessment(obj)
    except:
      pass
    return obj

    
class radvol_nmet_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(radvol_nmet_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.poo
  def getQualityFields(self):
    return ["pl.imgw.radvolqc.nmet"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj):
    try:
      import _radvol
      _radvol.nmetRemoval(obj)
    except:
      pass
    return obj
    

class radvol_speck_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(radvol_speck_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.poo
  def getQualityFields(self):
    return ["pl.imgw.radvolqc.speck"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj):
    try:
      import _radvol
      _radvol.speckRemoval(obj)
    except:
      pass
    return obj


class radvol_spike_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(radvol_spike_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.poo
  def getQualityFields(self):
    return ["pl.imgw.radvolqc.spike"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj):
    try:
      import _radvol
      _radvol.spikeRemoval(obj)
    except:
      pass
    return obj
