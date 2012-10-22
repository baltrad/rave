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
# A composite quality plugin for generating the probability of overshooting quality
# field. 
#
# This is no interactive registry, instead you will have to modify the xml file
# in COMPOSITE_QUALITY_REGISTRY manually.
#
# <?xml version='1.0' encoding='UTF-8'?>
# <rave-pgf-composite-quality-registry>
#   <quality-plugin name="ropo" class="ropo_pgf_composite_quality_plugin" />
#   <quality-plugin name="rave-overshooting" class="rave_overshooting_quality_plugin" />
# </rave-pgf-composite-quality-registry>

## 
# @file
# @author Anders Henja, SMHI
# @date 2011-11-04

from rave_quality_plugin import rave_quality_plugin
import _polarvolume

class rave_overshooting_quality_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self):
    super(rave_overshooting_quality_plugin, self).__init__()
  
  ##
  # @return a list containing the string se.smhi.detector.poo
  def getQualityFields(self):
    return ["se.smhi.detector.poo"]
  
  ##
  # @param obj: A rave object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj):
    if obj != None and _polarvolume.isPolarVolume(obj):
      import _detectionrange
      ascending = obj.isAscendingScans()
      drgenerator = _detectionrange.new()
      maxscan = obj.getScanWithMaxDistance()
      # We want to have same resolution as maxdistance scan since we are going to add the poo-field to it
      # The second argument is dbz threshold, modify it accordingly
      topfield = drgenerator.top(obj, maxscan.rscale, -40.0)       # Topfield is a scan
      filterfield = drgenerator.filter(topfield)                   # filterfield is a scan
      poofield = drgenerator.analyze(filterfield, 60, 0.1, 0.35)   # poofield is a quality field, add it to maxscan
      maxscan.addQualityField(poofield)
      if ascending:
        obj.sortByElevations(1)
    return obj

  ##
  # @return: The poo composite algorithm
  #
  def algorithm(self):
    import _poocompositealgorithm
    return _poocompositealgorithm.new()
  