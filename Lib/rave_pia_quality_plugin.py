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

'''


##
# A composite quality plugin for generating the PIA quality field.
#
# This is no interactive registry, instead you will have to modify the xml file
# in COMPOSITE_QUALITY_REGISTRY manually.
#
# <?xml version='1.0' encoding='UTF-8'?>
# <rave-pgf-composite-quality-registry>
#   <quality-plugin name="ropo" class="ropo_pgf_composite_quality_plugin" />
#   <quality-plugin name="pia" class="rave_pia_quality_plugin" />
# </rave-pgf-composite-quality-registry>

##
# 
# @file
# @author Yngve Einarsson, SMHI
# @date 2025-10-13

# Module/Project:
from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
from compute_pia import ComputePIA,TASK
import _polarscan, _polarscanparam, _polarvolume, _pia

# It is possible to use either native or python implenmentation of the PIA adjustment.
USE_NATIVE_IMPLEMENTATION=True

class rave_pia_quality_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(rave_pia_quality_plugin, self).__init__()
        self._HOWTASK=_pia.getHowTaskName()
        self._process = self._process_native

        if not USE_NATIVE_IMPLEMENTATION:
            self._HOWTASK = TASK
            self._process = self._process_python

    ##
    # @return a list containing the string TASK
    def getQualityFields(self):
        return [self._HOWTASK]

    def _process_native(self, obj, reprocess_quality_flag, quality_control_mode, arguments):
        """ Process PIA using the Native C implementation
        """
        applyPIA = (quality_control_mode == QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY)
        addPIAParameter=True
        pia = _pia.new()
        if obj is not None and _polarvolume.isPolarVolume(obj) or _polarscan.isPolarScan(obj):
            if _polarvolume.isPolarVolume(obj):
                for i in range(obj.getNumberOfScans()):
                    scan = obj.getScan(i)
                    pia.process(scan, "DBZH", addPIAParameter, reprocess_quality_flag, applyPIA)
            else:
                pia.process(scan, "DBZH", addPIAParameter, reprocess_quality_flag, applyPIA)
        return obj, self.getQualityFields()

    def _process_python(self, obj, reprocess_quality_flag, quality_control_mode, arguments):
        """ Process PIA using the Python implementation
        """
        if obj is not None and _polarvolume.isPolarVolume(obj) or _polarscan.isPolarScan(obj):
            ComputePIA(obj, "DBZH", True, reprocess_quality_flag, quality_control_mode)
        return obj, self.getQualityFields()

    ##
    # @param obj: A rave object that should be processed.
    # @param reprocess_quality_flag: Specifies if the quality flag should be reprocessed or not.
    # If False, then if possible the plugin should avoid generating the quality field again.
    # @param arguments: Not used
    # @return: The modified object if this quality plugin has performed changes
    # to the object.
    def process(
        self,
        obj,
        reprocess_quality_flag=True,
        quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY,
        arguments=None):

        return self._process(obj, reprocess_quality_flag, quality_control_mode, arguments)

    ##
    # @return: Nothing for now
    #
    def algorithm(self):

        return None
