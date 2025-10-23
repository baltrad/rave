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
# @author Yngve Einarsson, SMHI
# @date 2025-10-13

# Module/Project:
from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
from compute_pia import ComputePIA

import _polarscan, _polarscanparam, _polarvolume


class rave_pia_quality_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(rave_pia_quality_plugin, self).__init__()

    ##
    # @return a list containing the string remco.van.de.beek.qc.compute_pia
    def getQualityFields(self):
        return ["remco.van.de.beek.qc.compute_pia"]

    ##
    # @param obj: A rave object that should be processed.
    # @param reprocess_quality_flag: Specifies if the quality flag should be reprocessed or not.
    #                                If False, then if possible the plugin should avoid generating the quality field again.
    # @param arguments: Not used
    # @return: The modified object if this quality plugin has performed changes
    # to the object.
    def process(
        self,
        obj,
        reprocess_quality_flag=True,
        quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY,
        arguments=None,
    ):
        # Sanity check
        if obj is not None and _polarvolume.isPolarVolume(obj) or _polarscan.isPolarScan(obj):
            ComputePIA(obj, "DBZH", True, reprocess_quality_flag=True, quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY)
        return obj, self.getQualityFields()

    ##
    # @return: Nothing for now
    #
    def algorithm(self):

        return None
