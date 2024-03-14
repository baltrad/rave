'''
Copyright (C) 2018- Environment and Climate Change Canada

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
# A composite quality plugin for generating the surface distance field.
#
# This is no interactive registry, instead you will have to modify the xml file
# in COMPOSITE_QUALITY_REGISTRY manually.
#

##
# @file
# @author Daniel Michelson, ECCC
# @date 2018-02-09

# Module/Project:
from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY


class rave_height_quality_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(rave_height_quality_plugin, self).__init__()

    ##
    # @return a list containing the string se.smhi.composite.height.radar
    def getQualityFields(self):
        return ["se.smhi.composite.height.radar"]

    ##
    # @param obj: A rave object that should be processed, bogus in this case.
    # @param reprocess_quality_flag: Not used
    # @param arguments: Not used
    # @return: obj - without doing anything to it
    def process(
        self,
        obj,
        reprocess_quality_flag=True,
        quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY,
        arguments=None,
    ):
        return obj, self.getQualityFields()

    ##
    # @return: The height information - dummy
    #
    def algorithm(self):
        return None
