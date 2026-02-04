'''
Copyright (C) 2015 The Crown (i.e. Her Majesty the Queen in Right of Canada)

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
# A quality plugin with scansun support

##
# @file
# @author Daniel Michelson, Environment and Climate Change Canada
# @date 2015-12-16


# Module/Project:
from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

import _polarscan, _polarvolume
import rave_pgf_logger
from rave_pgf_scansun_plugin import writeHits

logger = rave_pgf_logger.create_logger()


class scansun_quality_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(scansun_quality_plugin, self).__init__()
    
    ##
    # @return a list containing the string nl.knmi.scansun
    # This is just a placeholder. String won't be used in this case.
    def getQualityFields(self):
        return ["nl.knmi.scansun"]
    
    ##
    # @param obj: A RAVE object that should be processed.
    # @param reprocess_quality_flag: If quality flag should be reprocessed or not
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
        try:
            import _scansun
            
            # scansun doesn't modify the payload, so there's no indicator that scansun has been run before.
            source, hits = _scansun.scansunFromObject(obj)
            scansunfolder = None
            if arguments is not None and "scansunfolder" in arguments:
                scansunfolder = arguments["scansunfolder"]
            if len(hits) > 0:
                writeHits(source, hits, scansunfolder)
        except:
            logger.exception("Failure during scansun processing")
        return obj, self.getQualityFields()
