'''
Copyright (C) 2026 Swedish Meteorological and Hydrological Institute, SMHI,

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
# A quality plugin for enabling support for HOOF dealiasing.
# Like the dealias plugin, it leaves the original field unaffected and adds a new
# field where the dealiasing has been applied. This is done for all quality control
# modes, i.e., both for "analyze" and "analyze & apply".
# HOOF dealiasing fits the wind model using all scans of a volume, so only polar
# volumes are processed.

##
# @file
# @author Peter Smerkol, Vito Švagelj, ARSO
# @date 2026-11-13

# Module/Project:
from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
import rave_pgf_logger

logger = rave_pgf_logger.create_logger()

# same naming of the dealiased parameters as in the dealias plugin
QUANTITY_CONVERSION_MAP = {"VRAD": "VRADDH", "VRADH": "VRADDH", "VRADV": "VRADDV"}


class hoofdealias_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(hoofdealias_plugin, self).__init__()

    ##
    # @return a list containing the string si.arso.detector.hoofdealias
    def getQualityFields(self):
        return ["si.arso.detector.hoofdealias"]

    ##
    # @param obj: A RAVE object that should be processed.
    # @param reprocess_quality_flag: Not used
    # @param quality_control_mode: Not used
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
            add_hoofdealiased_param(obj)
        except:
            logger.exception("Failure during HOOF dealias processing")

        return obj, self.getQualityFields()


##
# Adds the dealiased parameters to all scans of a polar volume and dealiases them.
# @param obj: the polar volume
def add_hoofdealiased_param(obj):
    import _polarvolume
    import _hoofdealias

    if obj is None or not _polarvolume.isPolarVolume(obj):
        logger.warning("HOOF dealiasing requires a polar volume, object not processed")
        return

    # copy the original quantities to the new quantities in all scans, then dealias the new
    # quantities in the whole volume, since the wind model is fitted using all scans
    newQuantities = []
    for i in range(obj.getNumberOfScans()):
        scan = obj.getScan(i)
        for originalQuantity, newQuantity in QUANTITY_CONVERSION_MAP.items():
            if scan.hasParameter(originalQuantity) and not scan.hasParameter(newQuantity):
                param = scan.getParameter(originalQuantity).clone()
                param.quantity = newQuantity
                scan.addParameter(param)
                if newQuantity not in newQuantities:
                    newQuantities.append(newQuantity)

    for newQuantity in newQuantities:
        logger.debug("HOOF dealiasing of %s" % newQuantity)
        _hoofdealias.dealias(obj, newQuantity)
