'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

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
# A composite quality plugin for generating the qi total field
#
# This is no interactive registry, instead you will have to modify the xml file
# in COMPOSITE_QUALITY_REGISTRY manually.
#

##
# @file
# @author Anders Henja, SMHI
# @date 2014-03-12

# Module/Project:
import odim_source
import qitotal_options
import _polarscan, _polarvolume, _qitotal, _rave

from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

from rave_defines import QITOTAL_METHOD
## Contains site-specific argument settings

QITOTAL_DTYPE = _rave.RaveDataType_UCHAR
QITOTAL_GAIN = 1.0 / 255.0
QITOTAL_OFFSET = 0.0
# QITOTAL_METHOD = "minimum"


class rave_qitotal_quality_plugin(rave_quality_plugin):
    ##
    # Default constructor
    def __init__(self):
        super(rave_qitotal_quality_plugin, self).__init__()
        self._qitotal_option_file = None  # Mostly for test purpose

    ##
    # @return a list containing the string pl.imgw.quality.qi_total
    def getQualityFields(self):
        return ["pl.imgw.quality.qi_total"]

    ##
    # @return the qitotal site information for in object
    def get_object_information(self, inobj):
        odim_source.CheckSource(inobj)
        S = odim_source.ODIM_Source(inobj.source)
        try:
            return qitotal_options.get_qitotal_site_information(self._qitotal_option_file)[S.nod]
        except KeyError:
            return qitotal_options.get_qitotal_site_information(self._qitotal_option_file)["default"]

    ##
    # Performs the qi-total processing for the provided scan. The provided scan will get the qitotal field added to itself.
    # @param self: self
    # @param objinfo: the qitotal options for the provided scan
    # @param scan: the actual scan to process
    def processScan(self, objinfo, scan):
        qitotal = _qitotal.new()
        qitotal.datatype = QITOTAL_DTYPE
        qitotal.gain = QITOTAL_GAIN
        qitotal.offset = QITOTAL_OFFSET
        qitotalfields = []
        for f in objinfo.qifields():
            qf = scan.findQualityFieldByHowTask(f.name())
            if qf != None:
                qitotal.setWeight(f.name(), f.weight())
                qitotalfields.append(qf)
        if len(qitotalfields) > 0:
            if hasattr(qitotal, QITOTAL_METHOD):
                method = getattr(qitotal, QITOTAL_METHOD)
                result = method(qitotalfields)
                scan.addOrReplaceQualityField(result)

    ##
    # @param obj: A rave object that should be processed, bogus in this case.
    # @param reprocess_quality_flag: Not used, we always want to reprocess qi-total
    # @param arguments: Not used
    # @return: obj - without doing anything to it
    def process(
        self,
        obj,
        reprocess_quality_flag=True,
        quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY,
        arguments=None,
    ):
        objinfo = self.get_object_information(obj)

        if _polarscan.isPolarScan(obj):
            self.processScan(objinfo, obj)
        elif _polarvolume.isPolarVolume(obj):
            nscans = obj.getNumberOfScans(obj)
            for i in range(nscans):
                self.processScan(objinfo, obj.getScan(i))

        return obj, self.getQualityFields()

    ##
    # @return: The distance information - dummy
    #
    def algorithm(self):
        return None


if __name__ == "__main__":
    a = rave_qitotal_quality_plugin()
