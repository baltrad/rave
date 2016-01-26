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
import _polarscan, _polarvolume
import rave_pgf_logger
logger = rave_pgf_logger.rave_pgf_syslog_client()

def should_perform_qc_process(reprocess, obj, how_task):
  if reprocess:
    return True
  
  if _polarscan.isPolarScan(obj) and obj.findQualityFieldByHowTask(how_task):
    return False
  
  if _polarvolume.isPolarVolume(obj):
    for i in range(obj.getNumberOfScans()):
      scan = obj.getScan(i)
      if not scan.findQualityFieldByHowTask(how_task):
        return True
    return False
  
  return True
    
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
  # @param reprocess_quality_flag: If quality flag should be reprocessed or not
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, arguments=None):
    try:
      import _radvol, rave_radvol_realtime
      rpars = rave_radvol_realtime.get_options(obj)
      if should_perform_qc_process(reprocess_quality_flag, obj, "pl.imgw.radvolqc.att"):
        _radvol.attCorrection(obj, rpars)
    except:
      logger.exception("Failure during radvol processing")
    return obj, self.getQualityFields()


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
  # @param reprocess_quality_flag: If quality flag should be reprocessed or not
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, arguments=None):
    try:
      import _radvol, rave_radvol_realtime
      rpars = rave_radvol_realtime.get_options(obj)
      if should_perform_qc_process(reprocess_quality_flag, obj, "pl.imgw.radvolqc.broad"):
        _radvol.broadAssessment(obj, rpars)
    except:
      logger.exception("Failure during radvol processing")
    return obj, self.getQualityFields()

    
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
  # @param reprocess_quality_flag: If quality flag should be reprocessed or not
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, arguments=None):
    try:
      import _radvol, rave_radvol_realtime
      rpars = rave_radvol_realtime.get_options(obj)
      if should_perform_qc_process(reprocess_quality_flag, obj, "pl.imgw.radvolqc.nmet"):
        _radvol.nmetRemoval(obj, rpars)
    except:
      logger.exception("Failure during radvol processing")
    return obj, self.getQualityFields()
    

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
  # @param reprocess_quality_flag: If quality flag should be reprocessed or not
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, arguments=None):
    try:
      import _radvol, rave_radvol_realtime
      rpars = rave_radvol_realtime.get_options(obj)
      if should_perform_qc_process(reprocess_quality_flag, obj, "pl.imgw.radvolqc.speck"):      
        _radvol.speckRemoval(obj, rpars)
    except:
      logger.exception("Failure during radvol processing")
    return obj, self.getQualityFields()


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
  # @param reprocess_quality_flag: If quality flag should be reprocessed or not
  # @param arguments: Not used
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, arguments=None):
    try:
      import _radvol, rave_radvol_realtime
      rpars = rave_radvol_realtime.get_options(obj)
      if should_perform_qc_process(reprocess_quality_flag, obj, "pl.imgw.radvolqc.spike"):      
        _radvol.spikeRemoval(obj, rpars)
    except:
      logger.exception("Failure during radvol processing")
    return obj, self.getQualityFields()
