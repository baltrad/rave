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
## Plugin for performing qc-chains on various sources.
## 
## @file
## @author Anders Henja, SMHI
## @date 2014-12-06
import rave_quality_chain_registry
from rave_quality_plugin import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY

import odim_source
import rave_pgf_quality_registry
import rave_pgf_logger

logger = rave_pgf_logger.create_logger()

class rave_quality_chain_plugin(rave_quality_plugin):
  ##
  # Default constructor
  def __init__(self, chain_registry = None):
    super(rave_quality_chain_plugin, self).__init__()
    self.chain_registry = chain_registry
    if self.chain_registry is None:
      self.chain_registry = rave_quality_chain_registry.get_global_registry() 
  
  ##
  # @return a list containing the string se.smhi.quality.chain.qc
  def getQualityFields(self):
    return ["se.smhi.quality.chain.qc"]
  
  ##
  # @param obj: A RAVE object that should be processed.
  # @return: The modified object if this quality plugin has performed changes 
  # to the object.
  def process(self, obj, reprocess_quality_flag=True, quality_control_mode=QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY, arguments=None):
    src = odim_source.NODfromSource(obj)
    try:
      chain = self.chain_registry.get_chain(src)
    except LookupError:
      return obj, []
    
    algorithm = None
    qfields = []
    for link in chain.links():
      p = rave_pgf_quality_registry.get_plugin(link.refname())
      if p != None:
        try:
          if link.arguments() is not None:
            newargs = {}
            if arguments != None:
              newargs.update(arguments)
            newargs.update(link.arguments())
            obj, plugin_qfield = p.process(obj, reprocess_quality_flag, quality_control_mode, newargs)
          else:
            obj, plugin_qfield = p.process(obj, reprocess_quality_flag, quality_control_mode, arguments)
          na = p.algorithm()
          qfields += plugin_qfield
          if algorithm == None and na != None: # Try to get the generator algorithm != None 
            algorithm = na
        except Exception:
          logger.exception("Caught exception when processing object")
    
    return obj, qfields
