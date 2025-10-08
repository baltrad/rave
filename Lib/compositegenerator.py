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
## Python interface to composite generator functionality.

## @file
## @author Anders Henja, SMHI
## @date 2025-02-27
import os, sys, math
import _compositefactorymanager, _compositegenerator, _compositearguments, _raveproperties, _odimsources
import _rave

from rave_defines import COMPOSITE_GENERATOR_FILTER_FILENAME, ACQVA_CLUTTERMAP_DIR, ODIM_SOURCE_FILE, COMPOSITE_GENERATOR_PROPERTY_FILE

class Generator(object):
  def __init__(self, properties_file=COMPOSITE_GENERATOR_PROPERTY_FILE, generatorfilter=COMPOSITE_GENERATOR_FILTER_FILENAME):
    super(Generator, self).__init__()
    self._manager = _compositefactorymanager.new()
    self._generator = _compositegenerator.create(self._manager, generatorfilter)
    self._generator.properties = self.load_properties(properties_file)

  def create_arguments(self):
    return _compositearguments.new()

  ##
  # Converts a string into a number, either int or float. If value already is an int or float, that value is returned.
  # @param sval the string to translate
  # @return the translated value
  # @throws ValueError if value not could be translated
  #
  def _strToNumber(self, sval):
    if isinstance(sval, float) or isinstance(sval, int):
      return sval

    try:
      return int(sval)
    except ValueError:
      return float(sval)

  def update_arguments_with_prodpar(self, arguments, prodpar):
    if prodpar is not None:
      if arguments.product in ["CAPPI", "PCAPPI"]:
        try:
          arguments.height = self._strToNumber(prodpar)
        except ValueError:
          pass
      elif arguments.product in ["PMAX"]:
        if isinstance(prodpar, str):
          pp = self.prodpar.split(",")
          if len(pp) == 2:
            try:
              arguments.height = self._strToNumber(pp[0].strip())
              arguments.range = self._strToNumber(pp[1].strip())
            except ValueError:
              pass
          elif len(pp) == 1:
            try:
              arguments.height = self._strToNumber(pp[0].strip())
            except ValueError:
              pass
      elif arguments.product in ["PPI"]:
        try:
          v = self._strToNumber(prodpar)
          arguments.elangle = v * math.pi / 180.0
        except ValueError:
          pass

  def load_properties(self, properties_file):
    try:
      properties = _raveproperties.load(properties_file)
    except:
      properties = _raveproperties.new()
      properties.set("rave.acqva.cluttermap.dir", ACQVA_CLUTTERMAP_DIR)
    
    if not properties.sources:
      properties.sources = _odimsources.load(ODIM_SOURCE_FILE)

    return properties

  def generate(self, arguments):
    return self._generator.generate(arguments)


    

    
