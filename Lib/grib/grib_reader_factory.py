'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## 
# grib reader factory that provides the user with a reader.

## @file
## @author Anders Henja, SMHI
## @date 2015-03-03
class grib_reader_factory(object):
  def __init__(self, gribreaderfactory=None):
    self.gribreaderfactory = gribreaderfactory
    #if self.gribreader is not None:
    #  self.gribreader = grib_reader.pygrib_grib_reader()

  def open(self, filename):
    if self.gribreaderfactory is None:
      from grib import grib_reader
      return grib_reader.pygrib_grib_reader.openfile(filename)
    return self.gribreaderfactory(filename)

def get_factory(configfile=None):
  if configfile is None:
    return grib_reader_factory()
  else:
    raise Exception, "Dynamic configuration of grib reader not found"