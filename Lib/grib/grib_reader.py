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
# abstract grib reader.
## @file
## @author Anders Henja, SMHI
## @date 2015-03-03

import _ravefield
try:
  import pygrib
except:
  pass 

class grib_reader(object):
  CONVECTIVE_AVAILABLE_POTENTIAL_ENERGY = "Convective available potential energy"
  CONVECTIVE_PRECIPITATION = "Convective precipitation"

  LOW_CLOUD_COVER = "Low cloud cover"
  TOTAL_CLOUD_COVER = "Total cloud cover"
  MEDIUM_CLOUD_COVER = "Medium cloud cover"
  HIGH_CLOUD_COVER = "High cloud cover"

  DIVERGENCE = "Divergence"
  GEOPOTENTIAL_HEIGHT = "Geopotential Height"
  LARGE_SCALE_PRECIPITATION = "Large-scale precipitation"
  LOGARITHM_OF_SURFACE_PRESSURE = "Logarithm of surface pressure"
  MEAN_SEA_LEVEL_PRESSURE = "Mean sea level pressure"
  RELATIVE_HUMIDITY = "Relative humidity"
  SKIN_TEMPERATURE = "Skin temperature"
  SNOWFALL = "Snowfall"
  SPECIFIC_CLOUD_ICE_WATER_CONTENT = "Specific cloud ice water content"
  SPECIFIC_CLOUD_LIQUID_WATER_CONTENT = "Specific cloud liquid water content"
  SPECIFIC_HUMIDITY = "Specific humidity"
  SPECIFIC_RAIN_WATER_CONTENT = "Specific rain water content"
  SPECIFIC_SNOW_WATER_CONTENT = "Specific snow water content"
  SURFACE_PRESSURE = "Surface pressure"
  TEMPERATURE = "Temperature"
  TOTAL_COLUMN_OZONE = "Total column ozone"
  TOTAL_COLUMN_WATER_VAPOUR = "Total column water vapour"
  TOTAL_PRECIPITATION = "Total precipitation"
  TWO_METRE_DEWPOINT_TEMPERATURE = "2 metre dewpoint temperature"
  TWO_METRE_TEMPERATURE = "2 metre temperature"
  TEN_METRE_U_WIND_COMPONENT = "10 metre U wind component"
  TEN_METRE_V_WIND_COMPONENT = "10 metre V wind component"
  TEN_METRE_WIND_GUST_SINCE_PREVIOUS_POST_PROCESSING= "10 metre wind gust since previous post-processing"
  U_COMPONENT_OF_WIND = "U component of wind"
  V_COMPONENT_OF_WIND = "V component of wind"
  VERTICAL_VELOCITY = "Vertical velocity"
  
  def __init__(self):
    pass
  
  ##
  # Reads the lon/lat fields as defined by the name and level
  # @param name - the name of the field (see constants) (or ignore and first field will be used for lon/lat)
  # @param level - the level or leave out
  # @return a tuple of two rave fields (lons, lats)
  def lonlats(self, name=None, level=None):
    raise Exception, "Not implemented"
  
  ##
  # Reads one field as defined by the name and level.
  # @param name - the name of the field (see constants)
  # @param level - the level or leave out and in that case, first found entry will be returned from the grib file
  # @return a rave field
  def get_field(self, name, level=None):
    raise Exception, "Not implemented"

  ##
  # Reads zero or many field as defined by the name and level.
  # @param name - the name of the field (see constants)
  # @param level - the level or leave out
  # @return a list of rave fields
  def get_fields(self, name, level=None):
    raise Exception, "Not implemented"

##
# Iterator for reading a full grib file and return the result as rave fields
#
class pygrib_grib_reader_iter(object):
  ##
  # Constructor
  # @param reader the reader
  def __init__(self, reader):
    self.reader = reader
    self.reader.grbs.seek(0)
  
  ##
  # Supports iteratable
  #
  def __iter__(self):
    return self

  ##
  # Next in line
  # @return a rave field
  def next(self):
    line = self.reader.grbs.readline()
    if line is None:
      raise StopIteration, "No more lines found"
    return self.reader.create_rave_field(line)
    
  ##
  # @see #next
  def __next__(self):
    return self.next()

##
# Grib reader that uses pygrib
#
class pygrib_grib_reader(grib_reader):
  ##
  # Constructor
  # @param filename the file that should be read
  def __init__(self, filename):
    self.filename = filename
    self.grbs = pygrib.open(filename)
  
  ##
  # Returns a tuple with lon, lat as rave fields. Without attributes. If both name and level is left out, first
  # available grib field is used for querying for lon / lats.
  # @param name the parameter name
  # @param level the level the parameter should be taken from
  #
  def lonlats(self, name=None, level=None):
    self.grbs.seek(0)
    try:
      if name != None and level != None:
        latlons = self.grbs.select(name=name, level=level)[0].latlons()
      elif name != None and level == None:
        latlons = self.grbs.select(name=name)[0].latlons()
      else:
        latlons = self.grbs.readline().latlons()
      lats = _ravefield.new()
      lats.setData(latlons[0])
      lons = _ravefield.new()
      lons.setData(latlons[1])
      return (lons, lats)
    except ValueError, e:
      raise IOError(e)
  
  ##
  # Returns specified field
  # @param name the parameter name
  # @param level the level the parameter should exist at
  # @return a rave field
  def get_field(self, name, level=None):
    try:
      if level is None:
        grb = self.grbs.select(name=name)[0]
      else:
        grb = self.grbs.select(name=name, level=level)[0]
      return self.create_rave_field(grb)
    except ValueError, e:
      raise IOError(e)
  
  ##
  # Returns matching fields
  # @param name the id
  # @param level the level the parameter should be fetched from
  # @return a list of rave fields
  def get_fields(self, name, level=None):
    result = []
    try:
      if level is None:
        grblist = self.grbs.select(name=name)
      else:
        grblist = self.grbs.select(name=name, level=level)
      for g in grblist:
        result.append(self.create_rave_field(g))
    except ValueError, e:
      pass
    return result
  
  ##
  # Creates a rave field from a grib field
  # @param grb grib field
  # @return a rave field
  def create_rave_field(self, grb):
    result = _ravefield.new()
    result.setData(grb.values)
    result.addAttribute("what/date", "%d"%grb.dataDate)
    result.addAttribute("what/time", "%02d%02d00"%(grb.hour, grb.minute))
    result.addAttribute("what/level", grb.level)
    result.addAttribute("what/name", grb.name.encode('ascii','replace'))
    result.addAttribute("what/units", grb.units.encode('ascii','replace'))
    result.addAttribute("what/gridType", grb.gridType.encode('ascii','replace'))
    result.addAttribute("what/nodata", grb.missingValue)
    return result    
  
  ##
  # Specific variant that is used to index a specific row
  # @param id - the id number (1..)
  # @return a rave field
  def get_field_by_rowid(self, id):
    self.grbs.seek(0)
    return self.create_rave_field(self.grbs.message(id))  
  
  ##
  # Creates an iterator on self
  # @return a pygrib_grib_reader_iter (ator)
  def iterator(self):
    return pygrib_grib_reader_iter(self)
  
  ##
  # So that we can use with xxx statements
  #
  def __enter__(self):
    return self

  ##
  # And close with block
  #
  def __exit__(self, type, value, traceback):
    if self.grbs != None:
      self.grbs.close()
      self.grbs = None
  
  @staticmethod
  def openfile(filename):
    return pygrib_grib_reader(filename)
  