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
import string, re, datetime
from rave_dom import wmo_station

##
# Used to parse the WMO provided flatfile containing all synop stations
class rave_wmo_flatfile(object):
  def __init__(self):
    pass
    
  ##
  # Parses the file.
  # @param filename - the name of the file to parse
  # @return a list of wmo_stations
  def parse(self, filename):
    result = []
    with open(filename) as fp:
      a = fp.readlines()
      #First line contains all identifiers (tab-separated)
      IDS=a[0].lstrip().rstrip().split("\t")
      for i in range(1,len(a)):
        s = self._parse_row(IDS, a[i].lstrip().rstrip())
        if s != None:
          result.append(s)
        
    return result
  
  ##
  # Converts the gps coordinate (degree, minute, seconds) to a decimal coordinate
  # @param dd - the degrees
  # @param mm - the minutes
  # @param ss - the seconds (and an eventual identifier saying if we are on N(orth), S(outh), W(est), E(ast)
  # @return the decimal coordinate
  def _gps_to_decimal(self, dd, mm, ss):
    return (dd + mm/60.0 + ss/3600.0)
  
  ##
  # The internal parsing function
  # @param ids - a list with all the column names in same order as data
  # @row the row we are going to tokenize
  # @return a wmo_station instance
  def _parse_row(self, ids, row):
    toks = row.split("\t")
    countryidx = ids.index("CountryArea")
    countrycodeidx = ids.index("CountryCode")
    indexidx = ids.index("IndexNbr")
    subindexidx = ids.index("IndexSubNbr")
    nameidx = ids.index("StationName")
    latidx = ids.index("Latitude")
    longidx = ids.index("Longitude")
    
    if len(toks) < 7:
      return None
    
    country=None
    countrycode=None
    stationnbr=None
    stationsubnbr=None
    stationname=None
    latitude=None
    longitude=None
    if countryidx >= 0:
      country = toks[countryidx]
    if countrycodeidx > 0:
      countrycode = toks[countrycodeidx]
    if indexidx > 0:
      stationnbr = toks[indexidx]
    if subindexidx > 0:
      stationsubnbr = toks[subindexidx]
    if nameidx > 0:
      stationname = toks[nameidx]
    
    if latidx > 0:
      lattok = toks[latidx].lstrip().rstrip().split()
      sectok = lattok[2]
      seconds = None
      if sectok[-1] in ["N","S"]:
        seconds = int(sectok[:-1])
      else:
        seconds = int(sectok)
      latitude = self._gps_to_decimal(int(lattok[0]), int(lattok[1]), seconds)
      if sectok[-1] == "S":
        latitude = -latitude
    if longidx > 0:
      lontok = toks[longidx].lstrip().rstrip().split()
      sectok = lontok[2]
      seconds = None
      if sectok[-1] in ["E","W"]:
        seconds = int(sectok[:-1])
      else:
        seconds = int(sectok)
      longitude = self._gps_to_decimal(int(lontok[0]), int(lontok[1]), seconds)
      if sectok[-1] == "W":
        longitude = -longitude
    
    if stationnbr == None or latitude == None or longitude == None:
      return None
    
    return wmo_station(country, countrycode, stationnbr, stationsubnbr, stationname, longitude, latitude)
    
if __name__=="__main__":
  a=rave_wmo_flatfile()
  b=a.parse("wmo_flatfile.txt")
  for bi in b:
    print(bi)
    #if bi.country.lower().find("SWEDEN") >= 0:
    #  print bi
  

  