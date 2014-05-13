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
import _rave
from numpy import log10

MIN_GMM = 0.1  # Minimum gauge precip allowed when deriving coefficients
MIN_RMM = 0.1  # Minimum radar precip allowed when deriving coefficients

##
# Represents one point to be used in the gra adjustment. Contains both information related to
# the radar and the observation
#
class grapoint(object):
  ## The constructor
  # @param rt the radar values data type
  # @param rv the radar value
  # @param rd the distance from the radar to this point
  # @param date the date of the observation
  # @param time the time of the observation
  # @param liquid_prcipitation the amount of rain
  # @param accumulation_period the time during which the precipitation has been measured
  def __init__(self, rt, rv, rd, longitude, latitude, date, time, liquid_precipitation, accumulation_period):
    self.radarvaluetype = rt
    self.radarvalue = rv
    self.radardistance = rd
    self.longitude = longitude
    self.latitude = latitude
    self.date = date
    self.time = time
    self.observation = liquid_precipitation
    self.accumulation_period = accumulation_period
    self.gr = -1
    if self.radarvaluetype == _rave.RaveValueType_DATA and self.radarvalue >= MIN_RMM:
      self.gr = float(10 * log10(self.observation / self.radarvalue))
      

  ## Creates a gra point from the radar information and a observation instance.
  # @param rt the radar values data type
  # @param rv the radar value
  # @param rd the distance from the radar to this point
  # @param obs the rave_dom observation object
  #
  @classmethod
  def from_observation(cls, rt, rv, rd, obs):
    return grapoint(rt, rv, rd, obs.longitude, obs.latitude, obs.date, obs.time, obs.liquid_precipitation, obs.accumulation_period)
  