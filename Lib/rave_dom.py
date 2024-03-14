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
##
# The domain object model for various objects used in rave.
#

##
# @file
# @author Anders Henja, SMHI
# @date 2013-11-06


##
# Defining a wmo station.
#
class wmo_station(object):
    ##
    # Constructor
    # @param country - the country name (like SWEDEN)
    # @param countrycode - a number in as string
    # @param stationnbr - the station number
    # @param subnbr - the station sub number
    # @param stationname - the name of the station
    # @param longitude - the longitude in degrees (decimal format)
    # @param latitude - the latitude in degrees (decimal format)
    def __init__(self, country, countrycode, stationnbr, subnbr, stationname, longitude, latitude):
        self.country = country
        self.countrycode = countrycode
        self.stationnumber = stationnbr
        self.stationsubnumber = subnbr
        self.stationname = stationname
        self.longitude = longitude
        self.latitude = latitude

    ##
    # String representation
    def __str__(self):
        return "%s (%s): %s (%s)  =>  %f, %f" % (
            self.country,
            self.countrycode,
            self.stationnumber,
            self.stationname,
            self.longitude,
            self.latitude,
        )

    ##
    # Representation of self
    def __repr__(self):
        return self.__str__()


class observation(object):
    SYNOP = 1
    SHIP = 2
    MOBILE_SYNOP = 3

    WIND_TYPE_ESTIMATED = 0  # m/s
    WIND_TYPE_ANEMOMETER = 1  # m/s

    ORIGINAL = 0
    DELAYED = 1
    CORRECTED = 2

    UPDATED_TYPES = {DELAYED: "Delayed", CORRECTED: "Corrected"}

    PARAM_VISIBILITY = 0
    PARAM_WINDTYPE = 1
    PARAM_CLOUDCOVER = 2
    PARAM_WINDDIRECTION = 3
    PARAM_WINDSPEED = 4
    PARAM_TEMPERATURE = 5
    PARAM_DEWPOINT = 6
    PARAM_RELATIVEHUMIDITY = 7
    PARAM_PRESSURE = 8
    PARAM_SEA_LVL_PRESSURE = 9
    PARAM_PRESSURE_CHANGE = 10
    PARAM_LIQUID_PRECIPTATION = 11

    def __init__(
        self, station, country, type, date, time, longitude, latitude, liquid_precipitation=0.0, accumulation_period=0
    ):
        self.station = station
        self.country = country
        self.type = type
        self.date = date
        self.time = time
        self.longitude = longitude
        self.latitude = latitude

        self.visibility = 0.0
        self.windtype = observation.WIND_TYPE_ANEMOMETER
        self.cloudcover = -1
        self.winddirection = 0
        self.windspeed = 0.0
        self.temperature = 0.0
        self.dewpoint = 0.0
        self.relativehumidity = 0.0
        self.pressure = 0.0
        self.sea_lvl_pressure = 0.0
        self.pressure_change = 0.0
        self.liquid_precipitation = liquid_precipitation
        self.accumulation_period = accumulation_period
        self.valid_fields_bitmask = 0x0000
        super(observation, self).__init__()

    def set_valid_fieldsbitmask(
        self,
        visibility=False,
        windtype=False,
        cloudcover=False,
        winddirection=False,
        windspeed=False,
        temperature=False,
        dewpoint=False,
        relativehumidity=False,
        pressure=False,
        sea_lvl_pressure=False,
        pressure_change=False,
        liquid_precipitation=False,
    ):
        if visibility:
            self.set_valid_field(self.PARAM_VISIBILITY)
        if windtype:
            self.set_valid_field(self.PARAM_WINDTYPE)
        if cloudcover:
            self.set_valid_field(self.PARAM_CLOUDCOVER)
        if winddirection:
            self.set_valid_field(self.PARAM_WINDDIRECTION)
        if windspeed:
            self.set_valid_field(self.PARAM_WINDSPEED)
        if temperature:
            self.set_valid_field(self.PARAM_TEMPERATURE)
        if dewpoint:
            self.set_valid_field(self.PARAM_DEWPOINT)
        if relativehumidity:
            self.set_valid_field(self.PARAM_RELATIVEHUMIDITY)
        if pressure:
            self.set_valid_field(self.PARAM_PRESSURE)
        if sea_lvl_pressure:
            self.set_valid_field(self.PARAM_SEA_LVL_PRESSURE)
        if pressure_change:
            self.set_valid_field(self.PARAM_PRESSURE_CHANGE)
        if liquid_precipitation:
            self.set_valid_field(self.PARAM_LIQUID_PRECIPTATION)

    def set_valid_field(self, parameter_field, is_valid=True):
        bit_value = 1 if is_valid else 0
        self.valid_fields_bitmask = self.valid_fields_bitmask | (bit_value << parameter_field)


class melting_layer(object):
    ## The constructor
    # @param nod the source nodename, e.g. sekkr
    # @param datetime the datetime of the observation
    # @param bottom bottom melting layer, in km, can be None
    # @param top top melting layer, in km, can be None
    def __init__(self, nod, datetime, bottom=None, top=None):
        self.nod = nod
        self.datetime = datetime
        self.top = top
        self.bottom = bottom
