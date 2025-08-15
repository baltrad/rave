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

'''
SMUR11 UKMS 300600 RRB
AAXX 30061
33415 11697 73401 10118 20095 39804 40194 52010 69902 70282
84330 333 20117=

SMNO01 ENMI 300600 RRC
AAXX 30061
01026 16/// /2004 10032 20013 39705 49843 52023 6//// 333 20026
      91112=

'''
# Standard python libs:
import string
import re
import datetime


class fm12_base_info(object):
    # (synop_id, headterm, country, centre, dayofmonth, hourofday, minuteofhour, updated)
    def __init__(self, synopid, headterm, country, centre, datestr, timestr, updated):
        self.synopid = synopid
        self.headterm = headterm
        self.country = country
        self.centre = centre
        self.datestr = datestr
        self.timestr = timestr
        self.updated = updated

    def __str__(self):
        s = "fm12_base_info:"
        if self.synopid == "S":
            s = s + "\n" + "SYNOP"
        else:
            s = s + "\n" + "Typeclass: %s" % self.synopid
        s = s + "\n" + "Headterm: %s" % self.headterm
        s = s + "\n" + "Country: %s" % self.country
        s = s + "\n" + "Centre: %s" % self.centre
        s = s + "\n" + "Datetime: %s %s" % (self.datestr, self.timestr)
        if self.updated == fm12_obs.CORRECTED:
            s = s + "\nCORRECTED"
        elif self.updated == fm12_obs.DELAYED:
            s = s + "\nDELAYED"
        else:
            s = s + "\nORIGINAL"
        return s

    def __repr__(self):
        return self.__str__()


class fm12_obs(object):
    SYNOP = 1
    SHIP = 2
    MOBILE_SYNOP = 3

    WIND_TYPE_ESTIMATED_METER_PER_SECOND = 0
    WIND_TYPE_ANEMOMETER_METER_PER_SECOND = 1
    WIND_TYPE_ESTIMATED_KNOTS = 2
    WIND_TYPE_ANEMOMETER_KNOTS = 3

    ORIGINAL = 0
    DELAYED = 1
    CORRECTED = 2
    INVALID = 3  # This is when the observation is not parsable due to for example breaking spec or similar

    OBS_TYPES = {SYNOP: "SYNOP", SHIP: "SHIP", MOBILE_SYNOP: "MOBILE SYNOP"}

    WIND_TYPES = {
        WIND_TYPE_ESTIMATED_METER_PER_SECOND: "Estimated m/s",
        WIND_TYPE_ANEMOMETER_METER_PER_SECOND: "Anemometer m/s",
        WIND_TYPE_ESTIMATED_KNOTS: "Estimated knots",
        WIND_TYPE_ANEMOMETER_KNOTS: "Anemometer knots",
    }

    UPDATED_TYPES = {DELAYED: "Delayed", CORRECTED: "Corrected"}

    def __init__(self, station, type, date, time, windtype=WIND_TYPE_ESTIMATED_METER_PER_SECOND):
        self.station = station
        self.visibility = None
        self.cloudbase = None
        self.type = type
        self.date = date
        self.time = time
        self.windtype = windtype

        self.cloudcover = None
        self.winddirection = None
        self.windspeed = None

        self.temperature = 0.0
        self.dewpoint = 0.0
        self.relativehumidity = 0.0
        self.pressure = 0.0
        self.sea_lvl_pressure = 0.0
        self.pressure_change = 0.0
        self.liquid_precipitation = 0.0
        self.accumulation_period = 0
        self.updated = fm12_obs.ORIGINAL
        self.max_24hr_temperature = None
        self.min_24hr_temperature = None
        super(fm12_obs, self).__init__()


##
# The parser of the fm12 synop data
#
class fm12_parser(object):
    def __init__(self):
        super(fm12_parser, self).__init__()

    ##
    # Parses the file.
    # @param filename the file containing the fm12 formatted data
    # @returns a list of fm12_obs
    def parse(self, filename):
        fp = open(filename)
        data = fp.read()
        fp.close()
        if data[:1] == '\x01':
            data = data[1:]
        data = data.lstrip().rstrip()
        groups = re.split(r"[~\r\n]+", data)
        ngroups = []
        for g in groups:
            if g.rstrip().lstrip() != "":
                ngroups.append(g)
        groups = ngroups
        blocks = self._extract_blocks(groups)

        result = []
        for groups in blocks:
            if len(groups) < 3:
                raise ValueError("Synop must at least contain Identifier, type and observation group")

            startpos = 0
            for x in range(len(groups)):
                if groups[x].lstrip().rstrip()[0:4] in ["AAXX", "BBXX", "OOXX"]:
                    startpos = x - 1
                    break
                if startpos < 0:
                    raise ValueError("Can not handle this file. AAXX/BBXX/OOXX block starts too early")

            groups = groups[startpos:]
            result.extend(self._parse_data(groups, filename))

        # return self._parse_data(groups, filename)
        return result

    def _extract_blocks(self, groups):
        blocks = []
        grp = []
        for g in groups:
            if g[0] == '\x03':
                blocks.append(grp)
                grp = []
            else:
                grp.append(g)
        if len(grp) > 0:
            blocks.append(grp)
        return blocks

    ##
    # First step of the parsing.
    #
    def _parse_data(self, groups, filename):
        baseinfo = self._parse_base_info(groups[0], filename)
        result = []
        extracted = []
        idx = 0
        l = len(groups)

        # First we need to sort out all AAXX/BBXX/OOXX blocks before we determine the
        # observations / block
        while idx < l:
            data = groups[idx].lstrip().rstrip()
            if data[0:4] in ["AAXX", "BBXX", "OOXX"]:
                idx = idx + 1
                while idx < l:
                    ndata = groups[idx].lstrip().rstrip()
                    if ndata[0:4] in ["AAXX", "BBXX", "OOXX"]:
                        idx = idx - 1  # We want to break out and run into the AAXX stuff again
                        break
                    data = data + "\n" + ndata
                    idx = idx + 1
                extracted.append(data)
            idx = idx + 1

        # Now we can extract each observation group and add them to the result
        for block in extracted:
            result.extend(self._parse_block(baseinfo, block))

        return result

    ##
    # Extracts the base information about this file
    #
    def _parse_base_info(self, base_id, filename):
        # SMSN86 ESWI 300000
        tokens = base_id.split()
        synopchar = tokens[0][0]
        headterm = tokens[0][1]
        country = tokens[0][2:4]
        centre = tokens[1]

        # Instead of trying to get time from the section information, we try the filename instead
        datetimestr = filename[-18:-4]
        datestr = datetimestr[:8]
        timestr = datetimestr[8:]

        if not re.match("^[0-9]{8}$", datestr) or not re.match("^[0-9]{6}$", timestr):
            if re.match(".*_[0-9]{8}.txt$", filename):
                datetimestr = filename[-27:-13]
                datestr = datetimestr[:8]
                timestr = datetimestr[8:]
            elif re.match(".*_[0-9]{12}.txt$", filename):
                datetimestr = filename[-16:-4]
                datestr = datetimestr[:8]
                timestr = datetimestr[8:] + "00"

        # If this filename does not contain datetime as last part, we will have to figure out the date from the actual synop data
        # FYI: this is very unsafe way since it does not take into account year shifts and month shifts.
        # Where is year and month info in a synop?
        if not re.match("^[0-9]{8}$", datestr) or not re.match("^[0-9]{6}$", timestr):
            dayofmonth = tokens[2][0:2]
            hourofday = tokens[2][2:4]
            minuteofhour = tokens[2][4:6]
            now = datetime.datetime.now()
            datestr = "%d%02d%s" % (now.year, now.month, dayofmonth)
            timestr = "%s%s00" % (hourofday, minuteofhour)

        updated = fm12_obs.ORIGINAL
        if len(tokens) > 3:
            if tokens[3][0] == "R":
                updated = fm12_obs.DELAYED
            elif tokens[3][0] == "C":
                updated = fm12_obs.CORRECTED

        return fm12_base_info(synopchar, headterm, country, centre, datestr, timestr, updated)

    ##
    # Parses one block of data (between AAXX/BBXX/OOXX and next AAXX/BBXX/OOXX
    # @param base_id the base info
    # @param blcok the block to parse
    # @return a list of fm12_obs
    def _parse_block(self, base_id, block):
        # 'AAXX 30061\n01026 16/// /2004 10032 20013 39705 49843 52023 6//// 333 20026\n91112='
        result = []
        id = block[0 : block.find('\n')]
        rest = block[len(id) + 1 :]
        observations = re.split("=", rest)

        for o in observations:
            data = o.lstrip().rstrip()
            # We need to take care of garbage characters that might occur in the transmission
            if data[:1] in ['\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\x09']:
                data = data[1:]
            elif data[-1:] in ['\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\x09']:
                data = data[:-1]
            if data != "":
                o = self._parse_obs(base_id, id, data)
                if o != None:
                    result.append(o)

        return result

    ##
    # Parses one observation. Currently, only AAXX (SYNOP) are handled
    # @param baseinfo - the base info
    # @param id - the identifier for this obs
    # @param data - the obs data to be parsed
    # @return a fm12_obs
    def _parse_obs(self, baseinfo, id, data):
        toks = id.split()
        if toks[0] == "AAXX":
            return self._parse_aaxx_obs(baseinfo, toks[1], data)

        return None

    ##
    # Parses an aaxx obs.
    # @param baseinfo - the base info
    # @param yyggi - (yy)day of month, (gg)hour of day, (i)wind type indicator
    def _parse_aaxx_obs(self, baseinfo, yyggi, data):
        toks = data.split()
        station = toks[0]

        if toks[1] == "NIL":  # This is really not something that should exist in the obs
            return None

        # Next token should define what we can find in the rest of the observation
        iR, iX, h, VV = toks[1][0], toks[1][1], toks[1][2], toks[1][3:5]

        dyofmonth, hourofday, windtypestr = yyggi[:2], yyggi[2:4], yyggi[4]

        section1, section2, section3, section4, section5 = self._create_sections_from_obs(toks[2:])

        # It is time to populate the observation with data from the different sections
        # __init__(self, type, date, time, windtype = WIND_TYPE_ESTIMATED_METER_PER_SECOND):
        obs = fm12_obs(station, fm12_obs.SYNOP, baseinfo.datestr, baseinfo.timestr, int(windtypestr))
        obs.updated = baseinfo.updated
        if VV != "//":
            try:
                v = int(VV)
                if v > 0 and v <= 55:  # 01 -- 50 = 0.1, 0.2 .. 5.0 km
                    obs.visibility = 0.1 * v
                elif v > 56 and v <= 80:  # 56 -- 80 = 6,7,..30 km
                    obs.visibility = (v - 56) + 6.0
                elif v >= 81 and v <= 88:
                    obs.visibility = (v - 81) * 5 + 35
                elif v == 89:
                    obs.visibility = 71  # Really means > 70 km
                elif v == 90:
                    obs.visibility = 0.04  # Really means < 0.05km
                elif v == 91:
                    obs.visibility = 0.05
                elif v == 92:
                    obs.visibility = 0.2
                elif v == 93:
                    obs.visibility = 0.5
                elif v == 94:
                    obs.visibility = 1.0
                elif v == 95:
                    obs.visibility = 2.0
                elif v == 96:
                    obs.visibility = 4.0
                elif v == 97:
                    obs.visibility = 10.0
                elif v == 98:
                    obs.visibility = 20.0
                elif v == 99:
                    obs.visibility = 51  # Really means > 50km
            except:
                pass

        if h != "/":
            try:
                obs.cloudbase = int(h)
            except:
                pass

        self._fill_obs_with_section1(obs, section1)
        self._fill_obs_with_section3(obs, section3)

        return obs

    ##
    # Creates sections from the tokens in an observation. The tokens should not include the station identifier
    # and first 111 group (iihVV)
    # @param tokens - the token list
    # @returns a tuple of sections (111,222,333,444,555) where the sections always will be a list even if there is no data.
    def _create_sections_from_obs(self, tokens):
        section1 = []
        section2 = []
        section3 = []
        section4 = []
        section5 = []

        secid = 1
        firstTokenParsedHandled = False
        for t in tokens:
            if firstTokenParsedHandled and t[:3] == '222':
                # The 222 section should be 222xx but some observations only have 222. So in order to determine
                # that it is a real 222 section we must have processed the 3:rd token in the section 1 before we
                # can determine that it actually is a 222 section and not data from the Nddff token
                secid = 2
                continue
            if t == '333':
                secid = 3
                continue
            elif t == '444':
                secid = 4
                continue
            elif t == '555':
                secid = 5
                continue

            if secid == 1:
                section1.append(t)
            elif secid == 2:
                section2.append(t)
            elif secid == 3:
                section3.append(t)
            elif secid == 4:
                section4.append(t)
            else:
                section5.append(t)

            firstTokenParsedHandled = True

        return (section1, section2, section3, section4, section5)

    ##
    # Handles group 1 in section 1
    # @param obs - the obs to fill with information
    # @param token - in the format 1sTTT (Temperature)
    def _parse_section1_1(self, obs, token):
        # 1sTTT -- Temperature
        s = token[1]
        TTT = token[2:5]
        try:
            tv = int(TTT) * 0.1
            if s == '1':
                tv = -tv
            obs.temperature = tv
        except:
            pass

    ##
    # Handles group 2 in section 1
    # @param obs - the obs to fill with information
    # @param token - in the format 2sTTT (Dewpoint)
    def _parse_section1_2(self, obs, token):
        # 2sTTT -- Dewpoint
        s = token[1]
        TTT = token[2:5]
        try:
            tv = int(TTT) * 0.1
            if s == '1':
                tv = -tv
            if s == '9':
                obs.relativehumidity = tv
            else:
                obs.dewpoint = tv
        except:
            pass

    ##
    # Handles group 3 in section 1
    # @param obs - the obs to fill with information
    # @param token - in the format 3PPPP (Station pressure in 0.1 mb )
    def _parse_section1_3(self, obs, token):
        # 3PPPP -- Station pressure in 0.1 mb (thousandths digit omitted, last digit can be slash, then pressure in full mb)
        try:
            if token[4] == '/':
                obs.pressure = float(int(token[1:4]))
            else:
                obs.pressure = int(token[1:5]) * 0.1
            if obs.pressure < 90.0:
                obs.pressure = 1000.0 + obs.pressure

        except:
            pass

    ##
    # Handles group 4 in section 1
    # @param obs - the obs to fill with information
    # @param token - in the format 4PPPP (Sea level pressure in 0.1 mb )
    def _parse_section1_4(self, obs, token):
        # 4PPPP -- Sea level pressure in 0.1 mb (thousandths digit omitted, last digit can be slash, then pressure in full mb)
        try:
            if token[4] == '/':
                obs.sea_lvl_pressure = float(int(token[1:4]))
            else:
                obs.sea_lvl_pressure = int(token[1:5]) * 0.1
            if obs.sea_lvl_pressure < 90.0:
                obs.sea_lvl_pressure = 1000.0 + obs.sea_lvl_pressure
        except:
            pass

    ##
    # Handles group 5 in section 1
    # @param obs - the obs to fill with information
    # @param token - in the format 5appp (Pressure tendency over 3 hours)
    def _parse_section1_5(self, obs, token):
        # 5appp -- Pressure tendency over 3 hours
        try:
            a = int(token[1])
            pchange = int(token[2:5]) * 0.1
            if a <= 4:
                obs.pressure_change = pchange
            else:
                obs.pressure_change = -pchange
        except:
            pass

    ##
    # Handles group 6 in section 1
    # @param obs - the obs to fill with information
    # @param token - in the format 6RRRt (Liquid precipitation)
    def _parse_section1_6(self, obs, token):
        # 6RRRt -- Liquid precipitation
        # Precipitation amount in mm (001-989)=>mm, 990 trace, 991-999 = 0.1 - 0.9mm
        # t -- Duration over which precipitation amount measured
        try:
            v = int(token[1:4])
            if v < 989:
                obs.liquid_precipitation = v
            elif v > 990:
                obs.liquid_precipitation = (v - 990) * 0.1
            if token[4] == '/':
                obs.accumulation_period = 24
            else:
                ap = int(token[4])
                if ap <= 4:
                    obs.accumulation_period = 6 * ap
                elif ap > 4 and ap <= 7:
                    obs.accumulation_period = 3 - (7 - ap)
                elif ap == 8:
                    obs.accumulation_period = 9
                elif ap == 9:
                    obs.accumulation_period = 15
        except:
            pass

    ##
    # Not used
    def _parse_section1_7(self, obs, token):
        # 7wwWW -- Present and past weather
        pass

    ##
    # Not used
    def _parse_section1_8(self, obs, token):
        # 8NCCC -- Cloud type information
        pass

    ##
    # Not used
    def _parse_section1_9(self, obs, token):
        # 9GGgg -- Time of observation in hours and minutes
        pass

    SECTION1_PARTS = {
        '1': _parse_section1_1,
        '2': _parse_section1_2,
        '3': _parse_section1_3,
        '4': _parse_section1_4,
        '5': _parse_section1_5,
        '6': _parse_section1_6,
        '7': _parse_section1_7,
        '8': _parse_section1_8,
        '9': _parse_section1_9,
    }

    ##
    # Processes section 1 of the observation and sets the relevant information in the obs
    # @param obs - the observation to get data filled in
    # @param section1 - the section 1 groups
    def _fill_obs_with_section1(self, obs, section1):
        idx = 0
        if len(section1) > 0:
            # section1[0] = nddff
            N = section1[idx][0]
            dd = section1[idx][1:3]
            ff = section1[idx][3:5]
            if N != '/':
                obs.cloudcover = int(N)

            if dd != '//':
                obs.winddirection = int(dd) * 10

            if ff != '//':
                obs.windspeed = int(ff)

            idx = idx + 1

            if len(section1) > 1 and section1[idx][0:2] == '00':
                try:
                    obs.windspeed = int(section1[idx][2:5])
                except:
                    pass
                idx = idx + 1

        while idx < len(section1):
            secpart = section1[idx][0]
            if secpart in self.SECTION1_PARTS:
                self.SECTION1_PARTS[secpart](self, obs, section1[idx])
            idx = idx + 1

    ##
    # Not used
    def _parse_section3_0(self, obs, token):
        # 0.... -- Regionally developed data
        pass

    ##
    # Handles group 1 in section 3
    # @param obs - the obs to fill with information
    # @param token - in the format 1sTTT (Maximum temperature over previous 24 hours)
    def _parse_section3_1(self, obs, token):
        # 1sTTT -- Maximum temperature over previous 24 hours
        s = token[1]
        try:
            tv = int(token[2:5]) * 0.1
            if s == '1':
                tv = -tv
            obs.max_24hr_temperature = tv
        except:
            pass

    ##
    # Handles group 2 in section 3
    # @param obs - the obs to fill with information
    # @param token - in the format 2sTTT (Minimum temperature over previous 24 hours)
    def _parse_section3_2(self, obs, token):
        # 2sTTT -- Minimum temperature over previous 24 hours
        s = token[1]
        try:
            tv = int(token[2:5]) * 0.1
            if s == '1':
                tv = -tv
            obs.min_24hr_temperature = tv
        except:
            pass

    ##
    # Not used
    def _parse_section3_3(self, obs, token):
        # 3Ejjj -- Regionally developed data
        pass

    ##
    # Not used
    def _parse_section3_4(self, obs, token):
        # 4Esss -- Snow depth
        pass

    ##
    # Not used
    def _parse_section3_5(self, obs, token):
        # 5jjjj jjjjj -- Additional information (can be multiple groups)
        pass

    ##
    # Not used
    def _parse_section3_6(self, obs, token):
        # 6RRRt -- Liquid precipitation
        pass

    ##
    # Not used
    def _parse_section3_7(self, obs, token):
        # 24 hour precipitation in mm
        pass

    def _parse_section3_8(self, obs, token):
        # Cloud layer data
        pass

    ##
    # Not used
    def _parse_section3_9(self, obs, token):
        # 9SSss -- Supplementary information
        pass

    SECTION3_PARTS = {
        '0': _parse_section3_0,
        '1': _parse_section3_1,
        '2': _parse_section3_2,
        '3': _parse_section3_3,
        '4': _parse_section3_4,
        '5': _parse_section3_5,
        '6': _parse_section3_6,
        '7': _parse_section3_7,
        '8': _parse_section3_8,
        '9': _parse_section3_9,
    }

    ##
    # Fills the observation with information from section 3
    # @param obs - the observation to be filles
    # @param section3 a tokenized list of the groups in section 3
    def _fill_obs_with_section3(self, obs, section3):
        idx = 0
        while idx < len(section3):
            secpart = section3[idx][0]
            if secpart in self.SECTION3_PARTS:
                self.SECTION3_PARTS[secpart](self, obs, section3[idx])
            idx = idx + 1


if __name__ == "__main__":
    a = fm12_parser()
    # a.parse("A_SMNO01ENMI300600RRC_C_ENMI_20131030061900.txt")
    a.parse("simulated_gisc_20131031061530.txt")
