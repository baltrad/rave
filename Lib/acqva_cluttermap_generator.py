'''
Copyright (C) 2024- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Support for creating static cluttermaps that can be used by the ACQVA algorithm.

## @file
## @author Anders Henja, SMHI
## @date 2024-04-27
import _polarscan, _polarscanparam, _polarvolume
import _raveio
import rave_pgf_logger
import json
import math
import numpy as np
import re
import datetime

TUPLE_PATTERN=re.compile("\s*([+-]?[0-9]+(.[0-9]+)?)\s*(->\s*([+-]?[0-9]+(.[0-9]+)?))?\s*")

logger = rave_pgf_logger.create_logger()

# {
#   "acqva_static": {
#     "seang":[
#       {"elangles":"0.0 -> 10.0",
#        "ranges":"0.0 -> 24000.0",
#        "azimuths":"-45.0 -> 45.0",
#        "blocked":true
#       },
#       {"elangles":"10.1 -> 45.0",
#        "ranges":"0.0 -> 2000.0",
#        "azimuths":"-5.0 -> 5.0",
#        "blocked":true
#       }
#     ]
#   }
# }

class ParserException(Exception):
    """thrown when parsing failed
    """

class InitializerException(Exception):
    """thrown when parsing failed
    """

class GeneratorException(Exception):
    """thrown when generating failed
    """

class acqva_static_item(object):
    def __init__(self, elangles, scans, ranges, bins, azimuths, rays, blocked=True):
        if elangles is None and scans is None:
            raise InitializerException("must either specify elangles or scans")
        if ranges is None and bins is None:
            raise InitializerException("must either specify ranges or bins")
        if azimuths is None and rays is None:
            raise InitializerException("must either specify azimuths or rays")

        if (elangles and (len(elangles) < 1 or len(elangles) > 2)) or \
           (scans and (len(scans) < 1 or len(scans) > 2)) or \
           (ranges and (len(ranges) < 1 or len(ranges) > 2)) or \
           (bins and (len(bins) < 1 or len(bins) > 2)) or \
           (azimuths and (len(azimuths) < 1 or len(azimuths) > 2)) or \
           (rays and (len(rays) < 1 or len(rays) > 2)):
           raise InitializerException("elangles, scans, ranges, bins, azimuths and rays can either be one single item or a tuple with start and stop.")

        self._elangles = elangles
        self._scans = scans
        self._ranges = ranges
        self._bins = bins
        self._azimuths = azimuths
        self._rays = rays
        self._blocked = blocked

    def to_range(self, values):
        if values and len(values) > 0:
            estr = None
            if len(values) == 1:
                return f"{values[0]}"
            else:
                return f"{values[0]} -> {values[1]}"
        return None

    def __str__(self):
        result = {}
        if self._elangles and len(self._elangles) > 0:
            result["elangles"] = self.to_range(self._elangles)
        if self._ranges and len(self._ranges) > 0:
            result["ranges"] = self.to_range(self._ranges)
        if self._azimuths and len(self._azimuths) > 0:
            result["azimuths"] = self.to_range(self._azimuths)
        if self._scans and len(self._scans) > 0:
            result["scans"] = self.to_range(self._scans)
        if self._bins and len(self._bins) > 0:
            result["bins"] = self.to_range(self._bins)
        if self._rays and len(self._rays) > 0:
            result["rays"] = self.to_range(self._rays)
        result["blocked"] = self._blocked

        return json.dumps(result)

    def create_scans(self, volume):
        result = []
        volume.sortByElevations(1)
        if self._scans:
            if len(self._scans) == 1:
                if self._scans[0] >= 0 and self._scans[0] < volume.getNumberOfScans():
                    result.append(self._scans[0])
            else:
                for i in range(volume.getNumberOfScans()):
                    if i >= self._scans[0] and i <= self._scans[1]:
                        result.append(i)

        if self._elangles:
            if len(self._elangles) == 1:
                for i in range(volume.getNumberOfScans()):
                    elangle = volume.getScan(i).elangle * 180.0 / math.pi
                    if math.isclose(self._elangles[0], elangle, rel_tol=1e-03):
                        result.append(i)
                        break
            else:
                for i in range(volume.getNumberOfScans()):
                    elangle = volume.getScan(i).elangle * 180.0 / math.pi
                    if elangle >= self._elangles[0] and elangle <= self._elangles[1]:
                        result.append(i)

        return result

    def create_ray_range(self, nrays, r1, r2):
        result=[]
        while r1 < 0: r1 += nrays
        while r2 < 0: r2 += nrays
        while r1 >= nrays: r1 -= nrays
        while r2 >= nrays: r2 -= nrays
        if r1 <= r2:
            result.extend(range(r1,r2))
        else:
            # We must have a wrap around 0 unless they added strange values
            if r1 >= nrays/2 and r2 <= nrays/2:
                result.extend(range(r1, nrays))
                result.extend(range(0, r2))
        return result

    def create_rays(self, scan):
        result = []
        nrays = scan.nrays
        nbins = scan.nbins

        beamwidth = 360.0 / nrays   # Can't use scan.beamwidth since that can be wider

        if self._rays:
            if len(self._rays) == 1:
                if self._rays[0] >= 0 and self._rays[0] < nrays:
                    result.append(self._rays[0])
            else:
                r1 = self._rays[0]
                r2 = self._rays[1]
                result.extend(self.create_ray_range(nrays, r1, r2))

        if self._azimuths:
            if len(self._azimuths) == 1:
                rayi = int(self._azimuths[0]/beamwidth)
                if rayi >= 0 and rayi < nrays:
                    result.append(self._rays[0])
            else:
                r1 = int(self._azimuths[0] / beamwidth)
                r2 = int(self._azimuths[1] / beamwidth)
                result.extend(self.create_ray_range(nrays, r1, r2))

        return result

    def create_bins(self, scan):
        result = []
        nrays = scan.nrays
        nbins = scan.nbins
        rscale = scan.rscale
        if self._bins:
            if len(self._bins) == 1:
                if (self._bins[0] > 0 and self._bins[0] < nbins ):
                    result.append(self._bins[0])
            else:
                b1 = self._bins[0]
                b2 = self._bins[1]
                if b1 < b2 and b1 >= 0 and b2 >= 0:
                    nb1 = b1
                    if nb1 > nbins:
                        nb1 = nbins - 1
                    nb2 = b2
                    if nb2 > nbins:
                        nb2 = nbins
                    result.extend(range(nb1, nb2))

        if self._ranges:
            if rscale == 0.0:
                raise InitializerException("Rscale == 0.0. Can't process ranges.")
            if len(self._ranges) == 1:
                bin = int(self._ranges[0]/rscale)
                result.append(bin)
            else:
                b1 = int(self._ranges[0]/rscale)
                b2 = int(self._ranges[1]/rscale)
                if b1 < b2 and b1 >= 0 and b2 >= 0:
                    nb1 = b1
                    if nb1 > nbins:
                        nb1 = nbins - 1
                    nb2 = b2
                    if nb2 > nbins:
                        nb2 = nbins
                    result.extend(range(nb1, nb2))

        return result


class acqva_static_source(object):
    def __init__(self, source, geometry=None):
        self._source = source
        self._items = []
        self._geometry = {}
        if geometry:
            self._geometry = geometry
    
    def add(self, item):
        self._items.append(item)

    def items(self):
        return self._items

    def elangles(self):
        if "elangles" in self._geometry:
            return self._geometry["elangles"]
        return []

    def __str__(self):
        iteml = []
        for item in self._items:
            iteml.append("  " + str(item))
        result = f"{self._source}:[\n" + ",\n".join(iteml) + "\n]"
        return result


class acqva_cluttermap_generator(object):
    """Support for creating static cluttermaps from a json configuration file. The format of the json file should be according to acqva_static.json.
    """
    def __init__(self, configfile=None):
        self._configfile = configfile
        self._config = self.parse_config(self._configfile)

    def parse_config(self, configfile):
        """ Loads and validates the acqva static cluttermap definition. 
        :param configfile: The file to load
        :return: the config on success
        :throws: Exception if configuration not according to allowed format.
        """
        if configfile:
            with open(configfile, "r") as fp:
                config = json.load(fp)
                return self.jsonmap_to_config(config)
        return {}

    def parse_float_tuple(self, values):
        result = []
        if values:
            m = TUPLE_PATTERN.match(values)
            if m:
                result.append(float(m.group(1)))
                if m.group(4):
                    result.append(float(m.group(4)))
                return result
            else:
                raise ParserException(f"Could not parse float tuple% {values}")
        return None

    def parse_int_tuple(self, values):
        result = []
        if values:
            m = TUPLE_PATTERN.match(values)
            if m:
                result.append(int(m.group(1)))
                if m.group(4):
                    result.append(int(m.group(4)))
                return result
            else:
                raise ParserException(f"Could not parse int tuple% {values}")
        return None

    def jsonmap_to_config(self, config):
        """ Translates a json map according to format definition into a structure of objects..
        :config: a json structure according to spec
        :return: the object mapping
        :throws: Exception if there are problems with configuration
        """
        statics = config["acqva_static_volume"]
        result = {}
        for x in statics:
            items = statics[x]["blockings"]
            geometry = {}
            if "geometry" in statics[x]:
                geometry = statics[x]["geometry"]
            source = acqva_static_source(x, geometry)
            for item in items:
                azimuths = ranges = elangles = bins = rays = scans = None
                blocked = True
                if "elangles" in item:
                    elangles = self.parse_float_tuple(item["elangles"])
                if "azimuths" in item:
                    azimuths = self.parse_float_tuple(item["azimuths"])
                if "ranges" in item:
                    ranges = self.parse_float_tuple(item["ranges"])
                if "scans" in item:
                    scans = self.parse_int_tuple(item["scans"])
                if "bins" in item:
                    bins = self.parse_int_tuple(item["bins"])
                if "rays" in item:
                    rays = self.parse_int_tuple(item["rays"])
                source.add(acqva_static_item(elangles, scans, ranges, bins, azimuths, rays))
            result[x] = source
        return result

    def copy_volume(self, volume, source):
        dt = datetime.datetime.now()
        result = _polarvolume.new()
        result.date = dt.strftime("%Y%m%d")
        result.time = dt.strftime("%H%M%S")
        result.source = source
        nscans = volume.getNumberOfScans()
        for i in range(nscans):
            scan = volume.getScan(i)
            newscan = _polarscan.new()
            newscan.elangle = scan.elangle
            newscan.rscale = scan.rscale
            newscan.rstart = scan.rstart
            newscan.startdate = result.date
            newscan.enddate = result.date
            newscan.starttime = result.time
            newscan.endtime = result.time
            param = _polarscanparam.new()
            param.quantity = "ACQVA"
            param.gain = 1.0
            param.offset = 0.0
            param.nodata = 255.0
            param.undetect = 0.0            
            data = np.ones(scan.getParameter("DBZH").getData().shape, np.uint8)
            param.setData(data)
            newscan.addParameter(param)
            result.addScan(newscan)
        return result

    def create(self, volume, nod=None):
        if not _polarvolume.isPolarVolume(volume):
            raise Exception("Must provide a polar volume")

        src = volume.source

        if nod is None:
            toks = src.split(",")
            for t in toks:
                t = t.strip()
                if t.startswith("NOD:"):
                    nod = t.split(":")[1]
                    break
        if nod is None:
            raise GeneratorException("File is missing NOD in source and no NOD specified")

        if nod not in self._config:
            return None

        volume.sortByElevations(1)
        
        result = self.copy_volume(volume, f"NOD:{nod}")

        cfg = self._config[nod]
        for item in cfg.items():
            scans = item.create_scans(result)
            for si in scans:
                if si >= 0 and si < result.getNumberOfScans():
                    scan = result.getScan(si)
                    rays = item.create_rays(scan)
                    bins = item.create_bins(scan)
                    parameter = scan.getParameter("ACQVA")
                    for rayi in rays:
                        for bini in bins:
                            parameter.setValue((bini, rayi), 255)
        return result


if __name__=="__main__":
    generator = acqva_cluttermap_generator("/projects/baltrad/rave/config/acqva_static.json")
    #print(generator._config["seang"])
    #generator.create(_raveio.open("/projects/baltrad/rave/test/pytest/fixtures/sehem_pvol_pn215_20171204T071500Z_0x81540b.h5").object)
    result = generator.create(_raveio.open("/projects/baltrad/rave/test/pytest/fixtures/seang_qcvol_20120131T0000Z.h5").object)
    rio = _raveio.new()
    rio.object = result
    rio.save("cluttermap.h5")
    #print(create_rays([-20,20]))
    #print(create_rays([0,20]))
    #print(create_rays([340,20]))
    #print(create_rays([20,340]))
