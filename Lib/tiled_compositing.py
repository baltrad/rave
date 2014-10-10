'''
Copyright (C) 2014- Swedish Meteorological and Hydrological Institute (SMHI)

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
## Python interface to be able to perform tiled compositing. This functionallity relies on the use of multiprocessing and
# storing intermediate files representing tiles. When the other processes are finishied with the processing, the result is
# put back into the full area again.
#

## @file
# @author Anders Henja, SMHI (original odc version Daniel Michelson, SMHI) 
# @date 2014-09-09
#
import rave_tile_registry
import compositing
import multiprocessing, rave_mppool
import math, os
import rave_pgf_logger
import area_registry
import numpy
import _rave, _area, _pycomposite, _projection, _raveio, _polarscan, _polarvolume, _transform, _cartesianvolume
from rave_defines import CENTER_ID, GAIN, OFFSET
from rave_defines import RAVE_TILE_COMPOSITING_PROCESSES
import rave_tempfile

logger = rave_pgf_logger.rave_pgf_syslog_client()

##
# The area registry to be used by this composite generator.
my_area_registry = area_registry.area_registry()

##
# The basic area definition that should be transfered to the tiling compositing instance.
# This definition will be pickled and sent to the receiving product generator.
#
class tiled_area_definition(object):
  def __init__(self, id, pcsdef, xscale, yscale, xsize, ysize, extent):
    self.id = id
    self.pcsdef = pcsdef
    self.xscale = xscale
    self.yscale = yscale
    self.xsize = xsize
    self.ysize = ysize
    self.extent = extent
  
  def __repr__(self):
    return "<%s, scale=%f * %f, size=%d * %d, extent=%s />"%(self.pcsdef, self.xscale, self.yscale, self.xsize, self.ysize, `self.extent`)

##
# The argument wrapper so that the arguments can be transfered to the composite generator taking care of the tile.
#
class multi_composite_arguments(object):
  ##
  # Constructor
  def __init__(self):
    self.xscale = 2000.0
    self.yscale = 2000.0
    self.detectors = []
    self.filenames = []
    self.ignore_malfunc = False
    self.prodpar = None
    self.product = _rave.Rave_ProductType_PCAPPI
    self.height = 1000.0
    self.elangle = 0.0
    self.range = 200000.0
    self.selection_method = _pycomposite.SelectionMethod_NEAREST 
    self.qitotal_field = None
    self.applygra = False
    self.zr_A = 200.0
    self.zr_b = 1.6    
    self.applygapfilling = False
    self.applyctfilter = False
    self.quantity = "DBZH"
    self.gain = GAIN
    self.offset = OFFSET
    self.area_definition = None
  
  ##
  # Generate function. Basically same as calling compositing.generate but the pyarea is created from the
  # area definition.
  # @param dd: date
  # @param dt: time
  # @param tid: the area identifier (only used for identification purpose, actual area is taken from the area_definition)
  # @return a filename pointing to the tile
  def generate(self, dd, dt, tid):
    comp = compositing.compositing()
    comp.xscale = self.xscale
    comp.yscale = self.yscale
    comp.detectors = self.detectors
    comp.ignore_malfunc = self.ignore_malfunc
    comp.prodpar = self.prodpar
    comp.product = self.product
    comp.height = self.height
    comp.elangle = self.elangle
    comp.range = self.range
    comp.selection_method = self.selection_method
    comp.qitotal_field = self.qitotal_field
    comp.applygra = self.applygra
    comp.zr_A = self.zr_A
    comp.zr_b = self.zr_b
    comp.applygapfilling = self.applygapfilling
    comp.applyctfilter = self.applyctfilter
    comp.quantity = self.quantity
    comp.gain = self.gain
    comp.offset = self.offset    
    comp.filenames = self.filenames
    
    pyarea = _area.new()
    pyarea.id = "tiled area subset %s"%tid
    pyarea.xsize = self.area_definition.xsize
    pyarea.ysize = self.area_definition.ysize
    pyarea.xscale = self.area_definition.xscale
    pyarea.yscale = self.area_definition.yscale
    pyarea.extent = self.area_definition.extent
    pyarea.projection = _projection.new("dynamic pcsid", "dynamic pcs name", self.area_definition.pcsdef)    
    
    logger.info("Generating composite for tile %s"%self.area_definition.id)
    result = comp.generate(dd, dt, pyarea)
    logger.info("Finished generating composite for tile %s"%self.area_definition.id)
      
    fileno, outfile = rave_tempfile.mktemp(suffix='.h5', close="True")
  
    rio = _raveio.new()
    rio.object = result
    rio.filename = outfile
    rio.save()
  
    return (tid, rio.filename)

##
# The actual compositing instance forwarding requests to the tilers.
class tiled_compositing(object):
  ##
  # Constructor
  # @param c: the compositing instance
  def __init__(self, c):
    self.compositing = c
    self.verbose = c.verbose
    self.logger = logger
    self.file_objects = self._fetch_file_objects(c)

  def _fetch_file_objects(self, comp):
    result = {}
    self.logger.info("Fetching %d files for tiled compositing"%len(comp.filenames))
    for fname in comp.filenames:
      if comp.ravebdb != None:
        obj = comp.ravebdb.get_rave_object(fname)
      else:
        obj = _raveio.open(fname).object
      result[fname] = obj
    self.logger.info("Finished fetching %d files"%len(comp.filenames))
    return result

  ##
  # Creates the composite arguments that should be sent to one tiler.
  # @param adef: the area definition
  # @return the composite argument instance
  def _create_multi_composite_argument(self, adef=None):
    a = multi_composite_arguments()
    a.xscale = self.compositing.xscale
    a.yscale = self.compositing.yscale
    a.detectors = self.compositing.detectors
    a.ignore_malfunc = self.compositing.ignore_malfunc
    a.prodpar = self.compositing.prodpar
    a.product = self.compositing.product
    a.height = self.compositing.height
    a.elangle = self.compositing.elangle
    a.range = self.compositing.range
    a.selection_method = self.compositing.selection_method
    a.qitotal_field = self.compositing.qitotal_field
    a.applygra = self.compositing.applygra
    a.zr_A = self.compositing.zr_A
    a.zr_b = self.compositing.zr_b
    a.applygapfilling = self.compositing.applygapfilling
    a.applyctfilter = self.compositing.applyctfilter
    a.quantity = self.compositing.quantity
    a.gain = self.compositing.gain
    a.offset = self.compositing.offset
    a.area_definition = adef
    
    return a
  
  ##
  # Creates an area definition used for passing to the tiler
  # @param pyarea: the python c area object
  # @return the area defintion
  def _create_tiled_area_definition(self, pyarea):
    return tiled_area_definition(pyarea.id, pyarea.projection.definition, pyarea.xscale, pyarea.yscale, pyarea.xsize, pyarea.ysize, pyarea.extent)
  
  ##
  # Creates the list of arguments to be sent to the tilers. Each item in the returned list is supposed to represent one tile
  # @param dd: date
  # @param dt: time
  # @param aid: the area id (that might or not be tiled)
  # @return a list of argument lists
  def _create_arguments(self, dd, dt, pyarea):
    tiled_areas = rave_tile_registry.get_tiled_areas(pyarea)
    
    args=[]
    
    for t in tiled_areas:
      mcomp = self._create_multi_composite_argument(self._create_tiled_area_definition(t))
      args.append([mcomp, dd, dt, t.id])
    
    # Now, make sure we have the correct files in the various areas
    self._add_files_to_argument_list(args, tiled_areas)
    
    return args
  
  def _add_files_to_argument_list(self, args, tiled_areas):
    self.logger.info("Distributing polar objects among %d tiles"%len(args))

    # Loop through tile areas
    for i in range(len(tiled_areas)):
        p = tiled_areas[i].projection
        llx, lly, urx, ury = tiled_areas[i].extent

        # Loop through radars
        for k in self.file_objects.keys():
            v = self.file_objects[k]
            if not _polarscan.isPolarScan(v) and not _polarvolume.isPolarVolume(v):
                continue
            if _polarvolume.isPolarVolume(v):
                v = v.getScanWithMaxDistance()
            scan = v
            
            if self.compositing.quantity not in scan.getParameterNames():
                self.logger.info("Quantity %s not in data from %s" % (self.compositing.quantity, scan.source))
                continue

            bi = scan.nbins - 1
            
            # Loop around the scan
            for ai in range(scan.nrays):
                lon, lat = scan.getLonLatFromIndex(bi, ai)
                x, y = p.fwd((lon, lat))
                
                # If this position is inside the tile, then add the radar's file string to the list and then bail
                if x >= llx and x <= urx and y >= lly and y <= ury:
                    if not k in args[i][0].filenames:
                        args[i][0].filenames.append(k)
                        break # No need to continue

    for idx in range(len(args)):
      self.logger.info("Tile %s contains %d files and dimensions %i x %i"%(args[idx][0].area_definition.id, len(args[idx][0].filenames), args[idx][0].area_definition.xsize, args[idx][0].area_definition.ysize))
      
    self.logger.info("Finished splitting polar object")
       
  def _create_lon_lat_extent(self, carg):
    pj = _projection.new("x", "y", carg.area_definition.pcsdef)
    lllon,lllat = pj.inv((carg.area_definition.extent[0], carg.area_definition.extent[1]))
    urlon,urlat = pj.inv((carg.area_definition.extent[2], carg.area_definition.extent[3]))
    return (lllon,lllat,urlon,urlat)

  ##
  # Same as compositing generate but this is supposed to forward requests to a tiling mechanism
  # @param dd: date
  # @param dt: time
  # @param area: the area id
  def generate(self, dd, dt, area=None):
    pyarea = my_area_registry.getarea(area)

    args = self._create_arguments(dd, dt, pyarea)
    
    results = []
    
    ntiles = len(args)
    ncpucores = multiprocessing.cpu_count()

    nrprocesses = ntiles
    if not RAVE_TILE_COMPOSITING_PROCESSES is None:
       if nrprocesses > RAVE_TILE_COMPOSITING_PROCESSES:
         nrprocesses = RAVE_TILE_COMPOSITING_PROCESSES

    if nrprocesses > ncpucores:
      nrprocesses = ncpucores
    if nrprocesses == ncpucores and ncpucores > 1:
      nrprocesses = nrprocesses - 1 # We always want to leave at least one core for something else
    
    pool = rave_mppool.RavePool(nrprocesses)
    
    r = pool.map_async(comp_generate, args, callback=results.append)
    
    r.wait()

    self.logger.info("Finished processing tiles, combining tiles")
    objects = []
    try:
      for v in results[0]:
        o = _raveio.open(v[1]).object
        if _cartesianvolume.isCartesianVolume(o):
          o = o.getImage(0)        
        objects.append(o)
        
      t = _transform.new()

      result = t.combine_tiles(pyarea, objects)
      
      self.logger.info("Tiles combined")
            
      return result
    finally:
      if results != None:
        for v in results[0]:
          if v != None and v[1] != None and os.path.exists(v[1]):
            try:
              os.unlink(v[1])
            except Exception, e:
              logger.exception("Failed to unlink %s"%v[1])
    return None

def comp_generate(args):
  try:
    return args[0].generate(args[1], args[2], args[3])
  except Exception, e:
    logger.exception("Failed to call composite generator in tiler")
  return None
    
if __name__=="__main__":
  comp = compositing.compositing()
  comp.filenames=["/projects/baltrad/rave/test/pytest/fixtures/pvol_seang_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_searl_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_sease_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_sehud_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_sekir_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_sekkr_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_selek_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_selul_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_seosu_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_seovi_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_sevar_20090501T120000Z.h5",
                  "/projects/baltrad/rave/test/pytest/fixtures/pvol_sevil_20090501T120000Z.h5"]
  
  comp.filenames=["/projects/baltrad/baltrad-test/fixtures4/seang_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/searl_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/sease_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/sehud_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/sekir_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/sekkr_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/selek_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/selul_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/seosu_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/seovi_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/sevar_pvol_20140220T1300Z.h5",
                  "/projects/baltrad/baltrad-test/fixtures4/sevil_pvol_20140220T1300Z.h5"]
  
  tc = tiled_compositing(comp)
  #tc.generate("20090501","120000", "swegmaps_2000")
  tc.generate("20090501","120000", "bltgmaps_4000")
  #execute_tiled_compositing(comp, my_area_registry.getarea("swegmaps_2000"))
