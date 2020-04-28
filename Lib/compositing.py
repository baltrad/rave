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
## Python interface to composite generation functionality.

## @file
## @author Anders Henja and Daniel Michelson, SMHI
## @date 2014-08-12
import os  # Provisional, until compositing can handle prefab QC
import re
import sys

import _rave
import _polarscan
import _polarvolume
import _pycomposite
import _transform
import _area
import _projection
import _gra
import _raveio
import _bitmapgenerator

import string
import datetime
import math
import tempfile
import rave_pgf_logger
import rave_pgf_quality_registry
import rave_ctfilter
import rave_util
import area_registry
import rave_area
import odim_source
import rave_projection
import rave_quality_plugin
from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY   

from rave_defines import CENTER_ID, GAIN, OFFSET
from rave_defines import DEFAULTA, DEFAULTB, DEFAULTC

logger = rave_pgf_logger.create_logger()

ravebdb = None

try:
  import rave_bdb
  ravebdb = rave_bdb.rave_bdb()
except:
  pass

nodomdb=False
try:
  import rave_dom_db
except:
  nodomdb=True

is_py27=False
if sys.version_info < (3,):
    is_py27=True

##
# The area registry to be used by this composite generator.
my_area_registry = area_registry.area_registry()

##
# Compositing class instance
class compositing(object):
  ##
  # Constructor
  def __init__(self, rbdb=None):
    self.ravebdb = rbdb
    if self.ravebdb is None:
      self.ravebdb = ravebdb
    self.pcsid = "gmaps"
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
    self.interpolation_method = _pycomposite.InterpolationMethod_NEAREST
    self.quality_control_mode = rave_quality_plugin.QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
    self.qitotal_field = None
    self.applygra = False
    self.zr_A = 200.0
    self.zr_b = 1.6    
    self.applygapfilling = False
    self.applyctfilter = False
    self.quantity = "DBZH"
    self.gain = GAIN
    self.offset = OFFSET
    self.minvalue = -30.0
    self.reprocess_quality_field=False
    self.verbose = False
    self.logger = logger
    self.dumppath = None
    self.dump = False
    self.use_site_source = False
    self.use_azimuthal_nav_information = True
    self.radar_index_mapping = {}
    
  def generate(self, dd, dt, area=None):
    return self._generate(dd, dt, area)

  def _debug_generate_info(self, area):
    if self.verbose:
      self.logger.info("Generating cartesian image from %d files"%len(self.filenames))
      self.logger.debug("Detectors = %s"%str(self.detectors))
      self.logger.debug("Quality control mode = %s"%str(self.quality_control_mode))
      self.logger.debug("Product = %s"%self._product_repr())
      self.logger.debug("Quantity = %s"%self.quantity)
      self.logger.debug("Range = %f"%self.range)
      self.logger.debug("Gain = %f, Offset = %f, Minvalue = %f"%(self.gain, self.offset, self.minvalue))
      self.logger.debug("Prodpar = %s"%self.prodpar)
      self.logger.debug("Selection method = %s"%self._selection_method_repr())
      self.logger.debug("Interpolation method = %s"%str(self._interpolation_method_repr()))
      self.logger.debug("Gap filling = %s"%str(self.applygapfilling))
      self.logger.debug("Ct filtering = %s"%str(self.applyctfilter))
      self.logger.debug("Gra filtering = %s"%str(self.applygra))
      self.logger.debug("Ignoring malfunc = %s"%str(self.ignore_malfunc))
      self.logger.debug("QI-total field = %s"%self.qitotal_field)
      self.logger.debug("Reprocess quality fields = %s"%str(self.reprocess_quality_field))
      self.logger.debug("Dumping path = %s"%str(self.dumppath))
      self.logger.debug("Dumping output = %s"%str(self.dump))
      self.logger.debug("Use site source = %s"%str(self.use_site_source))
      
      if area is not None:
        self.logger.debug("Area = %s"%area)
      else:
        self.logger.debug("Area = best fit")
        self.logger.debug("  pcsid = %s"%self.pcsid)
        self.logger.debug("  xscale = %f, yscale = %f"%(self.xscale, self.yscale))    

  ## Removes CMT:<...> from the string
  # @param[in] str - the string from which CMT should be removed
  # @return the source with CMT removed
  #
  def remove_CMT_from_source(self, str):
    v=re.sub("CMT:[^,]+", "", str)
    v=re.sub(",,",",",v)
    if v.endswith(","):
      v=v[0:-1]
    if v.startswith(","):
      v=v[1:]
    return v

  ##
  # Returns the next available radar
  def get_next_radar_index(self):
    if len(self.radar_index_mapping)==0:
      return 1
    v = list(self.radar_index_mapping.values())
    v.sort()
    idx = 1
    for i in v:
      if idx != i:
        return idx
      idx = idx + 1
    return idx

  ## Generates the cartesian image.
  #
  # @param dd: date in format YYYYmmdd
  # @param dt: time in format HHMMSS
  # @param area: the area to use for the cartesian image. If none is specified, a best fit will be atempted.  
  def _generate(self, dd, dt, area=None):
    self._debug_generate_info(area)
 
    if self.verbose:
      self.logger.info("Fetching objects and applying quality plugins")
    
    self.logger.debug("Generating composite with date and time %sT%s for area %s", dd, dt, area)
    
    objects, nodes, how_tasks, all_files_malfunc = self.fetch_objects()
    
    if all_files_malfunc:
      self.logger.info("Content of all provided files were marked as 'malfunc'. Since option 'ignore_malfunc' is set, no composite is generated!")
      return None
    
    objects, algorithm, qfields = self.quality_control_objects(objects)
    
    self.logger.debug("Quality controls for composite generation: %s", (",".join(qfields)))
    
    if len(objects) == 0:
      self.logger.info("No objects provided to the composite generator. No composite will be generated!")
      return None

    objects=list(objects.values())

    if self.dump:
      self._dump_objects(objects)

    generator = _pycomposite.new()
    if area is not None:
      if _area.isArea(area):
        pyarea = area
      else:
        pyarea = my_area_registry.getarea(area)
    else:
      if self.verbose:
        self.logger.info("Determining best fit for area")
      A = rave_area.MakeAreaFromPolarObjects(objects, self.pcsid, self.xscale, self.yscale)

      pyarea = _area.new()
      pyarea.id = "auto-generated best-fit"
      pyarea.xsize = A.xsize
      pyarea.ysize = A.ysize
      pyarea.xscale = A.xscale
      pyarea.yscale = A.yscale
      pyarea.extent = A.extent
      pcs = rave_projection.pcs(A.pcs)
      pcsname = pcs.name
      if not is_py27:
        pcsname = pcsname.decode()
      pyarea.projection = _projection.new(pcs.id, pcsname, ' '.join(pcs.definition))
  
      if len(objects) == 1:
        try:
          tmpid = odim_source.NODfromSource(objects[0])
          pyarea.id = "auto_%s_%s"%(A.pcs, tmpid)
        except:
          pass
    
    generator.addParameter(self.quantity, self.gain, self.offset, self.minvalue)
    generator.product = self.product
    if algorithm is not None:
      generator.algorithm = algorithm
      
    for o in objects:
      generator.add(o)
      # We want to ensure that we get a proper indexing of included radar
      sourceid = o.source
      try:
        osource = odim_source.ODIM_Source(o.source)
        if osource.wmo:
          sourceid = "WMO:%s"%osource.wmo
        elif osource.rad:
          sourceid = "RAD:%s"%osource.rad
        elif osource.nod:
          sourceid = "NOD:%s"%osource.nod
      except:
        pass
      
      if not sourceid in self.radar_index_mapping.keys():
        self.radar_index_mapping[sourceid] = self.get_next_radar_index()
    
    generator.selection_method = self.selection_method
    generator.interpolation_method = self.interpolation_method
    generator.date=o.date if dd is None else dd 
    generator.time=o.time if dt is None else dt
    generator.height = self.height
    generator.elangle = self.elangle
    generator.range = self.range
    
    if self.qitotal_field is not None:
      generator.quality_indicator_field_name = self.qitotal_field
    
    if self.prodpar is not None:
      self._update_generator_with_prodpar(generator)
    
    if self.verbose:
      self.logger.info("Generating cartesian composite")
    
    generator.applyRadarIndexMapping(self.radar_index_mapping)
    
    result = generator.generate(pyarea, qfields)
    
    if self.applyctfilter:
      if self.verbose:
        self.logger.debug("Applying ct filter")
      rave_ctfilter.ctFilter(result, self.quantity)
    
    if self.applygra:
      if not "se.smhi.composite.distance.radar" in qfields:
        self.logger.info("Trying to apply GRA analysis without specifying a quality plugin specifying the se.smhi.composite.distance.radar q-field, disabling...")
      else:
        if self.verbose:
          self.logger.info("Applying GRA analysis (ZR A = %f, ZR b = %f)"%(self.zr_A, self.zr_b))
        grafield = self._apply_gra(result, dd, dt)
        if grafield:
          result.addParameter(grafield)
        else:
          self.logger.warn("Failed to generate gra field....")
    
    # Hack to create a BRDR field if the qfields contains se.smhi.composite.index.radar
    if "se.smhi.composite.index.radar" in qfields:
      bitmapgen = _bitmapgenerator.new()
      brdr_field = bitmapgen.create_intersect(result.getParameter(self.quantity), "se.smhi.composite.index.radar")
      brdr_param = result.createParameter("BRDR", _rave.RaveDataType_UCHAR)
      brdr_param.setData(brdr_field.getData())
      
    if self.applygapfilling:
      if self.verbose:
        self.logger.debug("Applying gap filling")
      t = _transform.new()
      gap_filled = t.fillGap(result)
      result.getParameter(self.quantity).setData(gap_filled.getParameter(self.quantity).getData())
    
    # Fix so that we get a valid place for /what/source and /how/nodes 
    plc = result.source
    result.source = "%s,CMT:%s"%(CENTER_ID,plc)
    result.addAttribute('how/nodes', nodes)
    if self.use_site_source and len(objects) == 1:
      try:
        result.source = objects[0].source
        if result.source.find("NOD:") == -1:
          tmpid = odim_source.NODfromSource(objects[0])
          result.source="%s,NOD:%s,CMT:%s"%(self.remove_CMT_from_source(result.source), tmpid, plc)
        else:
          result.source="%s,CMT:%s"%(self.remove_CMT_from_source(result.source), plc)
      except:
        self.logger.exception("Failed to get source from object")
        
    if how_tasks != "":
      result.addAttribute('how/task', how_tasks)

    if self.verbose:
      self.logger.debug("Returning resulting composite image")

    return result

  def set_product_from_string(self, prodstr):
    prodstr = prodstr.lower()

    if prodstr == "ppi":
      self.product = _rave.Rave_ProductType_PPI
    elif prodstr == "cappi":
      self.product = _rave.Rave_ProductType_CAPPI
    elif prodstr == "pcappi":
      self.product = _rave.Rave_ProductType_PCAPPI
    elif prodstr == "pmax":
      self.product = _rave.Rave_ProductType_PMAX
    elif prodstr == "max":
      self.product = _rave.Rave_ProductType_MAX
    else:
      raise ValueError("Only supported product types are ppi, cappi, pcappi, pmax and max")    
  
  def set_method_from_string(self, methstr):
    if methstr.upper() == "NEAREST_RADAR":
      self.selection_method = _pycomposite.SelectionMethod_NEAREST
    elif methstr.upper() == "HEIGHT_ABOVE_SEALEVEL":
      self.selection_method = _pycomposite.SelectionMethod_HEIGHT
    else:
      raise ValueError("Only supported selection methods are NEAREST_RADAR or HEIGHT_ABOVE_SEALEVEL")
    
  def set_interpolation_method_from_string(self, methstr):
    if methstr.upper() == "NEAREST_VALUE":
      self.interpolation_method = _pycomposite.InterpolationMethod_NEAREST
    elif methstr.upper() == "LINEAR_HEIGHT":
      self.interpolation_method = _pycomposite.InterpolationMethod_LINEAR_HEIGHT
    elif methstr.upper() == "LINEAR_RANGE":
      self.interpolation_method = _pycomposite.InterpolationMethod_LINEAR_RANGE
    elif methstr.upper() == "LINEAR_AZIMUTH":
      self.interpolation_method = _pycomposite.InterpolationMethod_LINEAR_AZIMUTH
    elif methstr.upper() == "LINEAR_RANGE_AND_AZIMUTH":
      self.interpolation_method = _pycomposite.InterpolationMethod_LINEAR_RANGE_AND_AZIMUTH
    elif methstr.upper() == "LINEAR_3D":
      self.interpolation_method = _pycomposite.InterpolationMethod_LINEAR_3D
    elif methstr.upper() == "QUADRATIC_HEIGHT":
      self.interpolation_method = _pycomposite.InterpolationMethod_QUADRATIC_HEIGHT
    elif methstr.upper() == "QUADRATIC_3D":
      self.interpolation_method = _pycomposite.InterpolationMethod_QUADRATIC_3D
    else:
      raise ValueError("Only supported interpolation methods are NEAREST_VALUE, LINEAR_HEIGHT, LINEAR_RANGE, LINEAR_AZIMUTH, LINEAR_RANGE_AND_AZIMUTH, LINEAR_3D, QUADRATIC_HEIGHT or QUADRATIC_3D")
  
  def set_quality_control_mode_from_string(self, modestr):
    if modestr.lower() not in [QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY]:
      raise ValueError("Invalid quality control mode (%s), only supported modes are analyze_and_apply or analyze"%modestr.lower())
    self.quality_control_mode = modestr.lower()
  
  def quality_control_objects(self, objects):
    algorithm = None
    result = {}
    qfields = []
    for k in objects.keys():
      obj = objects[k]
      for d in self.detectors:
        p = rave_pgf_quality_registry.get_plugin(d)
        if p != None:
          process_result = p.process(obj, self.reprocess_quality_field, self.quality_control_mode)
          if isinstance(process_result, tuple):
            obj = process_result[0]
            detector_qfields = process_result[1]
          else:
            obj = process_result
            detector_qfields = p.getQualityFields()
          for qfield in detector_qfields:
            if qfield not in qfields:
              qfields.append(qfield)
          na = None
          if isinstance(obj, tuple):
            obj,na = obj[0],obj[1]
          if na is None:
            na = p.algorithm()
          if algorithm == None and na != None: # Try to get the generator algorithm != None 
            algorithm = na

      result[k] = obj

    return result, algorithm, qfields
  
  ##
  # Generates the objects that should be used in the compositing.
  # returns a triplet with [objects], nodes (as comma separated string), 'how/tasks' (as comma separated string)
  #
  def fetch_objects(self):
    nodes = ""
    objects={}
    tasks = []
    malfunc_files = 0
    for fname in self.filenames:
      obj = None
      try:
        if self.ravebdb != None:
          obj = self.ravebdb.get_rave_object(fname)
        else:
          obj = _raveio.open(fname).object
      except IOError:
        self.logger.exception("Failed to open %s", fname)
      
      is_scan = _polarscan.isPolarScan(obj)
      if is_scan:
        is_pvol = False
      else:
        is_pvol = _polarvolume.isPolarVolume(obj)
        
      if not is_scan and not is_pvol:
        self.logger.warn("Input file %s is neither polar scan or volume, ignoring.", fname)
        continue
      
      # Force azimuthal nav information usage if requested
      obj.use_azimuthal_nav_information = self.use_azimuthal_nav_information
      
      if self.ignore_malfunc:
        obj = rave_util.remove_malfunc(obj)
        if obj is None:
          self.logger.info("Input file %s detected as 'malfunc', ignoring.", fname)
          malfunc_files += 1
          continue
      
      node = odim_source.NODfromSource(obj)
      
      if len(nodes):
        nodes += ",'%s'" % node
      else:
        nodes += "'%s'" % node
        
      objects[fname] = obj
          
      if is_scan:
        self.logger.debug("Scan used in composite generation - UUID: %s, Node: %s, Nominal date and time: %sT%s", fname, node, obj.date, obj.time)
        self.add_how_task_from_scan(obj, tasks)
      elif is_pvol:
        self.logger.debug("PVOL used in composite generation - UUID: %s, Node: %s, Nominal date and time: %sT%s", fname, node, obj.date, obj.time)
        for i in range(obj.getNumberOfScans()):
          scan = obj.getScan(i)
          self.add_how_task_from_scan(scan, tasks)
      
    how_tasks = ",".join(tasks)
    
    all_files_malfunc = (len(self.filenames) > 0 and malfunc_files == len(self.filenames))
    
    return objects, nodes, how_tasks, all_files_malfunc
  
  def add_how_task_from_scan(self, scan, tasks):
    if scan.hasAttribute('how/task'):
      how_task_string = scan.getAttribute('how/task')
      if how_task_string not in tasks:
        tasks.append(how_task_string)
  
  def create_filename(self, pobj):
    #_polarscan.isPolarScan(obj) and not _polarvolume.isPolarVolume(obj):
    if _polarvolume.isPolarVolume(pobj):
      ptype = "pvol"
    elif _polarscan.isPolarScan(pobj):
      ptype = "scan"
    else:
      try:
        ptype = pobj.getAttribute("what/object").tolower()
      except:
        ptype = "unknowntype"
    src = odim_source.NODfromSource(pobj)
    dstr = "19700101"
    tstr = "000000"
    try:
      dstr = pobj.date
      tstr = pobj.time
    except:
      pass
    
    t = tempfile.mkstemp(prefix="%s_%s_%s_%s_"%(ptype, src, dstr, tstr), suffix=".h5", dir=self.dumppath)
    os.close(t[0])
    return t[1]
  
  ##
  # Dumps the objects on the ingoing polar objects onto the file system. The names will contain a unique identifier
  # to allow for duplicate versions of the same object.
  # @param objects the objects to write to disk
  def _dump_objects(self, objects):
    for o in objects:
      filename = self.create_filename(o)
      rio = _raveio.new()
      rio.object = o
      rio.save(filename)
  
  ##
  # Returns the backup coefficients to use. First the newest coefficient between
  # dt - maxage <= found <= dt is located. If none is found, then the climatologic
  # coefficients are used instead.
  #
  def get_backup_gra_coefficient(self, db, agedt, nowdt):
    try:
      coeff = db.get_newest_gra_coefficient(agedt, nowdt)
      if coeff and not math.isnan(coeff.a) and not math.isnan(coeff.b) and not math.isnan(coeff.c):
        logger.info("Reusing gra coefficients from %s %s"%(coeff.date, coeff.time))
        return coeff.significant, coeff.points, coeff.loss, coeff.r, coeff.r_significant, coeff.corr_coeff, coeff.a, coeff.b, coeff.c, coeff.mean, coeff.stddev
    except Exception:
      logger.exception("Failed to aquire coefficients")

    logger.warn("Could not aquire coefficients newer than %s, defaulting to climatologic"%agedt.strftime("%Y%m%d %H:%M:%S"))
    return "False", 0, 0, 0.0, "False", 0.0, DEFAULTA, DEFAULTB, DEFAULTC, 0.0, 0.0
  
  ##
  # Apply gra coefficient adjustment.
  # @param result: The cartesian product to be adjusted
  # @param d: the date string representing now (YYYYmmdd)
  # @param t: the time string representing now (HHMMSS)
  # @return the gra field with the applied corrections
  def _apply_gra(self, result, d, t):
    if nodomdb:
      self.logger.info("Could not load rave_dom_db, probably due to missing dependencies like jprops or sqlalchemy, ignoring gra correction")
      return
    
    try:
      zrA = self.zr_A
      zrb = self.zr_b
      db = rave_dom_db.create_db_from_conf()
      dt = datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:]), int(t[:2]), int(t[2:4]), 0)
      dt = dt - datetime.timedelta(seconds=3600*12) # 12 hours back in time for now..
      
      gra = _gra.new()
      gra.A = DEFAULTA
      gra.B = DEFAULTB
      gra.C = DEFAULTC
      gra.zrA = zrA
      gra.zrb = zrb
    
      grac = db.get_gra_coefficient(dt)
      if grac != None and not math.isnan(grac.a) and not math.isnan(grac.b) and not math.isnan(grac.c):
        if self.verbose:
          self.logger.debug("Applying gra coefficients from database")
        gra.A = grac.a
        gra.B = grac.b
        gra.C = grac.c
      else:
        self.logger.info("Could not find coefficients for given time, trying to get aged or climatologic coefficients")
        nowdt = datetime.datetime(int(d[:4]), int(d[4:6]), int(d[6:]), int(t[:2]), int(t[2:4]), 0)
        agedt = nowdt - datetime.timedelta(seconds=3600 * 48) # 2 days back
        sig,pts,loss,r,rsig,corr,gra.A,gra.B,gra.C,mean,dev = self.get_backup_gra_coefficient(db, agedt, nowdt)
        
      dfield = result.findQualityFieldByHowTask("se.smhi.composite.distance.radar")
      param = result.getParameter(self.quantity)
      gra_field = gra.apply(dfield, param)
      gra_field.quantity = self.quantity + "_CORR"
      return gra_field
    except Exception:
      import traceback
      traceback.print_exc()
      self.logger.error("Failed to apply gra coefficients", exc_info=1)
    return None    
  
  ##
  # @return the string representation of the selection method
  def _selection_method_repr(self):
    if self.selection_method == _pycomposite.SelectionMethod_NEAREST:
      return "NEAREST_RADAR"
    elif self.selection_method == _pycomposite.SelectionMethod_HEIGHT:
      return "HEIGHT_ABOVE_SEALEVEL"
    
  ##
  # @return the string representation of the interpolation method
  def _interpolation_method_repr(self):
    if self.interpolation_method == _pycomposite.InterpolationMethod_NEAREST:
      return "NEAREST_VALUE"
    elif self.selection_method == _pycomposite.InterpolationMethod_LINEAR_HEIGHT:
      return "LINEAR_HEIGHT"
    elif self.selection_method == _pycomposite.InterpolationMethod_LINEAR_RANGE:
      return "LINEAR_RANGE"
    elif self.selection_method == _pycomposite.InterpolationMethod_LINEAR_AZIMUTH:
      return "LINEAR_AZIMUTH"
    elif self.selection_method == _pycomposite.InterpolationMethod_LINEAR_RANGE_AND_AZIMUTH:
      return "LINEAR_RANGE_AND_AZIMUTH"
    elif self.selection_method == _pycomposite.InterpolationMethod_LINEAR_3D:
      return "LINEAR_3D"
    elif self.selection_method == _pycomposite.InterpolationMethod_QUADRATIC_HEIGHT:
      return "QUADRATIC_HEIGHT"
    elif self.selection_method == _pycomposite.InterpolationMethod_QUADRATIC_3D:
      return "QUADRATIC_3D"

  ##
  # @return the string representation of the product type  
  def _product_repr(self):
    if self.product == _rave.Rave_ProductType_PPI:
      return "ppi"
    elif self.product == _rave.Rave_ProductType_CAPPI:
      return "cappi"
    elif self.product == _rave.Rave_ProductType_PCAPPI:
      return "pcappi"
    elif self.product == _rave.Rave_ProductType_PMAX:
      return "pmax"
    elif self.product == _rave.Rave_ProductType_MAX:
      return "max"
    else:
      return "unknown"
      
  ##
  # If prodpar has been set, the generator is updated with the apropriate values
  # @param generator: the generator to be updated with the information from the prodpar
  #
  def _update_generator_with_prodpar(self, generator):
    if self.prodpar is not None:
      if self.product in [_rave.Rave_ProductType_CAPPI, _rave.Rave_ProductType_PCAPPI]:
        try:
          generator.height = self._strToNumber(self.prodpar)
        except ValueError:
          pass
      elif self.product in [_rave.Rave_ProductType_PMAX]:
        if isinstance(self.prodpar, str):
          pp = self.prodpar.split(",")
          if len(pp) == 2:
            try:
              generator.height = self._strToNumber(pp[0].strip())
              generator.range = self._strToNumber(pp[1].strip())
            except ValueError:
              pass
          elif len(pp) == 1:
            try:
              generator.height = self._strToNumber(pp[0].strip())
            except ValueError:
              pass
      elif generator.product in [_rave.Rave_ProductType_PPI]:
        try:
          v = self._strToNumber(self.prodpar)
          generator.elangle = v * math.pi / 180.0
        except ValueError:
          pass

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

  def test_func(self, a):
    print("Called with area %s"%a)


## Main function. 
# @param options a set of parsed options from the command line 
def main(options):
  comp = compositing()

  comp.filenames = options.infiles.split(",")
  comp.detectors = options.qc.split(",")
  comp.quantity = options.quantity
  comp.set_product_from_string(options.product)
  comp.range = options.range
  comp.gain = options.gain
  comp.offset = options.offset
  comp.prodpar = options.prodpar
  comp.minvalue = options.minvalue
  comp.set_method_from_string(options.method)
  comp.qitotal_field = options.qitotal_field
  comp.pcsid = options.pcsid
  comp.xscale = options.scale
  comp.yscale = options.scale
  
  comp.zr_A = options.zr_A
  comp.zr_b = options.zr_b
  
  if options.gf:
    comp.applygapfilling = True
  if options.ctfilter:
    comp.applyctfilter = True
  if options.grafilter:
    comp.applygra = True
  if options.ignore_malfunc:
    comp.ignore_malfunc = True
  if options.verbose:
    comp.verbose = True
    
  result = comp.generate(options.date, options.time, options.area)
    
  rio = _raveio.new()
  rio.object = result
  rio.filename = options.outfile
  
  if comp.verbose:
    logger.info("Saving %s"%rio.filename)
  rio.save()

if __name__ == "__main__":
  from optparse import OptionParser

  usage = "usage: %prog -i <infile(s)> -o <outfile> [-a <area>] [args] [h]"
  usage += "\nGenerates weather radar composites directly from polar scans and volumes."
  parser = OptionParser(usage=usage)

  parser.add_option("-i", "--input", dest="infiles",
                    help="Name of input file(s) to composite, comma-separated in quotations.")

  parser.add_option("-o", "--output", dest="outfile",
                    help="Name of output file to write.")

  parser.add_option("-a", "--area", dest="area",
                    help="Name of Cartesian area to which to generate the composite. If not specified, a best fit composite will be created.")

  parser.add_option("-c", "--pcsid", dest="pcsid",
                    default="gmaps",
                    help="Name of the pcsid to use if the area should be automatically generated from a best fit. Default is 'gmaps'.")

  parser.add_option("-s", "--scale", dest="scale",
                    type="float", default=2000.0,
                    help="The x/y-scale to use if the area should be automatically generated from a best fit. Default is 2000.0.")

  parser.add_option("-q", "--quantity", dest="quantity",
                    default="DBZH",
                    help="The radar parameter to composite. Default=DBZH.")

  parser.add_option("-p", "--product", dest="product",
                    default="PCAPPI",
                    help="The type of Cartesian product to generate [PPI, CAPPI, PCAPPI, PMAX]. Default=PCAPPI.")

  parser.add_option("-P", "--prodpar", dest="prodpar",
                    type="float", default=1000.0,
                    help="Product parameter. For (P)CAPPIs it is the height of the desired layer. For PPIs, it is the elevation angle. Default=1000.0 (meters).")

  parser.add_option("-r", "--range", dest="range",
                    type="float", default=200000.0,
                    help="Maximum range to apply PMAX algorithm. Applies only to PMAX algorithm. Defaults to 200 km.")

  parser.add_option("-g", "--gain", dest="gain",
                    type="float", default=GAIN,
                    help="Linear gain applied to output data. Default=as defined in rave_defines.py.")

  parser.add_option("-O", "--offset", dest="offset",
                    type="float", default=OFFSET,
                    help="Linear offset applied to output data. Default=as defined in rave_defines.py.")
  
  parser.add_option("-M", "--minvalue", dest="minvalue",
                    type="float", default=-30.0,
                    help="Minimum value that can be represented in composite. Relevant when interpolation is performed. Default=-30.0")

  parser.add_option("-d", "--date", dest="date",
                    default=None,
                    help="Nominal date of the composite to be written. Defaults to the nominal date of the last input file.")

  parser.add_option("-t", "--time", dest="time",
                    default=None,
                    help="Nominal time of the composite to be written. Defaults to the nominal time of the last input file.")

  parser.add_option("-m", "--method", dest="method",
                    default="NEAREST_RADAR",
                    help="Compositing algorithm to apply. Current choices are NEAREST_RADAR or HEIGHT_ABOVE_SEALEVEL. Default=NEAREST_RADAR.")
  
  parser.add_option("-I", "--interpolation_method", dest="interpolation_method",
                    type="choice", choices=["NEAREST_VALUE", "LINEAR_HEIGHT", "LINEAR_RANGE", "LINEAR_AZIMUTH", "LINEAR_RANGE_AND_AZIMUTH", "LINEAR_3D", "QUADRATIC_HEIGHT", "QUADRATIC_3D"], default="NEAREST_VALUE",
                    help="Interpolation method to use in composite generation. Default=NEAREST_VALUE")

  parser.add_option("-Q", "--qc", dest="qc",
                    default="",
                    help="Which quality-controls to apply. Comma-separated, no white spaces. Default=None")

  parser.add_option("-G", "--gap-fill", action="store_true", dest="gf",
                    help="Gap-fill small holes in output composite. Default=False")

  parser.add_option("-C", "--ctfilter", action="store_true", dest="ctfilter",
                    help="Filter residual non-precipitation echoes using SAF-NWC cloud-type product. Default=False")

  parser.add_option("-A", "--applygra", action="store_true", dest="grafilter",
                    help="Applies the GRA correction coefficients. Default=False")

  parser.add_option("-y", "--zr_A", dest="zr_A",
                    type="float", default="200.0",
                    help="The ZR A attribute to use for the gra correction. Default=200.0")

  parser.add_option("-z", "--zr_b", dest="zr_b",
                    type="float", default="1.6",
                    help="The ZR b attribute to use for the gra correction. Default=200.0")

  parser.add_option("-F", "--qitotal_field", dest="qitotal_field",
                    default=None, help="The QI-total field to use when creating the composite from the qi-total Default=Not used.")

  parser.add_option("-I", "--ignore-malfunc", action="store_true", dest="ignore_malfunc",
                    help="If scans/volumes contain malfunc information. Don't use them in the composite. Default is to always use everything.")
  
  parser.add_option("-V", "--verbose", action="store_true", dest="verbose",
                    help="If the different steps should be displayed. I.e. verbose information.")
  
  (options, args) = parser.parse_args()

  if options.infiles != None and options.outfile != None:
    main(options)
  else:
    parser.print_help()
