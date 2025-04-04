#include "tiled_compositing.h"
#include "thread_pool_executor.hpp"
#include "cartesian.h"
#include "compositing.h"
#include "rave_defines.h"

#include "raveobject_list.h"

extern "C" {
#include "rave_attribute.h"
#include "composite.h"
#include "rave_debug.h"
#include "rave_io.h"
#include "rave_object.h"
#include "rave_types.h"
#include "polarscan.h"
#include "polarvolume.h"
#include "arearegistry.h"
#include "area.h"
#include "projection.h"
#include "tiledef.h"
#include "tileregistry.h"
#include "cartesianvolume.h"
#include "transform.h"
#include "odim_source.h"
}

#include <sys/sysinfo.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>

#include <algorithm>
#include <sstream>
#include <iostream>
#include <map>
#include <mutex>



std::mutex multi_composite_arguments::mutex;

/**
 * FIXME: DEPRICATED documentation. This is taken from the Python code.
 * The basic area definition that should be transfered to the tiling compositing instance.
 * This definition will be pickled and sent to the receiving product generator.
 */
tiled_area_definition::tiled_area_definition() = default;
tiled_area_definition::~tiled_area_definition() = default;

void tiled_area_definition::init(const char* id,
                                 const char* pcsdef,
                                 double xscale,
                                 double yscale,
                                 int xsize,
                                 int ysize,
                                 const char* extent)
{
  _id.assign(id);
  _pcsdef.assign(pcsdef);
  _xscale = xscale;
  _yscale = yscale;
  _xsize = xsize;
  _ysize = ysize;
  _extent.assign(extent);
}

const char* tiled_area_definition::repr()
{
  thread_local char buffer[256];
  snprintf(buffer,
           sizeof(buffer) - 1,
           "<%s, scale=%f * %f, size=%d * %d, extent=%s />",
           _pcsdef.c_str(),
           _xscale,
           _yscale,
           _xsize,
           _ysize,
           _extent.c_str());
  return buffer;
}

/**
 * FIXME: Python documentation modify to align with C++.
 * The argument wrapper so that the arguments can be transfered to the composite generator taking care of the tile.
 * The parameters to the tiler.
 */
multi_composite_arguments::multi_composite_arguments()
{
  init();
}

multi_composite_arguments::~multi_composite_arguments()
{
  if (radar_index_mapping != NULL) {
    RAVE_OBJECT_RELEASE(radar_index_mapping);
  }
  for (auto & k : _file_objects) {
    RaveCoreObject* v = k.second;
    if (v != 0) {
      RAVE_OBJECT_RELEASE(v)
    }
  }
}

void multi_composite_arguments::init()
{
  // Inits the class with suitable defaults.
  xscale = 2000.0;
  yscale = 2000.0;
  ignore_malfunc = false;
  product = Rave_ProductType::Rave_ProductType_PCAPPI;
  height = 1000.0, elangle = 0.0;
  range = 200000.0;
  selection_method = CompositeSelectionMethod_t::CompositeSelectionMethod_NEAREST;
  interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_NEAREST;
  applygra = false;
  zr_A = 200.0;
  zr_b = 1.6;
  applygapfilling = false;
  applyctfilter = false;
  quantity = "DBZH";
  gain = 0.4;
  offset = -30.0;
  minvalue = -30.0;
  reprocess_quality_field = false;
  area_definition = 0;
  verbose = false;
  dump = false;
  radar_index_mapping = 0;
  use_lazy_loading = false;
  use_lazy_loading_preloads = false;
}

void multi_composite_arguments::set_area_definition(tiled_area_definition* areadef)
{
  area_definition = areadef;
}

/**
 * @brief Basically same as calling compositing.generate but the pyarea is created from the area definition.
 * @param dd: date
 * @param dt: time
 * @param tid: the area identifier (only used for identification purpose, actual area is taken from the area_definition)
 * @return a filename pointing to the tile
 */
result_from_tiler multi_composite_arguments::generate(std::string dd, std::string dt, std::string tid)
{
  // mpname = multiprocessing.current_process().name
  result_from_tiler tile_result;
  std::time_t starttime = std::time(0);
  std::string mpname = "radarcomp_c";
  Compositing comp;
  // starttime = time.time()
  comp.xscale = xscale;
  comp.yscale = yscale;
  comp.detectors = detectors;
  comp.ignore_malfunc = ignore_malfunc;
  comp.prodpar = prodpar;
  comp.product = product;
  comp.height = height;
  comp.elangle = elangle;
  comp.range = range;
  comp.selection_method = selection_method;
  comp.interpolation_method = interpolation_method;
  comp.qitotal_field = qitotal_field;
  comp.applygra = applygra;
  comp.zr_A = zr_A;
  comp.zr_b = zr_b;
  comp.applygapfilling = applygapfilling;
  comp.applyctfilter = applyctfilter;
  comp.quantity = quantity;
  comp.gain = gain;
  comp.offset = offset;
  comp.minvalue = minvalue;
  comp.filenames = _filenames;
  comp.verbose = verbose;
  comp.reprocess_quality_field = reprocess_quality_field;
  comp.dump = dump;
  comp.dumppath = dumppath;
  comp.radar_index_mapping = (RaveObjectHashTable_t*)RAVE_OBJECT_CLONE(radar_index_mapping);
  comp.use_lazy_loading = use_lazy_loading;
  comp.use_lazy_loading_preloads = use_lazy_loading_preloads;


  Area_t* area = (Area_t*)RAVE_OBJECT_NEW(&Area_TYPE);
  if (area != NULL) {
    char temp_areaid[256];
    snprintf(temp_areaid, sizeof(temp_areaid), "tiled area subset %s", tid.c_str());
    Area_setID(area, temp_areaid);
    Area_setXSize(area, area_definition->getXSize());
    Area_setYSize(area, area_definition->getYSize());
    Area_setXScale(area, area_definition->getXScale());
    Area_setYScale(area, area_definition->getYScale());
    double llX, llY, urX, urY;
    llX = llY = urX = urY = 0.;
    std::string the_extent = area_definition->getExtent();
    std::stringstream ss(the_extent);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, ',')) {
      elems.push_back(item);
    }
    if (elems.size() == 4) {
      llX = strtod(elems[0].c_str(), NULL);
      llY = strtod(elems[1].c_str(), NULL);
      urX = strtod(elems[2].c_str(), NULL);
      urY = strtod(elems[3].c_str(), NULL);
    }
    Area_setExtent(area, llX, llY, urX, urY);
    Projection_t* proj = (Projection_t*)RAVE_OBJECT_NEW(&Projection_TYPE);
    if (proj != NULL) {
      Projection_init(proj, "dynamic pcsid", "dynamic pcs name", area_definition->getPcsdef().c_str());
      Area_setProjection(area, proj);
    }
  }
  RAVE_DEBUG2("[%s] multi_composite_arguments.generate: Generating composite tile=%s",
              mpname.c_str(),
              area_definition->getId().c_str());

  // NOTE: the tiled area is not in area_registry, call generate with dummy argument as areaid.
  std::string dummy_areaid;
  Cartesian_t* result = comp.generate(dd, dt, dummy_areaid, area);
  if (result == 0) {
    std::time_t totaltime = std::time(0) - starttime;
    RAVE_INFO2("[%s] multi_composite_arguments.generate: No composite for tile=%s could be generated.",
               mpname.c_str(),
               area_definition->getId().c_str());
    RAVE_OBJECT_RELEASE(area);
    tile_result.tileid = tid;
    tile_result.filename.clear();
    tile_result.totaltime = totaltime;
    return tile_result;
  } else {
    RAVE_DEBUG2("[%s] multi_composite_arguments.generate: Finished generating composite for tile=%s.",
                mpname.c_str(),
                area_definition->getId().c_str());

    char tempname[16];
    strcpy(tempname, "/tmp/fileXXXXXX");

    int fd = mkstemp(tempname);
    close(fd);
    unlink(tempname);

    std::string outfile = tempname;
    outfile += ".h5";
    std::unique_lock<std::mutex> lock(mutex);
    RaveIO_t* rio = (RaveIO_t*)RAVE_OBJECT_NEW(&RaveIO_TYPE);
    if (rio == 0) {
      RAVE_CRITICAL0("Failed to allocate memory for raveIO.");
      RAVE_OBJECT_RELEASE(result);
      tile_result.tileid = tid;
      tile_result.filename.clear();
      tile_result.totaltime = 0;
      return tile_result;
    }
    RaveIO_setObject(rio, (RaveCoreObject*)result);
    RaveIO_setFilename(rio, outfile.c_str());
    if (comp.verbose) {
      RAVE_INFO1("Saving %s", RaveIO_getFilename(rio));
    }
    RaveIO_save(rio, 0);
    RaveIO_close(rio);
    RAVE_OBJECT_RELEASE(result);
    std::time_t totaltime = std::time(0) - starttime;
    tile_result.tileid = tid;
    tile_result.filename = outfile;
    tile_result.totaltime = totaltime;
    return tile_result;
  }
}

// The thread func for tile generation.

/**
 * FIXME: Python documentation modify to align with C++.
 * Function that handles the multiprocessing call for the multiprocessing
 * @param args: tuple of 4 args (multi_composite_arguments, date, time, area identifier)
 * @return result of multi_composite_arguments.generate
 */
result_from_tiler comp_generate(args_to_tiler arg)
{
  result_from_tiler result;
  // mpname = multiprocessing.current_process().name
  // starttime = time.time()
  std::time_t starttime = std::time(0);
  RAVE_INFO3("[radarcomp_c] tiled_compositing.comp_generate. Starting generation of tile %s - %s%s",
             arg.areaid.c_str(),
             arg.dd.c_str(),
             arg.tt.c_str());
  try {
    result = arg.mcomp->generate(arg.dd, arg.tt, arg.areaid);
    std::time_t etime = std::time(0) - starttime;
    RAVE_INFO2("[radarcomp_c] tiled_compositing.comp_generate. Finished creating tile %s in %d seconds",
               arg.areaid.c_str(),
               etime);
    return result;
  } catch (...) {
    return result;
  }
}

TiledCompositing::TiledCompositing()
{}

TiledCompositing::~TiledCompositing()
{}

void TiledCompositing::init(Compositing* c,
                            bool & preprocess_qc,
                            bool & mp_process_qc,
                            bool & mp_process_qc_split_evenly)
{
  // self.mpname = multiprocessing.current_process().name
  compositing = c;
  /*# If preprocess_qc = False, then the tile generators will take care of the preprocessing of the tiles
  # otherwise, the files will be qc-processed, written to disk and these filepaths will be sent to the
  # tile generators instead. Might or might not improve performance depending on file I/O etc..*/
  _preprocess_qc = preprocess_qc;
  _mp_process_qc = mp_process_qc;
  _mp_process_qc_split_evenly = mp_process_qc_split_evenly;
  verbose = c->verbose;
  // self.logger = logger
  // self.file_objects = {}
  _nodes = "";
  _how_tasks = "";
  number_of_quality_control_processes = RAVE_QUALITY_CONTROL_PROCESSES;
  _do_remove_temporary_files = false;
  mpname = "radarcomp_c";
  // init the inherited member variables
  xscale = compositing->xscale;
  yscale = compositing->yscale;
  detectors = compositing->detectors;
  ignore_malfunc = compositing->ignore_malfunc;
  prodpar = compositing->prodpar;
  product = compositing->product;
  height = compositing->height;
  elangle = compositing->elangle;
  range = compositing->range;
  selection_method = compositing->selection_method;
  interpolation_method = compositing->interpolation_method;
  qitotal_field = compositing->qitotal_field;
  applygra = compositing->applygra;
  zr_A = compositing->zr_A;
  zr_b = compositing->zr_b;
  applygapfilling = compositing->applygapfilling;
  applyctfilter = compositing->applyctfilter;
  quantity = compositing->quantity;
  gain = compositing->gain;
  offset = compositing->offset;
  dump = compositing->dump, dumppath = compositing->dumppath;
  reprocess_quality_field = compositing->reprocess_quality_field;
  use_lazy_loading_preloads = compositing->use_lazy_loading_preloads;
  use_lazy_loading = compositing->use_lazy_loading;
}

/**
 * FIXME: Python documentation modify to align with C++.
 * Fetches the file objects including the quality control by utilizing the multiprocessing capabilities.
 * I.e. instead of performing the fetching and quality controls within this process.
 * This job is spawned of to separate processors that manages the qc.
 * This will generate a number of temporary files that should be removed if self._do_remove_temporary_files=True
 *
 * @return (a dictionary with filenames as keys and objects as values, and a string with all included nodes names)
 */
std::map<std::string, RaveCoreObject*> TiledCompositing::_fetch_file_objects_mp(std::string & nodes, std::string & how_tasks, bool & all_files_malfunc)
{
  RAVE_INFO2("[%s] tiled_compositing.fetch_file_objects_mp: Fetching (and processing) %d files for tiled compositing",
             mpname.c_str(),
             compositing->filenames.size());

  std::map<std::string, RaveCoreObject*> result;

  std::vector<args_to_qc> args;
  args_to_qc arg;
  int ncpucores = get_nprocs();
  int nobjects = 0;
  int nrfiles = 0;
  int nrprocesses = 0;
  int nrslices = 0;

  _do_remove_temporary_files = false;
  /*
  # We want to determine how many processes we are going to get prior
  # splitting the files
  # */
  if (_mp_process_qc_split_evenly && number_of_quality_control_processes > 0) {
    nobjects = compositing->filenames.size();
    nrfiles = compositing->filenames.size();
    nrprocesses = number_of_quality_control_processes;
    if (nrprocesses > ncpucores) {
      nrprocesses = ncpucores;
    }
    if ((nrprocesses == ncpucores) && (ncpucores > 1)) {
      nrprocesses = nrprocesses - 1;
      nrslices = nrfiles / nrprocesses;
    }
    int nrfiles_per_slice = nrfiles / nrslices;
    for (int x = 0; x < nrslices; x++) {
      for (int y = 0; y < nrfiles_per_slice; y++) {
        int index = x * y;
        arg.filenames.push_back(compositing->filenames[index]);
      }
      arg.detectors = compositing->detectors;
      arg.reprocess_quality_field = compositing->reprocess_quality_field;
      arg.ignore_malfunc = compositing->ignore_malfunc;
      args.push_back(arg);
    }
    // args.append((self.compositing.filenames[x:x+nrslices], self.compositing.detectors,
    // self.compositing.reprocess_quality_field,  self.compositing.ignore_malfunc))
  } else {
    for (auto const & fname : compositing->filenames) {
      arg.filenames.push_back(fname);
      arg.detectors = compositing->detectors;
      arg.reprocess_quality_field = compositing->reprocess_quality_field;
      arg.ignore_malfunc = compositing->ignore_malfunc;
      args.push_back(arg);
    }
  }

  nobjects = args.size();
  nrprocesses = nobjects;
  if (nrprocesses > number_of_quality_control_processes) {
    nrprocesses = number_of_quality_control_processes;
  }
  if (nrprocesses > ncpucores) {
    nrprocesses = ncpucores;
  }
  if ((nrprocesses == ncpucores) && (ncpucores > 1)) {
    nrprocesses = nrprocesses - 1;  // # We always want to leave at least one core for something else
  }
  /* clang-format off */
  /*
   * FIXME: to be implemented!
  pool = multiprocessing.Pool(nrprocesses)

  results = [] //# Storage for the result from the mp processes
  r = pool.map_async(execute_quality_control, args, callback=results.append)

  r.wait()
  pool.terminate()
  pool.join()
  */
  // std::vector<std::string> filenames;
  /*
  for r in results[0]:
    filenames.extend(r[0])
      if r[1] == False:
        self.logger.info(f"[{self.mpname}] tiled_compositing.fetch_file_objects: quality control processing of %s
  failed."%str(r[2]))
   */
  /* clang-format on */
  compositing->filenames = filenames;
  _do_remove_temporary_files = true;

  result = compositing->fetch_objects(nodes, how_tasks, all_files_malfunc);

  RAVE_DEBUG2(
    "[%s] tiled_compositing.fetch_file_objects_mp: Finished fetching (and processing) %d files for tiled compositing",
    mpname.c_str(),
    compositing->filenames.size());

  return result;
}
/**
 * FIXME: Python documentation modify to align with C++.
 * Fetches the file objects and if self.preprocess_qc is True performs the quality controls.
 * If quality control processing is performed successfully, then temporary files are created
 * and their removal is required if self._do_remove_temporary_files = True
 *
 * @return (a dictionary with filenames as keys and objects as values, and a string with all included nodes names)
 */
std::map<std::string, RaveCoreObject*> TiledCompositing::_fetch_file_objects(std::string & nodes, std::string & how_tasks, bool & all_files_malfunc)
{
  RAVE_DEBUG2("[%s] tiled_compositing.fetch_file_objects: Fetching (and processing) %d files for tiled compositing",
              mpname.c_str(),
              compositing->filenames.size());

  std::map<std::string, RaveCoreObject*> result;

  result = compositing->fetch_objects(nodes, how_tasks, all_files_malfunc);
  if (_preprocess_qc) {
    RAVE_DEBUG0("quality_control_objects not implemented yet!");
    /*
    _do_remove_temporary_files=false;
    result, algorithm, qfields = self.compositing.quality_control_objects(result)
    try:
      result = store_temporary_files(result)
      self.compositing.filenames = result.keys()
      self._do_remove_temporary_files=True
    except Exception:
      self.logger.exception(f"[{self.mpname}] tiled_compositing.fetch_file_objects: Failed to create temporary files.
    will not preprocess qc.")
      */
  }
  RAVE_DEBUG2(
    "[%s] tiled_compositing.fetch_file_objects: Finished fetching (and processing) %d files for tiled compositing",
    mpname.c_str(),
    compositing->filenames.size());

  return result;
}

/**
 * FIXME: Python documentation modify to align with C++.
 * Get all the tiled areas belonging to the specified area.
 * The area has to reside in the area_registry in order for this area to be registered.
 * @param a the AreaCore (_area) instance
 * @return: a list of tiled area definitions
 */
RaveObjectList_t* TiledCompositing::_get_tiled_areas(Area_t* area)
{
  RaveObjectList_t* the_tiles = TileRegistry_getByArea(compositing->tile_registry, Area_getID(area));

  if (RaveObjectList_size(the_tiles) == 0) {
    RAVE_CRITICAL1("No such area (%s) with tiles defined", Area_getID(area));
    RAVE_OBJECT_RELEASE(the_tiles);
    return 0;
  }

  RaveObjectList_t* tiledareas = (RaveObjectList_t*)RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (tiledareas == NULL) {
    RAVE_CRITICAL0("Failed to create list for tiled areas.");
    RAVE_OBJECT_RELEASE(the_tiles);
    return 0;
  }

  double llX, llY, urX, urY;
  char the_area_id[256];
  int xsize, ysize;
  double area_xscale = Area_getXScale(area);
  double area_yscale = Area_getYScale(area);
  Projection_t* the_projection = Area_getProjection(area);
  int i = 0;
  for (i = 0; i < RaveObjectList_size(the_tiles); i++) {
    TileDef_t* ta = (TileDef_t*)RaveObjectList_get(the_tiles, i);
    Area_t* the_area = (Area_t*)RAVE_OBJECT_NEW(&Area_TYPE);
    if (the_area != NULL) {
      std::snprintf(the_area_id, sizeof(the_area_id) - 1, "%s_%s", Area_getID(area), TileDef_getID(ta));
      Area_setID(the_area, the_area_id);
      TileDef_getExtent(ta, &llX, &llY, &urX, &urY);
      xsize = (int)(round((urX - llX) / area_xscale));
      ysize = (int)(round((urY - llY) / area_yscale));
      Area_setXSize(the_area, xsize);
      Area_setYSize(the_area, ysize);
      Area_setXScale(the_area, area_xscale);
      Area_setYScale(the_area, area_yscale);
      Area_setProjection(the_area, the_projection);
      Area_setExtent(the_area, llX, llY, urX, urY);
      RaveObjectList_add(tiledareas, (RaveCoreObject*)the_area);
    }
  }
  RAVE_OBJECT_RELEASE(the_tiles);
  return tiledareas;
}

/**
 * @brief Creates the composite arguments that should be sent to one tiler.
 * @param adef: the area definition
 * @return the composite argument instance
 * */
multi_composite_arguments* TiledCompositing::_create_multi_composite_argument(tiled_area_definition* adef)
{
  multi_composite_arguments* a = new multi_composite_arguments();

  a->xscale = compositing->xscale;
  a->yscale = compositing->yscale;
  a->detectors = compositing->detectors;
  a->ignore_malfunc = compositing->ignore_malfunc;
  a->prodpar = compositing->prodpar;
  a->product = compositing->product;
  a->height = compositing->height;
  a->elangle = compositing->elangle;
  a->range = compositing->range;
  a->selection_method = compositing->selection_method;
  a->interpolation_method = compositing->interpolation_method;
  a->qitotal_field = compositing->qitotal_field;
  a->applygra = compositing->applygra;
  a->zr_A = compositing->zr_A;
  a->zr_b = compositing->zr_b;
  a->applygapfilling = compositing->applygapfilling;
  a->applyctfilter = compositing->applyctfilter;
  a->quantity = compositing->quantity;
  a->gain = compositing->gain;
  a->offset = compositing->offset;
  a->verbose = verbose;
  a->dump = compositing->dump, a->dumppath = compositing->dumppath;
  a->reprocess_quality_field = compositing->reprocess_quality_field;
  a->area_definition = adef;
  a->use_lazy_loading = compositing->use_lazy_loading;
  a->use_lazy_loading_preloads = compositing->use_lazy_loading_preloads;
  a->use_legacy_compositing = compositing->use_legacy_compositing;
  a->strategy = compositing->strategy;

  return a;
}

/**
 * @brief Creates an area definition used for passing to the tiler
 * @param pyarea: the python c area object
 * @return the area defintion
 */
tiled_area_definition* TiledCompositing::_create_tiled_area_definition(Area_t* area)
{
  // NOTE: extent as a string.
  tiled_area_definition* result = new tiled_area_definition();
  double llX, llY, urX, urY;
  Area_getExtent(area, &llX, &llY, &urX, &urY);
  std::ostringstream ost;
  ost << llX << "," << llY << "," << urX << "," << urY << std::ends;
  result->init(Area_getID(area),
               Projection_getDefinition(Area_getProjection(area)),
               Area_getXScale(area),
               Area_getYScale(area),
               Area_getXSize(area),
               Area_getYSize(area),
               ost.str().c_str());
  return result;
}

void TiledCompositing::_add_files_to_argument_list(std::vector<args_to_tiler> & args, RaveObjectList_t* tiled_areas)
{
  RAVE_DEBUG2("[%s] tiled_compositing._add_files_to_argument_list: Distributing polar objects among %d tiles",
              mpname.c_str(),
              args.size());

  // # Loop through tile areas
  for (int i = 0; i < RaveObjectList_size(tiled_areas); i++) {
    Area_t* ta = (Area_t*)RaveObjectList_get(tiled_areas, i);
    Projection_t* p = Area_getProjection(ta);
    double llx, lly, urx, ury;
    Area_getExtent(ta, &llx, &lly, &urx, &ury);

    // # Loop through radars
    //   std::map<std::string,RaveCoreObject*> file_objects;
    for (auto const & k : file_objects) {
      RaveCoreObject* v = k.second;
      // Jump over garbage in map.
      bool is_scan = RAVE_OBJECT_CHECK_TYPE(v, &PolarScan_TYPE);
      bool is_volume = RAVE_OBJECT_CHECK_TYPE(v, &PolarVolume_TYPE);
      if (!is_scan && !is_volume) {
        // Garbage found in map
        continue;
      }
      // scan is a read only
      PolarScan_t* scan = (PolarScan_t*)v;
      if (is_volume) {
        scan = PolarVolume_getScanWithMaxDistance((PolarVolume_t*)v);
      }

      if (!PolarScan_hasParameter(scan, compositing->quantity.c_str())) {
        RAVE_INFO3("[%s] tiled_compositing._add_files_to_argument_list: Quantity %s not in data from %s",
                   mpname.c_str(),
                   compositing->quantity.c_str(),
                   PolarScan_getSource(scan));
        continue;
      }
      int bi = PolarScan_getNbins(scan) - 1;

      // # Loop around the scan
      int nrays = PolarScan_getNrays(scan);

      for (int ai = 0; ai < nrays; ai++) {
        double lon, lat, x, y;
        PolarScan_getLonLatFromIndex(scan, bi, ai, &lon, &lat);
        Projection_fwd(p, lon, lat, &x, &y);

        // # If this position is inside the tile, then add the radar's file string to the list and then bail
        if (x >= llx && x <= urx && y >= lly && y <= ury) {
          bool found = false;
          for (auto const & j : args[i].mcomp->_filenames) {
            if (k.first == j) {
              found = true;
              break;  // # No need to continue
            }
          }
          if (found) {
            break;
          } else {
            args[i].mcomp->_filenames.push_back(k.first);
            // Copy the object for thread safety?
            // RaveCoreObject * the_object = (RaveCoreObject *)RAVE_OBJECT_CLONE(k.second);
            RaveCoreObject* the_object = (RaveCoreObject*)RAVE_OBJECT_COPY(k.second);
            args[i].mcomp->_file_objects[k.first] = the_object;
            break;
          }
        }
      }
    }
  }

  // Info and debug print
  for (size_t idx = 0; idx < args.size(); idx++) {
    RAVE_INFO4(
      "[radarcomp_c] tiled_compositing._add_files_to_argument_list: Tile %s contains  %d files and dimensions %d x %d",
      args[idx].mcomp->area_definition->getId().c_str(),
      args[idx].mcomp->_filenames.size(),
      args[idx].mcomp->area_definition->getXSize(),
      args[idx].mcomp->area_definition->getYSize());
    RAVE_DEBUG1("[%s] tiled_compositing._add_files_to_argument_list: Finished splitting polar object", mpname.c_str());
  }
}

void TiledCompositing::_add_radar_index_value_to_argument_list(std::vector<args_to_tiler> & args)
{
  int ctr = 1;
  for (auto const & k : file_objects) {
    RaveCoreObject* v = k.second;
    // Jump over garbage in map.
    bool is_scan = RAVE_OBJECT_CHECK_TYPE(v, &PolarScan_TYPE);
    bool is_volume = RAVE_OBJECT_CHECK_TYPE(v, &PolarVolume_TYPE);
    if (!is_scan && !is_volume) {
      // Garbage found in map
      continue;
    }
    std::string sourceid, vsource;
    if (is_scan) {
      vsource.assign(PolarScan_getSource((PolarScan_t*)v));
    } else if (is_volume) {
      vsource.assign(PolarVolume_getSource((PolarVolume_t*)v));
    }

    // Set default
    sourceid = vsource;

    if (OdimSource_getIdFromOdimSourceInclusive(vsource.c_str(), "NOD") != NULL) {
      sourceid = OdimSource_getIdFromOdimSourceInclusive(vsource.c_str(), "NOD");
    } else if (OdimSource_getIdFromOdimSourceInclusive(vsource.c_str(), "WMO") != NULL) {
      sourceid = OdimSource_getIdFromOdimSourceInclusive(vsource.c_str(), "WMO");
    } else if (OdimSource_getIdFromOdimSourceInclusive(vsource.c_str(), "RAD") != NULL) {
      sourceid = OdimSource_getIdFromOdimSourceInclusive(vsource.c_str(), "RAD");
    }

    for (auto & arg : args) {
      // Check if sourceid is in radar_indedx_mapping
      if (arg.mcomp->radar_index_mapping == 0) {
        // Create a new one.
        arg.mcomp->radar_index_mapping = (RaveObjectHashTable_t*)RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
        if (arg.mcomp->radar_index_mapping == 0) {
          // Nothing more to do.
          RAVE_CRITICAL0("Failed to allocate memory for radar_index_mapping.");
          break;
        }
        // New one sucessfully created
      }
      if (!RaveObjectHashTable_exists(arg.mcomp->radar_index_mapping, sourceid.c_str())) {
        RaveAttribute_t* attr = RaveAttributeHelp_createLong(sourceid.c_str(), ctr);
        if (attr != NULL) {
          if (!RaveObjectHashTable_put(arg.mcomp->radar_index_mapping, sourceid.c_str(), (RaveCoreObject*)attr)) {
            RAVE_ERROR0("Failed to add attribute to radar index mapping");
          }
        }
        RAVE_OBJECT_RELEASE(attr);
      }
    }
    ctr = ctr + 1;
  }
}

bool TiledCompositing::_ensure_date_and_time_on_args(std::vector<args_to_tiler> & args)
{
  std::string dtstr;
  std::string ddstr;
  for (const auto & k : file_objects) {
    RaveCoreObject* v = k.second;
    // Jump over garbage in map.
    bool is_scan = RAVE_OBJECT_CHECK_TYPE(v, &PolarScan_TYPE);
    bool is_volume = RAVE_OBJECT_CHECK_TYPE(v, &PolarVolume_TYPE);
    if (!is_scan && !is_volume) {
      // Garbage found in map
      continue;
    }
    if (is_scan) {
      dtstr.assign(PolarScan_getTime((PolarScan_t*)v));
      ddstr.assign(PolarScan_getDate((PolarScan_t*)v));
    } else if (is_volume) {
      dtstr.assign(PolarVolume_getTime((PolarVolume_t*)v));
      ddstr.assign(PolarVolume_getDate((PolarVolume_t*)v));
    }
    break;
  }
  if ((dtstr.length() == 0) || (ddstr.length() == 0)) {
    RAVE_DEBUG1("[%s] tiled_compositing._ensure_date_and_time_on_args: Could not determine any date and time string",
                mpname);
    return false;
  }

  for (auto & arg : args) {
    if ((arg.mcomp->_filenames.size() == 0) && ((arg.dd.length() == 0) || (arg.tt.length() == 0))) {
      arg.dd = ddstr;
      arg.tt = dtstr;
    }
  }
  return true;
}
/**
 * @brief Creates the list of arguments to be sent to the tilers. Each item in the returned list is supposed to represent one tile
 * @param dd: date
 * @param dt: time
 * @param aid: the area id (that might or not be tiled)
 * @return a list of argument lists
*/

std::vector<args_to_tiler> TiledCompositing::_create_arguments(std::string dd, std::string dt, Area_t* the_area)
{
  // FIXME: Check _get_tiled_areas!

  RaveObjectList_t* tiled_areas = _get_tiled_areas(the_area);

  std::vector<args_to_tiler> args;

  for (int i = 0; i < RaveObjectList_size(tiled_areas); i++) {
    Area_t* ta = (Area_t*)RaveObjectList_get(tiled_areas, i);
    multi_composite_arguments* mcomp = _create_multi_composite_argument(_create_tiled_area_definition(ta));
    args_to_tiler the_arg;
    the_arg.mcomp = mcomp;
    the_arg.dd = dd;
    the_arg.tt = dt;
    the_arg.areaid.assign(Area_getID(ta));
    args.push_back(the_arg);
  }

  // # Now, make sure we have the correct files in the various areas
  _add_files_to_argument_list(args, tiled_areas);

  RAVE_OBJECT_RELEASE(tiled_areas);

  // # And add the radar index value to be used for each radar source so that each tile
  // # have same information
  _add_radar_index_value_to_argument_list(args);

  // # We also must ensure that if any arg contains 0 files, there must be a date/time set
  if (!_ensure_date_and_time_on_args(args)) {
    RAVE_CRITICAL0("Could not ensure existing date and time for composite");
  }

  return args;
}
/**
 * @brief Same as compositing generate but this is supposed to forward requests to a tiling mechanism
 * @param dd: date
 * @param dt: time
 * @param area: the area id
 */
Cartesian_t* TiledCompositing::generate(std::string dd, std::string dt, std::string areaid, Area_t* area)
{
  // starttime = time.time()
  std::time_t starttime = std::time(0);
  // # Projection and area registries from compositing member object

  Area_t* the_area = 0;

  if (areaid.length()) {
    the_area = AreaRegistry_getByName(compositing->area_registry, areaid.c_str());
  }

  if (the_area == 0) {
    RAVE_CRITICAL1("Failed to get area %s from area registry.", areaid.c_str());
    return 0;
  }

  bool all_files_malfunc = false;
  if (_preprocess_qc && _mp_process_qc && number_of_quality_control_processes > 1) {
    file_objects = _fetch_file_objects_mp(_nodes, _how_tasks, all_files_malfunc);
  } else {
    file_objects = _fetch_file_objects(_nodes, _how_tasks, all_files_malfunc);
  }
  if (all_files_malfunc) {
    RAVE_INFO1("[%s] tiled_compositing.generate: Content of all provided files were marked as 'malfunc'. Since option "
               "'ignore_malfunc' is set, no composite is generated!",
               mpname.c_str());
    RAVE_OBJECT_RELEASE(the_area);
    return 0;
  }
  // To avoid reading of radar volumes once again, we copy the objects already in memory.
  std::vector<args_to_tiler> args = _create_arguments(dd, dt, the_area);

  std::vector<result_from_tiler> results;

  int ntiles = args.size();
  int ncpucores = get_nprocs();

  int nrprocesses = ntiles;
  if (RAVE_TILE_COMPOSITING_PROCESSES > 0) {
    if (nrprocesses > RAVE_TILE_COMPOSITING_PROCESSES) {
      nrprocesses = RAVE_TILE_COMPOSITING_PROCESSES;
    }
  }

  if (nrprocesses > ncpucores) {
    nrprocesses = ncpucores;
  }
  if ((nrprocesses == ncpucores) && (ncpucores > 1)) {
    nrprocesses = nrprocesses - 1;  // # We always want to leave at least one core for something else
  }
  // INVOKE thread pool implementation.
  thread_pool_executor executor(nrprocesses, nrprocesses, std::chrono::seconds(RAVE_TILE_COMPOSITING_TIMEOUT), ntiles);
  std::vector<std::future<result_from_tiler>> futures;
  for (size_t i = 0; i < args.size(); ++i) {
    futures.push_back(executor.submit(comp_generate, args[i]));
  }

  for (auto && future : futures) {
    result_from_tiler result = future.get();
    results.push_back(result);
  }

  executor.shutdown();
  executor.wait();

  bool processing_ok = false;

  // # Clean up results and remove None
  // results = [r for r in results if r is not None]

  if (results.size() == args.size()) {
    processing_ok = true;
  } else {
    // Map to allow effective search
    std::map<std::string, int> processed_areas;
    int index = 0;
    for (auto & v : results) {
      processed_areas[v.tileid] = index;
      index++;
    }
    for (auto & a : args) {
      if (!processed_areas.count(a.areaid)) {
        RAVE_ERROR1("[radarcomp_c] tiled_compositing.generate: No answer from subprocess when generating composite "
                    "tile with areaid: %s",
                    a.areaid.c_str());
        // # Either we want to hide this fact from user and create as much of the composite as possible or else we just
        // want # the product to dissapear since something ugly might have happened during processing.
        if (RAVE_TILE_COMPOSITING_ALLOW_MISSING_TILES) {
          result_from_tiler result;
          result.tileid = a.areaid;
          result.totaltime = RAVE_TILE_COMPOSITING_TIMEOUT;
          results.push_back(result);
        } else {
          RAVE_CRITICAL1("No answer from subprocess when generating composite tile with areaid: %s", a.areaid.c_str());
          return 0;
        }
      }
    }
  }
  // results = [results] //# To get same behavior as map_async

  if (results.size() > 0) {
    for (auto & v : results) {
      RAVE_INFO2("[radarcomp_c] tiled_compositing.generate: Tile with areaid: %s took %d seconds to process",
                 v.tileid.c_str(),
                 v.totaltime);
    }
  }

  RAVE_INFO0("[radarcomp_c] tiled_compositing.generate: Finished processing, combining tiles");

  RaveObjectList_t* objects = (RaveObjectList_t*)RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (objects == NULL) {
    // FIXME: memory handling.
    return 0;
  }
  for (auto & v : results) {
    std::string tile_file = v.filename;
    if (tile_file.empty()) {
      RAVE_WARNING1("[radarcomp_c] tiled_compositing.generate: No partial composite for tile area %s was created. This "
                    "tile will therefore not be included in complete composite.",
                    v.tileid.c_str());
    } else {
      RaveIO_t* instance = RaveIO_open(tile_file.c_str(), false, "DBZH");
      RaveCoreObject* o = RaveIO_getObject(instance);
      Cartesian_t* p = 0;
      if (RAVE_OBJECT_CHECK_TYPE(o, &CartesianVolume_TYPE)) {
        int i = CartesianVolume_getNumberOfImages((CartesianVolume_t*)o);
        if (i > 0) {
          p = CartesianVolume_getImage((CartesianVolume_t*)o, 0);
        }
        Cartesian_setObjectType(p, Rave_ObjectType::Rave_ObjectType_COMP);
        RaveObjectList_add(objects, (RaveCoreObject*)p);
      }
      RAVE_OBJECT_RELEASE(o);
      RaveIO_close(instance);
    }
  }
  Transform_t* t = (Transform_t*)RAVE_OBJECT_NEW(&Transform_TYPE);
  if (t == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory for transform.");
    // FIXME:Memory handling.
    return 0;
  }

  RAVE_DEBUG2("[radarcomp_c] tiled_compositing.generate: Combining %d tiles into one composite for area %s.",
              RaveObjectList_size(objects),
              areaid.c_str());

  Cartesian_t* result = Transform_combine_tiles(t, the_area, objects);
  RAVE_OBJECT_RELEASE(t);

  // # Fix so that we get a valid place for /what/source and /how/nodes
  std::string source_string("ORG:82,CMT:");
  source_string += areaid;
  Cartesian_setSource(result, source_string.c_str());
  // result.source = "%s,CMT:%s"%(ORG:82,areaid.c_str)
  RaveAttribute_t* nodes_attr = RaveAttributeHelp_createString("how/nodes", _nodes.c_str());
  if (nodes_attr != NULL) {
    if (!Cartesian_addAttribute(result, nodes_attr)) {
      RAVE_ERROR0("Failed to add attribute how/nodes to composite");
    }
  }
  RAVE_OBJECT_RELEASE(nodes_attr);
  if (_how_tasks.length()) {
    RaveAttribute_t* tasks_attr = RaveAttributeHelp_createString("how/tasks", _how_tasks.c_str());
    if (tasks_attr != NULL) {
      if (!Cartesian_addAttribute(result, tasks_attr)) {
        RAVE_ERROR0("Failed to add attribute how/tasks to composite");
      }
    }
    RAVE_OBJECT_RELEASE(tasks_attr);
  }

  std::time_t totaltime = (int)(std::time(0) - starttime);

  RAVE_INFO1("[radarcomp_c] tiled_compositing.generate: Tiled compositing took %d s to execute.", totaltime);

  // Clenn up temporary files.
  if (results.size() > 0) {
    for (auto & v : results) {
      if (v.filename.length()) {
        unlink(v.filename.c_str());
      }
    }
  }
  RAVE_OBJECT_RELEASE(objects);
  RAVE_OBJECT_RELEASE(the_area);
  for (auto & arg : args) {
    // The destructor takes care of releasing rave objects
    delete arg.mcomp;
    arg.mcomp = 0;
  }

  for (auto & k : file_objects) {
    RaveCoreObject* v = k.second;
    if (v != 0) {
      RAVE_OBJECT_RELEASE(v);
    }
  }

  return result;
}


std::map<std::string, RaveCoreObject*> TiledCompositing::quality_control_objects(std::map<std::string, RaveCoreObject*> & objects, CompositeAlgorithm_t* algorithm, std::string & qfields)
{
  algorithm = 0;
  std::map<std::string, RaveCoreObject*> result;
  qfields = "";
  for (auto const & k : objects) {
    RaveCoreObject* obj = k.second;
    for (std::string d : detectors) {
      /* clang-format off */
      /* FIXME: Implement quality plugins in c++!
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
        */
      /* clang-format on */
    }
    result[k.first] = obj;
  }
  return result;
}

bool TiledCompositing::_get_malfunc_from_obj(RaveCoreObject* obj, bool is_polar)
{
  if (is_polar) {
    if (PolarVolume_hasAttribute((PolarVolume_t*)obj, "how/malfunc")) {
      return (bool)PolarVolume_hasAttribute((PolarVolume_t*)obj, "how/malfunc");
    } else {
      return false;
    }
  } else {
    if (PolarScan_hasAttribute((PolarScan_t*)obj, "how/malfunc")) {
      return (bool)PolarScan_hasAttribute((PolarScan_t*)obj, "how/malfunc");
    } else {
      return false;
    }
  }
}

RaveCoreObject* TiledCompositing::_remove_malfunc_from_volume(RaveCoreObject* obj, bool is_polar)
{
  // FIXME: memory handling correct?
  RaveCoreObject* result = obj;
  if (is_polar) {
    if (_get_malfunc_from_obj(obj, is_polar)) {
      RAVE_DEBUG3("Malfunc volume found. Source: %s, Nominal date and time: %sT%s",
                  PolarVolume_getSource((PolarVolume_t*)obj),
                  PolarVolume_getDate((PolarVolume_t*)obj),
                  PolarVolume_getTime((PolarVolume_t*)obj));
      return 0;
    }
    int i = PolarVolume_getNumberOfScans((PolarVolume_t*)obj);
    for (; i > 0; i--) {
      RaveCoreObject* scan = (RaveCoreObject*)PolarVolume_getScan((PolarVolume_t*)obj, i);
      if (_get_malfunc_from_obj(scan, is_polar)) {
        RAVE_DEBUG4("Malfunc scan with elangle %f found. Removing from volume. Source: %s, Nominal date and time: %sT%s",
          (PolarScan_getElangle((PolarScan_t*)scan) * 180.0 / M_PI),
          PolarScan_getSource((PolarScan_t*)scan),
          PolarScan_getDate((PolarScan_t*)scan),
          PolarScan_getTime((PolarScan_t*)scan));
        PolarVolume_removeScan((PolarVolume_t*)obj, i);
      }
      i = PolarVolume_getNumberOfScans((PolarVolume_t*)obj) - 1;
    }
  }
  return result;
}

RaveCoreObject* TiledCompositing::_remove_malfunc(RaveCoreObject* obj, bool is_polar)
{
  RaveCoreObject* result = obj;
  if (is_polar) {
    result = _remove_malfunc_from_volume(obj, is_polar);
    if ((result != 0) && (PolarVolume_getNumberOfScans((PolarVolume_t*)result) == 0)) {
      RAVE_DEBUG0("All scans of the volume were detected as malfunc. Complete volume therefore considered as malfunc.");
      result = 0;
    }
  } else {
    if (_get_malfunc_from_obj(obj, is_polar)) {
      result = 0;
    }
  }
  return result;
}


void TiledCompositing::_debug_generate_info(std::string area)
{
  if (verbose) {
    RAVE_DEBUG1("Generating cartesian image from %d files", filenames.size());
    // loop over detectors.
    // RAVE_DEBUG1("Detectors = '%s'",detectors);
    RAVE_DEBUG1("Quality control mode = '%s'", quality_control_mode.c_str());
    RAVE_DEBUG1("Product = '%s'", _product_repr().c_str());
    RAVE_DEBUG1("Quantity = '%s'", quantity.c_str());
    RAVE_DEBUG1("Range = %f", range);
    RAVE_DEBUG3("Gain = %f, Offset = %f, Minvalue = %f", gain, offset, minvalue);
    RAVE_DEBUG1("Prodpar = '%s'", prodpar.c_str());
    RAVE_DEBUG1("Selection method = '%s'", _selection_method_repr().c_str());
    RAVE_DEBUG1("Interpolation method = '%s'", _interpolation_method_repr().c_str());
    RAVE_DEBUG1("Gap filling = %d", applygapfilling);
    RAVE_DEBUG1("Ct filtering = %d", applyctfilter);
    RAVE_DEBUG1("Gra filtering = %d", applygra);
    RAVE_DEBUG1("Ignoring malfunc = %d", ignore_malfunc);
    RAVE_DEBUG1("QI-total field = '%s'", qitotal_field.c_str());
    RAVE_DEBUG1("Reprocess quality fields = %d", reprocess_quality_field);
    RAVE_DEBUG1("Dumping path = '%s'", dumppath.c_str());
    RAVE_DEBUG1("Dumping output = %d", dump);
    RAVE_DEBUG1("Use site source = %d", use_site_source);
    RAVE_DEBUG1("Use lazy loading = %d", use_lazy_loading);
    RAVE_DEBUG1("Use lazy loading preload = %d", use_lazy_loading_preloads);

    if (area.length()) {
      RAVE_DEBUG1("Area = '%s'", area.c_str());
    } else {
      RAVE_DEBUG0("Area = 'best fit'");
      RAVE_DEBUG1("pcsid = '%s'", pcsid.c_str());
      RAVE_DEBUG2("xscale = %f, yscale = %f", xscale, yscale);
    }
  }
}

/**
 * Dumps the objects on the ingoing po*lar objects onto the file system.
 * The names will contain a unique identifierto allow for duplicate versions of the same object.
 * @param objects the objects to write to disk
 * */
void TiledCompositing::_dump_objects(std::vector<RaveCoreObject*> & vobjects)
{
  // Implement later
}
/**
 * Apply gra coefficient adjustment.
 * @param result: The cartesian product to be adjusted
 * @param d: the date string representing now (YYYYmmdd)
 * @param t: the time string representing now (HHMMSS)
 * @return the gra field with the applied corrections
 * */
CartesianParam_t* TiledCompositing::_apply_gra(Cartesian_t* result, std::string d, std::string t)
{
  return 0;
}
/**
 * @return the string representation o*f the selection method
 */
std::string TiledCompositing::_selection_method_repr()
{
  if (selection_method == CompositeSelectionMethod_t::CompositeSelectionMethod_NEAREST) {
    return "NEAREST_RADAR";
  } else if (selection_method == CompositeSelectionMethod_t::CompositeSelectionMethod_HEIGHT) {
    return "HEIGHT_ABOVE_SEALEVEL";
  }
  return "Unknown";
}
/**
 * @return the string representation o*f the interpolation method
 * */
std::string TiledCompositing::_interpolation_method_repr()
{
  if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_NEAREST) {
    return "NEAREST_VALUE";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_HEIGHT) {
    return "LINEAR_HEIGHT";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_RANGE) {
    return "LINEAR_RANGE";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_AZIMUTH) {
    return "LINEAR_AZIMUTH";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_RANGE_AND_AZIMUTH) {
    return "LINEAR_RANGE_AND_AZIMUTH";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_3D) {
    return "LINEAR_3D";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_QUADRATIC_HEIGHT) {
    return "QUADRATIC_HEIGHT";
  } else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_QUADRATIC_3D) {
    return "QUADRATIC_3D";
  }
  return "Unknown";
}
/**
 * @return the string representation o*f the product type
 */
std::string TiledCompositing::_product_repr()
{
  if (product == Rave_ProductType::Rave_ProductType_PPI) {
    return "ppi";
  } else if (product == Rave_ProductType::Rave_ProductType_CAPPI) {
    return "cappi";
  } else if (product == Rave_ProductType::Rave_ProductType_PCAPPI) {
    return "pcappi";
  } else if (product == Rave_ProductType::Rave_ProductType_PMAX) {
    return "pmax";
  } else if (product == Rave_ProductType::Rave_ProductType_MAX) {
    return "max";
  } else {
    return "unknown";
  }
}

void TiledCompositing::_update_generator_with_prodpar(Composite_t* generator)
{
  if (prodpar.length() != 0) {
    if ((product == Rave_ProductType::Rave_ProductType_CAPPI) ||
        (product == Rave_ProductType::Rave_ProductType_PCAPPI)) {
      Composite_setHeight(generator, _strToNumber(prodpar));
    } else if (product == Rave_ProductType::Rave_ProductType_PMAX) {
      std::vector<std::string> pp;
      std::istringstream f(prodpar);
      std::string s;
      while (getline(f, s, ',')) {
        pp.push_back(s);
      }
      // FIXME: Do we need to strip withspaces?
      if (pp.size() == 2) {
        Composite_setHeight(generator, _strToNumber(pp[0]));
        Composite_setRange(generator, _strToNumber(pp[1]));
      } else if (pp.size() == 1) {
        Composite_setHeight(generator, _strToNumber(pp[0]));
      }
    } else if (product == Rave_ProductType::Rave_ProductType_PPI) {
      float v = _strToNumber(prodpar);
      Composite_setElevationAngle(generator, v * M_PI / 180.0);
    }
  }
}

float TiledCompositing::_strToNumber(std::string sval)
{
  return std::stof(sval);
}

/* clang-format off */
// The thread func for quality control
  
  /**
   * Handles the multiprocessing call for the quality control section
   * @param args: tuple of 4 args, ([filenames],[detectors], reprocess_quality_field, ignore_malfunc)
   * @return a tuple of ([filenames], <execution status as boolean>, "filenames or source names")
   */
  
  // NOTE: All objects must be of local scope!
  /*def execute_quality_control(args):
  filenames,detectors,reprocess_quality_field,ignore_malfunc = args
  result = ([], False, "%s"%str(filenames))
  try:
  comp = compositing.compositing()
  comp.filenames.extend(filenames)
  comp.detectors.extend(detectors)
  comp.reprocess_quality_field = reprocess_quality_field
  comp.ignore_malfunc = ignore_malfunc
  
  logger.info(f"[{mpname}] tiled_compositing.execute_quality_control: Starting QC of {len(comp.filenames)} objects.")
  
  objects, nodes, how_tasks, all_files_malfunc = comp.fetch_objects()
  
  objects, algorithm = comp.quality_control_objects(objects)
  
  status = True
  try:
  objects = store_temporary_files(objects)
  except Exception:
  status = False
  result = (objects.keys(), status, nodes)
  except Exception:
  logger.exception(f"[{mpname}] tiled_compositing.execute_quality_control: Failed to run quality control")
  return result
*/
  
/* clang-format on */

