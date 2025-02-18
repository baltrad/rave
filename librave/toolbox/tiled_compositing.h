#ifndef _tiled_compositing_h
#define _tiled_compositing_h
#include "compositing.h"
extern "C" {
#include "rave_types.h"
#include "rave_io.h"
#include "rave_object.h"
#include "composite.h"
#include "polarscan.h"
}

#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <ctime>

/*
 * From rave_defines.py, RAVEROOT hardcoded for now
 */
extern const char* RAVEROOT;
extern const char* RAVECONFIG;
extern const char* RAVEETC;

extern const char* PROJECTION_REGISTRY;
extern const char* AREA_REGISTRY;
extern const char* RAVE_TILE_REGISTRY;


typedef struct _result_from_tiler {
  std::string tileid;
  std::string filename;
  std::time_t totaltime;
} result_from_tiler;

/**
 * The basic area def*inition that should be transfered to the tiling compositing instance.
 * This definition will be pickled and sent to the receiving product generator.
 */
class tiled_area_definition {
public:
  tiled_area_definition();
  ~tiled_area_definition();
  // clang-format off
  void init(const char * id, const char * pcsdef, double xscale, double yscale, int xsize, int ysize, const char * extent);
  const char * repr();
  std::string getId() {return _id;};
  std::string getPcsdef() {return _pcsdef;};
  double getXScale() {return _xscale;};
  double getYScale() {return _yscale;};
  int getXSize() {return _xsize;};
  int getYSize() {return _ysize;};
  std::string getExtent() {return _extent;};
  // clang-format on

private:
  std::string _id;
  std::string _pcsdef;
  double _xscale = 0.0;
  double _yscale = 0.0;
  int _xsize = 0;
  int _ysize = 0;
  std::string _extent;
};

/**
 * The argument wrapper so that the arguments can be transfered to the composite generator taking care of the tile.
 */
class multi_composite_arguments {
public:
  // The parameters to the tiler.
  multi_composite_arguments();
  ~multi_composite_arguments();
  void set_area_definition(tiled_area_definition* areadef);
  result_from_tiler generate(std::string dd, std::string dt, std::string tid);

private:
  void init();

public:
  float xscale;
  float yscale;
  std::vector<std::string> detectors;
  std::vector<std::string> _filenames;
  bool ignore_malfunc;
  std::string prodpar;
  Rave_ProductType product;
  float height;
  float elangle;
  float range;
  CompositeSelectionMethod_t selection_method;
  CompositeInterpolationMethod_t interpolation_method;
  std::string qitotal_field;
  bool applygra;
  float zr_A;
  float zr_b;
  bool applygapfilling;
  bool applyctfilter;
  std::string quantity;
  float gain;
  float offset;
  float minvalue;
  bool reprocess_quality_field;
  tiled_area_definition* area_definition;
  bool verbose;
  std::string dumppath;
  bool dump;
  RaveObjectHashTable_t* radar_index_mapping;
  static std::mutex mutex;
  std::map<std::string, RaveCoreObject*> _file_objects;
  bool use_lazy_loading;
  bool use_lazy_loading_preloads;
};

typedef struct _args_to_tiler {
  multi_composite_arguments* mcomp;
  std::string dd;
  std::string tt;
  std::string areaid;
} args_to_tiler;

typedef struct _args_to_qc {
  std::vector<std::string> filenames;
  std::vector<std::string> detectors;
  bool reprocess_quality_field;
  bool ignore_malfunc;
} args_to_qc;

/**
 * @brief TiledCompositing
 * TiledCompositing class instance
 * FIXME: logger class
 */
class TiledCompositing : public Compositing {
public:
  TiledCompositing();
  ~TiledCompositing();
  void init(Compositing* c, bool & preprocess_qc, bool & mp_process_qc, bool & mp_process_qc_split_evenly);
  /**
   * @brief Generates the cartesian image.
   * @param dd: date in format YYYYmmdd
   * @param dt: time in format HHMMSS
   * @param areaid: the id to look up in area registry.
   * @param area: the area to use for the cartesian image. If 0 is specified, areaid will be used.
   */
  Cartesian_t* generate(std::string dd, std::string dt, std::string areaid, Area_t* area = 0);
  /**
   * @brief Removes CMT:<...> from the string
   * @param[in] str - the string from which CMT should be removed
   * @return the source with CMT removed
   */
  // std::string remove_CMT_from_source(std::string str);
  // void set_product_from_string(std::string prodstr);
  // void set_method_from_string(std::string methstr);
  // void set_interpolation_method_from_string(std::string methstr);
  // void set_quality_control_mode_from_string(std::string modestr);
  std::map<std::string, RaveCoreObject*> quality_control_objects(std::map<std::string, RaveCoreObject*> & objects,
                                                                 CompositeAlgorithm_t* algorithm,
                                                                 std::string & qfields);
  /**
   * Generates the objects that should be used in the compositing.
   * returns a triplet with [objects], nodes (as comma separated string), 'how/tasks' (as comma separated string)
   **/
  // clang-format off
  // std::map<std::string,RaveCoreObject*> fetch_objects(std::string & nodes, std::string & how_tasks, bool &all_files_malfunc); void add_how_task_from_scan(PolarScan_t * scan, std::string &tasks); void create_filename(void* pobj); void get_backup_gra_coefficient(void * db, std::string agedt, std::string nowdt);
  // clang-format on
  // void test_func(std::string a);

private:
  RaveObjectList_t* _get_tiled_areas(Area_t* area);
  tiled_area_definition* _create_tiled_area_definition(Area_t* area);
  multi_composite_arguments* _create_multi_composite_argument(tiled_area_definition* adef);
  void _add_files_to_argument_list(std::vector<args_to_tiler> & args, RaveObjectList_t* tiled_areas);
  void _add_radar_index_value_to_argument_list(std::vector<args_to_tiler> & args);
  bool _ensure_date_and_time_on_args(std::vector<args_to_tiler> & args);
  std::vector<args_to_tiler> _create_arguments(std::string dd, std::string dt, Area_t* the_area);
  bool _get_malfunc_from_obj(RaveCoreObject* obj, bool is_polar);
  RaveCoreObject* _remove_malfunc_from_volume(RaveCoreObject* obj, bool is_polar);
  RaveCoreObject* _remove_malfunc(RaveCoreObject* obj, bool is_polar);

  void _debug_generate_info(std::string area);

  std::map<std::string, RaveCoreObject*> _fetch_file_objects(std::string & nodes, std::string & how_tasks, bool & all_files_malfunc);
  std::map<std::string, RaveCoreObject*> _fetch_file_objects_mp(std::string & nodes, std::string & how_tasks, bool & all_files_malfunc);
  /*#
  # Dumps the objects on the ingoing polar objects onto the file system. The names will contain a unique identifier
  # to allow for duplicate versions of the same object.
  # @param objects the objects to write to disk */
  void _dump_objects(std::vector<RaveCoreObject*> & vobjects);
  /*#
  # Apply gra coefficient adjustment.
  # @param result: The cartesian product to be adjusted
  # @param d: the date string representing now (YYYYmmdd)
  # @param t: the time string representing now (HHMMSS)
  # @return the gra field with the applied corrections */
  CartesianParam_t* _apply_gra(Cartesian_t* result, std::string d, std::string t);
  /*#
  # @return the string representation of the selection method
  */
  std::string _selection_method_repr();
  /*#
  # @return the string representation of the interpolation method */
  std::string _interpolation_method_repr();
  /*#
  # @return the string representation of the product type   */
  std::string _product_repr();
  void _update_generator_with_prodpar(Composite_t* generator);
  float _strToNumber(std::string sval);

public:
  Compositing* compositing;
  bool _preprocess_qc;
  bool _mp_process_qc;
  bool _mp_process_qc_split_evenly;
  int number_of_quality_control_processes;
  bool _do_remove_temporary_files;
  std::map<std::string, RaveCoreObject*> file_objects;
  std::string _nodes;
  std::string _how_tasks;
  std::string mpname;
  void* ravebdb;
  std::string pcsid;
  float xscale;
  float yscale;
  std::vector<std::string> detectors;
  std::vector<std::string> filenames;
  bool ignore_malfunc;
  std::string prodpar;
  Rave_ProductType product;
  float height;
  float elangle;
  float range;
  CompositeSelectionMethod_t selection_method;
  CompositeInterpolationMethod_t interpolation_method;
  std::string quality_control_mode;
  std::string qitotal_field;
  bool applygra;
  float zr_A;
  float zr_b;
  bool applygapfilling;
  bool applyctfilter;
  std::string quantity;
  float gain;
  float offset;
  float minvalue;
  bool reprocess_quality_field;
  bool verbose;
  void* logger;
  std::string dumppath;
  bool dump;
  bool use_site_source;
  bool use_azimuthal_nav_information;
  RaveObjectHashTable_t* radar_index_mapping;
  // std::map <std::string,int> radar_index_mapping;
  bool use_lazy_loading;
  bool use_lazy_loading_preloads;
};
#endif
