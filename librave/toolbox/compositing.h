#ifndef _compositing_h
#define _compositing_h

extern "C" {
  #include "rave_types.h"
  #include "rave_io.h"
  #include "rave_object.h"
  #include "composite.h"
  #include "polarscan.h"
  #include "arearegistry.h"
  #include "tileregistry.h"
}

#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <mutex>

/*
 * From rave_defines.py, RAVEROOT hardcoded for now
 */
extern const char * RAVEROOT;
extern const char * RAVECONFIG;
extern const char * RAVEETC;

extern const char * PROJECTION_REGISTRY;
extern const char * AREA_REGISTRY;
extern const char * RAVE_TILE_REGISTRY;

/**
 \ brie*f Compositing
 
 # Compositing class instance
 */

// FIXME: logger class
class Compositing  {
public:
  Compositing();
  ~Compositing();
  void init();
  Cartesian_t* generate( std::string dd, std::string dt, std::string areaid, Area_t * area = 0);
  /*# Removes CMT:<...> from the string
  # @param[in] str - the string from which CMT should be removed
  # @return the source with CMT removed
  #*/
  std::string remove_CMT_from_source(std::string str);
  void set_product_from_string(std::string prodstr);
  void set_method_from_string(std::string methstr);
  void set_interpolation_method_from_string(std::string methstr);
  void set_quality_control_mode_from_string(std::string modestr);
  std::map<std::string,RaveCoreObject*> quality_control_objects(std::map<std::string,RaveCoreObject*> &objects, CompositeAlgorithm_t* algorithm, std::string&qfields);
  /*#
  # Generates the objects that should be used in the compositing.
  # returns a triplet with [objects], nodes (as comma separated string), 'how/tasks' (as comma separated string)
  #*/
  std::map<std::string,RaveCoreObject*> fetch_objects(std::string & nodes, std::string & how_tasks, bool &all_files_malfunc);
  void add_how_task_from_scan(PolarScan_t * scan, std::vector<std::string> &tasks);
  void create_filename(void * pobj);
  void get_backup_gra_coefficient(void * db, std::string agedt, std::string nowdt);
  void test_func(std::string a);
  
private:
  bool _get_malfunc_from_obj(RaveCoreObject*obj, bool is_polar);
  RaveCoreObject * _remove_malfunc_from_volume(RaveCoreObject *obj, bool is_polar);
  RaveCoreObject * _remove_malfunc(RaveCoreObject * obj, bool is_polar);
  
  void _debug_generate_info(std::string area);
  /*# Generates the cartesian image.
  #
  # @param dd: date in format YYYYmmdd
  # @param dt: time in format HHMMSS
  # @param area: the area to use for the cartesian image. If none is specified, a best fit will be atempted.
  */
  Cartesian_t* _generate(std::string dd, std::string dt, std::string areaid="", Area_t * area = 0);
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
  CartesianParam_t  * _apply_gra(Cartesian_t* result, std::string d, std::string t);
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
  std::string mpname;
  void * ravebdb;
  std::string pcsid;
  float xscale;
  float yscale;
  std::vector <std::string> detectors;
  std::vector <std::string> filenames;
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
  void * logger;
  std::string dumppath;
  bool dump;
  bool use_site_source;
  bool use_azimuthal_nav_information;
  RaveObjectHashTable_t* radar_index_mapping;
  //std::map <std::string,int> radar_index_mapping;
  bool use_lazy_loading;
  bool use_lazy_loading_preloads;
  
  ProjectionRegistry_t* proj_registry;
  AreaRegistry_t* area_registry;
  TileRegistry_t* tile_registry;

  std::map<std::string,RaveCoreObject*> file_objects;

  static std::mutex mutex;
  
};
#endif
