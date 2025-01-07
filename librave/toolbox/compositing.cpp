#include "compositing.h"
#include "odim_source.h"

extern "C" {
#include "rave_attribute.h"
#include "composite.h"
#include "rave_debug.h"
#include "rave_io.h"
#include "rave_object.h"
#include "rave_types.h"
#include "rave_datetime.h"
#include "rave_attribute.h"
#include "polarscan.h"
#include "polarvolume.h"
#include "arearegistry.h"
#include "area.h"
#include "projectionregistry.h"
}

#include <math.h>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <map>


  Compositing::Compositing(){
    init();
  }
  
  Compositing::~Compositing() {
  }
  
  void Compositing::init(){
    //self.mpname = multiprocessing.current_process().name
    mpname="";
    ravebdb = 0;

      pcsid = "gmaps";
      xscale = 2000.0;
      yscale = 2000.0;
      // No need to init std::vector.
      //detectors = []
      //filenames = []
      ignore_malfunc = false;
      //prodpar = "1000.0";
      product = Rave_ProductType::Rave_ProductType_PCAPPI;
      height = 1000.0;
      elangle = 0.0;
      range = 200000.0;
      selection_method = CompositeSelectionMethod_t::CompositeSelectionMethod_NEAREST;
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_NEAREST;
      // Defined in rave_quality_plugin.py.
      quality_control_mode = "analyze_and_apply";
      //qitotal_field
      applygra = false;
      zr_A = 200.0;
      zr_b = 1.6;    
      applygapfilling = false;
      applyctfilter = false;
      quantity = "DBZH";
      // Defined in rave_defines.py
      // Default gain and offset values for linear transformation between raw and dBZ
      //GAIN = 0.4
      //OFFSET = -30.0
      gain = 0.4;
      offset = -30.0;
      minvalue = -30.0;
      reprocess_quality_field=false;
      verbose = false;
      logger = 0;
      dumppath = "";
      dump = false;
      use_site_source = false;
      use_azimuthal_nav_information = true;
      //radar_index_mapping = {}
      use_lazy_loading=true;
      use_lazy_loading_preloads=true;
  }
  
  Cartesian_t* Compositing::generate( std::string dd, std::string dt, std::string area){
    return _generate(dd, dt, area);
  }
  /*# Removes CMT:<...> from the string
   # @param[in] str - the string from wh*ich CMT should be removed
   # @return the source with CMT removed
   #*/
  std::string Compositing::remove_CMT_from_source(std::string str) {
    return str;
  }
  
  // sorting algoritm for integers
  int compare_ints(const void* a, const void* b)
  {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
    
    // return (arg1 > arg2) - (arg1 < arg2); // possible shortcut
    
    // return arg1 - arg2; // erroneous shortcut: undefined behavior in case of
    // integer overflow, such as with INT_MIN here
  }
  void Compositing::set_product_from_string(std::string prodstr){
    //prodstr = prodstr.lower()
    std::transform(prodstr.begin(), prodstr.end(), prodstr.begin(),[](unsigned char c){ return std::tolower(c); });
    
    if (prodstr == "ppi") {
      product = Rave_ProductType::Rave_ProductType_PPI;
    } else if (prodstr == "cappi") {
      product = Rave_ProductType::Rave_ProductType_CAPPI;
    } else if (prodstr == "pcappi") {
      product = Rave_ProductType::Rave_ProductType_PCAPPI;
    } else if (prodstr == "pmax") {
      product = Rave_ProductType::Rave_ProductType_PMAX;
    } else if (prodstr == "max") {
      product = Rave_ProductType::Rave_ProductType_MAX;
    } else {
        RAVE_WARNING0("Only supported product types are ppi, cappi, pcappi, pmax and max, default pcappi will be used!");
        product = Rave_ProductType::Rave_ProductType_PCAPPI;
    }
  }
  void Compositing::set_method_from_string(std::string methstr){
    
    std::transform(methstr.begin(), methstr.end(), methstr.begin(),[](unsigned char c){ return std::toupper(c); });
    
    if (methstr == "NEAREST_RADAR") {
      selection_method = CompositeSelectionMethod_t::CompositeSelectionMethod_NEAREST;
    } else if (methstr == "HEIGHT_ABOVE_SEALEVEL") {
      selection_method = CompositeSelectionMethod_t::CompositeSelectionMethod_HEIGHT;
    } else {
        RAVE_WARNING0("Only supported selection methods are NEAREST_RADAR or HEIGHT_ABOVE_SEALEVEL, default NEAREST_RADAR will be used!");
        selection_method = CompositeSelectionMethod_t::CompositeSelectionMethod_NEAREST;
    }
  }
  void Compositing::set_interpolation_method_from_string(std::string methstr){
    
    std::transform(methstr.begin(), methstr.end(), methstr.begin(),[](unsigned char c){ return std::toupper(c); });
    
    if (methstr == "NEAREST_VALUE") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_NEAREST;
    } else if (methstr == "LINEAR_HEIGHT") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_HEIGHT;
    } else if (methstr == "LINEAR_RANGE") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_RANGE;
    } else if (methstr == "LINEAR_AZIMUTH") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_AZIMUTH;
    } else if (methstr == "LINEAR_RANGE_AND_AZIMUTH") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_RANGE_AND_AZIMUTH;
    } else if (methstr == "LINEAR_3D") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_3D;
    } else if (methstr == "QUADRATIC_HEIGHT") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_QUADRATIC_HEIGHT;
    } else if (methstr == "QUADRATIC_3D") {
      interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_QUADRATIC_3D;
    } else {
        RAVE_WARNING0("Only supported interpolation methods are NEAREST_VALUE, LINEAR_HEIGHT, LINEAR_RANGE, LINEAR_AZIMUTH, LINEAR_RANGE_AND_AZIMUTH, LINEAR_3D, QUADRATIC_HEIGHT or QUADRATIC_3D, default NEAREST_VALUE will be used!");
        interpolation_method = CompositeInterpolationMethod_t::CompositeInterpolationMethod_NEAREST;
    }
  }
  void Compositing::set_quality_control_mode_from_string(std::string modestr){
    
    std::transform(modestr.begin(), modestr.end(), modestr.begin(),[](unsigned char c){ return std::tolower(c); });
    
    if ((modestr == "analyze" ) || (modestr == "analyze_and_apply")) {
      quality_control_mode = modestr;
    } else {
      RAVE_WARNING1("Invalid quality control mode (%s), only supported modes are analyze_and_apply or analyze, default analyze_and_apply will be used!",modestr.c_str());
      quality_control_mode = "analyze_and_apply";
    }
  };
  
  std::map<std::string,RaveCoreObject*> Compositing::quality_control_objects(
    std::map<std::string,RaveCoreObject*> &objects,
    CompositeAlgorithm_t* algorithm,
    std::string&qfields) {
    
    algorithm = 0;
    std::map<std::string,RaveCoreObject*> result;
    qfields = "";
    for (const auto& k : objects) {
      RaveCoreObject* obj = k.second;
      for (std::string d : detectors) {
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
      }   
      result[k.first] = obj;
    }                    
    return result;
  }
  /*#
   # Generates the objects that should b*e used in the compositing.
   # returns a triplet with [objects], nodes (as comma separated string), 'how/tasks' (as comma separated string)
   #*/
  std::map<std::string,RaveCoreObject*> Compositing::fetch_objects(std::string & nodes, std::string & how_tasks, bool &all_files_malfunc){
    nodes = "";
    std::map<std::string,RaveCoreObject*> objects;
    std::string tasks;
    int malfunc_files = 0;
    
    std::string preload_quantity;
    if (use_lazy_loading && use_lazy_loading_preloads) {
      preload_quantity=quantity;
    }
    std::string fname;  
    for (std::string fname : filenames) {
        RaveCoreObject*obj = 0;
        try {
          if (ravebdb != 0) {
            // ravebdb always 0 in first version.
            //obj = self.ravebdb.get_rave_object(fname, self.use_lazy_loading, preload_quantity)
          }
          else {
            RaveIO_t* instance = RaveIO_open(fname.c_str(), use_lazy_loading, preload_quantity.c_str());
            obj = RaveIO_getObject(instance);
          }
        }
        catch (...) {
          RAVE_ERROR1("Failed to open %s", fname.c_str());
          //FIXME: Memory handling!
          objects.clear();
          return objects;

        }
        bool is_scan = RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE);
        bool is_pvol;
        if (is_scan) {
          is_pvol = false;
        } else {
          is_pvol = RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE);
        }
                
        if (!is_scan && !is_pvol) {
          RAVE_WARNING2("[%s] compositing.fetch_objects: Input file %s is neither polar scan or volume, ignoring.",mpname.c_str(),fname.c_str());
          continue;
        }
                  
        // Force azimuthal nav information usage if requested
        if (is_pvol)
          PolarVolume_setUseAzimuthalNavInformation((PolarVolume_t*) obj, use_azimuthal_nav_information);
        else if(is_scan)
          PolarScan_setUseAzimuthalNavInformation((PolarScan_t*) obj, use_azimuthal_nav_information);
                  
        if (ignore_malfunc) {
            obj = _remove_malfunc(obj,is_pvol);
            if (obj == 0) {
              RAVE_INFO2("[%s] compositing.fetch_objects: Input file %s detected as 'malfunc', ignoring.",mpname.c_str(), fname.c_str());
              malfunc_files += 1;
              continue;
            }
        }
        std::string source;
        if (is_pvol)
          source = PolarVolume_getSource((PolarVolume_t*)obj);
        else
          source = PolarScan_getSource((PolarScan_t*)obj);
        ODIM_Source odim_source(source);
        std::string node = "n/a";
        if (odim_source.nod.length()) { 
          node= odim_source.nod;
        } else if (odim_source.wmo.length()) {
          node=odim_source.wmo;
        } else if (odim_source.wigos.length()) {
          node=odim_source.wigos;
        }  
                      
        if (nodes.length())
          nodes += "," + node;
        else
          nodes += node;
                          
        objects[fname] = obj;
                          
        if (is_scan) {
          RAVE_DEBUG4("[radarcomp_c] compositing.fetch_objects: Scan used in composite generation - UUID: %s, Node: %s, Nominal date and time: %sT%s",
            fname.c_str(), node.c_str(),
            PolarScan_getDate((PolarScan_t*)obj),
            PolarScan_getTime((PolarScan_t*)obj));
          add_how_task_from_scan((PolarScan_t*)obj, tasks);
        } else if (is_pvol) {
          RAVE_DEBUG4("[radarcomp_c] compositing.fetch_objects: PVOL used in composite generation - UUID: %s, Node: %s, Nominal date and time: %sT%s",
            fname.c_str(), node.c_str(),
            PolarVolume_getDate((PolarVolume_t*)obj),
            PolarVolume_getTime((PolarVolume_t*)obj));
          for (int i = 0; i < PolarVolume_getNumberOfScans((PolarVolume_t*) obj)-1; i++) {
            PolarScan_t * scan = PolarVolume_getScan((PolarVolume_t*) obj, i);
            add_how_task_from_scan(scan, tasks);
          }
        }
    }
                              
    how_tasks += "," + tasks;
                              
    all_files_malfunc = ((filenames.size() > 0) && (malfunc_files == filenames.size()));
                              
    return objects;
  }
  void Compositing::add_how_task_from_scan(PolarScan_t * scan, std::string &tasks) {
    if (PolarScan_hasAttribute(scan,"how/task")) {
      RaveAttribute_t *attr = PolarScan_getAttribute(scan, "how/task");
      if (attr != 0) {
        std::string how_task_string = attr->sdata;
        // duplicate check
        if (!tasks.find(how_task_string))
          tasks.append(how_task_string);
      }
    }
  }
  void Compositing::create_filename(void * pobj) {
  }
  void Compositing::get_backup_gra_coefficient(void * db, std::string agedt, std::string nowdt){
  }
  void Compositing::test_func(std::string a) {
  }
  
  bool Compositing::_get_malfunc_from_obj(RaveCoreObject*obj, bool is_polar) {
  
    if (is_polar) {
      if (PolarVolume_hasAttribute((PolarVolume_t*)obj,"how/malfunc")) 
        return (bool)PolarVolume_hasAttribute((PolarVolume_t*)obj,"how/malfunc");
      else
        return false;
    } else {
      if (PolarScan_hasAttribute((PolarScan_t*)obj,"how/malfunc")) 
        return (bool)PolarScan_hasAttribute((PolarScan_t*)obj,"how/malfunc");
      else
        return false;
      
    }
  }
  
  RaveCoreObject * Compositing::_remove_malfunc_from_volume(RaveCoreObject *obj, bool is_polar) {
    // FIXME: memory handling correct?  
    RaveCoreObject*result = obj;
    if (is_polar) {
      if (_get_malfunc_from_obj(obj,is_polar)) {
        RAVE_DEBUG3("Malfunc volume found. Source: %s, Nominal date and time: %sT%s",
                    PolarVolume_getSource((PolarVolume_t*)obj),
                    PolarVolume_getDate((PolarVolume_t*)obj),
                    PolarVolume_getTime((PolarVolume_t*)obj));
        return 0;
      }
      int i = PolarVolume_getNumberOfScans((PolarVolume_t*) obj);
      for (;  i > 0; i--) {
        RaveCoreObject * scan = (RaveCoreObject *)PolarVolume_getScan((PolarVolume_t*) obj, i);
        if (_get_malfunc_from_obj(scan,is_polar)) {
          RAVE_DEBUG4("Malfunc scan with elangle %f found. Removing from volume. Source: %s, Nominal date and time: %sT%s",
              (PolarScan_getElangle((PolarScan_t*) scan) * 180.0/M_PI),
              PolarScan_getSource((PolarScan_t*)scan),
              PolarScan_getDate((PolarScan_t*)scan),
              PolarScan_getTime((PolarScan_t*)scan));
          PolarVolume_removeScan((PolarVolume_t*) obj, i);
        }
        i=PolarVolume_getNumberOfScans((PolarVolume_t*) obj) - 1;  
      }
          
          
  }
    return result;
  }
          
  RaveCoreObject * Compositing::_remove_malfunc(RaveCoreObject*obj, bool is_polar)
  {
    RaveCoreObject*result = obj;
    if (is_polar) {
      result = _remove_malfunc_from_volume(obj,is_polar);
      if ((result != 0) && (PolarVolume_getNumberOfScans((PolarVolume_t*)result) == 0)) {
        RAVE_DEBUG0("All scans of the volume were detected as malfunc. Complete volume therefore considered as malfunc.");
        result = 0;
      }
    } else {
      if (_get_malfunc_from_obj(obj,is_polar)) {
        result = 0;
      }
    }            
    return result;
  }
  

  void Compositing::_debug_generate_info(std::string  area) {
    if (verbose) {
      RAVE_DEBUG1("Generating cartesian image from %d files",filenames.size());
      // loop over detectors.
      //RAVE_DEBUG1("Detectors = '%s'",detectors);
      RAVE_DEBUG1("Quality control mode = '%s'",quality_control_mode.c_str());
      RAVE_DEBUG1("Product = '%s'",_product_repr().c_str());
      RAVE_DEBUG1("Quantity = '%s'", quantity.c_str());
      RAVE_DEBUG1("Range = %f",range);
      RAVE_DEBUG3("Gain = %f, Offset = %f, Minvalue = %f",gain, offset, minvalue);
      RAVE_DEBUG1("Prodpar = '%s'",prodpar.c_str());
      RAVE_DEBUG1("Selection method = '%s'",_selection_method_repr().c_str());
      RAVE_DEBUG1("Interpolation method = '%s'",_interpolation_method_repr().c_str());
      RAVE_DEBUG1("Gap filling = %d",applygapfilling);
      RAVE_DEBUG1("Ct filtering = %d",applyctfilter);
      RAVE_DEBUG1("Gra filtering = %d",applygra);
      RAVE_DEBUG1("Ignoring malfunc = %d",ignore_malfunc);
      RAVE_DEBUG1("QI-total field = '%s'",qitotal_field.c_str());
      RAVE_DEBUG1("Reprocess quality fields = %d",reprocess_quality_field);
      RAVE_DEBUG1("Dumping path = '%s'",dumppath.c_str());
      RAVE_DEBUG1("Dumping output = %d", dump);
      RAVE_DEBUG1("Use site source = %d",use_site_source);
      RAVE_DEBUG1("Use lazy loading = %d",use_lazy_loading);
      RAVE_DEBUG1("Use lazy loading preload = %d",use_lazy_loading_preloads);
      
      if (area.length()) {
        RAVE_DEBUG1("Area = '%s'",area.c_str());
      }
      else {
          RAVE_DEBUG0("Area = 'best fit'");
          RAVE_DEBUG1("pcsid = '%s'",pcsid.c_str());
          RAVE_DEBUG2("xscale = %f, yscale = %f",xscale, yscale);
      }
    }
  }
  /*# Generates the cartesian image.
   #                                    *
   # @param dd: date in format YYYYmmdd
   # @param dt: time in format HHMMSS
   # @param area: the area to use for the cartesian image. If none is specified, a best fit will be atempted.
   */
  Cartesian_t* Compositing::_generate(std::string dd, std::string dt, std::string area) {
    _debug_generate_info(area);
    if (verbose) {
      RAVE_INFO1("Fetching objects and applying quality plugins", mpname.c_str());
    }
      
    RAVE_DEBUG3("Generating composite with date and time %sT%s for area %s", dd.c_str(), dt.c_str(), area.c_str());
    
    // In C++, we can only return one datatype, not many as i python.
    std::map<std::string,RaveCoreObject*> objects;
    std::string nodes;
    std::string how_tasks;
    bool all_files_malfunc;
    
    objects = fetch_objects(nodes,how_tasks,all_files_malfunc);
    
    if (all_files_malfunc) {
      RAVE_INFO0("[radarcomp_c] compositing.generate: Content of all provided files were marked as 'malfunc'. Since option 'ignore_malfunc' is set, no composite is generated!");
      return 0;
    }
    std::string qfields;
    // just a dummy for now.
    CompositeAlgorithm_t * algorithm = 0;
    objects = quality_control_objects(objects,algorithm,qfields);
    
    std::string qccontrols;
    if (qccontrols.length() == 0)
      qccontrols = qfields;
    else
      qccontrols+= "," + qfields;
    RAVE_DEBUG1("[radarcomp_c] compositing.generate: Quality controls for composite generation: %s", qccontrols.c_str());
      
    if (objects.size() == 0) {
      RAVE_INFO0("[radarcomp_c] compositing.generate: No objects provided to the composite generator. No composite will be generated!");
      return 0;
    }
    
    std::vector<RaveCoreObject*> vobjects;
        
    for(auto const& obj: objects)
      vobjects.push_back(obj.second);
        
    if (dump)
      _dump_objects(vobjects);
    
    Composite_t* generator = (Composite_t *)RAVE_OBJECT_NEW(&Composite_TYPE);
    if (generator == 0) {
      RAVE_CRITICAL0("Failed to allocate memory for composite.");
      return 0;
    }
    
    //# Projection and area registries
    // Hard coded standard installation
    std::string RAVECONFIG="/usr/lib/rave/config/";
    std::string PROJECTION_REGISTRY = RAVECONFIG + "projection_registry.xml";
    std::string AREA_REGISTRY = RAVECONFIG + "area_registry.xml";
    
    
    ProjectionRegistry_t* proj_registry = ProjectionRegistry_load(PROJECTION_REGISTRY.c_str());
    if (proj_registry == 0) {
      RAVE_CRITICAL0("Failed to create projection registry for composite.");
      RAVE_OBJECT_RELEASE(generator);
      return 0;
    }
    AreaRegistry_t* area_registry = AreaRegistry_load(AREA_REGISTRY.c_str(), proj_registry);
    if (area_registry == 0) {
      RAVE_CRITICAL0("Failed to create area registry for composite.");
      RAVE_OBJECT_RELEASE(proj_registry);
      RAVE_OBJECT_RELEASE(generator);
      return 0;
    }
    
    Area_t* the_area = 0;
    
    if (area.length()) {
       the_area = AreaRegistry_getByName(area_registry, area.c_str());
    }
    
    if (the_area==0) {
      RAVE_CRITICAL1("Failed to get area %s from area registry.", area.c_str());
      RAVE_OBJECT_RELEASE(proj_registry);
      RAVE_OBJECT_RELEASE(area_registry);
      RAVE_OBJECT_RELEASE(generator);
      return 0;
    }
    
    Composite_addParameter(generator, quantity.c_str(), gain, offset, minvalue);
    Composite_setProduct(generator, product);
    if (algorithm != 0)
      generator->algorithm = algorithm;
    // FIXME: Create RaveObjectHashTable_t* object
    radar_index_mapping = (RaveObjectHashTable_t*)RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
    if (radar_index_mapping == 0) {
      RAVE_CRITICAL0("Failed to allocate memory for radar_index_mapping.");
      RAVE_OBJECT_RELEASE(the_area);
      RAVE_OBJECT_RELEASE(proj_registry);
      RAVE_OBJECT_RELEASE(area_registry);
      RAVE_OBJECT_RELEASE(generator);
      return 0;
    }
    // Add the objects to composite
    int i = 0;
    for (auto const& o:vobjects) {
      Composite_add(generator, (RaveCoreObject*) o);
      //We want to ensure that we get a proper indexing of included radar
      std::string sourceid = PolarVolume_getSource((PolarVolume_t*) o);
      ODIM_Source odim_source(sourceid);
      if (odim_source.nod.length()) { 
        sourceid = "NOD:" + odim_source.nod;
      } else if (odim_source.wmo.length()) {
        sourceid = "WMO:" + odim_source.wmo;
      } else if (odim_source.rad.length()) {
        sourceid = "RAD:" + odim_source.rad;
      }
      if(!RaveObjectHashTable_exists(radar_index_mapping, sourceid.c_str())) {
        RaveAttribute_t* attr = RaveAttributeHelp_createLong(sourceid.c_str(), i+1);
        if (attr != NULL) {
          if (!RaveObjectHashTable_put(radar_index_mapping, sourceid.c_str(), (RaveCoreObject*)attr)) {
            RAVE_ERROR0("Failed to add attribute to radar index mapping");
          }
        }
        RAVE_OBJECT_RELEASE(attr);
      }
    } 
    /*
    for (int i = 0; i < nrObjects; i++) {
      char* sourceId = MyCode_extractSourceFromRadar(myobjects[i]);   // Returnerar WMO:12345 eller NOD:seang eller något sådant. Finns kod i composite.c hur man extraherar sådan information från volymer/scan
      RaveAttribute_t* attr = RaveAttribute_createLong(sourceid, i+1);
      if (attr != NULL) {
        if (!RaveObjectHashTable_put(sourceid, attr)) {
          RAVE_ERROR0("Failed to add attribute to radar index mapping");
        }
      }
      RAVE_OBJECT_RELEASE(attr);
    }*/
    Composite_setSelectionMethod(generator, selection_method);
    Composite_setInterpolationMethod(generator, interpolation_method);
    std::string date = PolarVolume_getDate((PolarVolume_t*)vobjects[0]);
    std::string time = PolarVolume_getTime((PolarVolume_t*)vobjects[0]);
    if (dd.length())
      date=dd;
    if (dt.length())
      time=dt;
    Composite_setDate(generator, date.c_str());
    Composite_setTime(generator, time.c_str());
    Composite_setHeight(generator, height);
    Composite_setElevationAngle(generator, elangle);
    Composite_setRange(generator, height);
    
    if (!qitotal_field.empty())
      Composite_setQualityIndicatorFieldName(generator, qitotal_field.c_str());
      
    if (!prodpar.empty())
      _update_generator_with_prodpar(generator);
        
    if (verbose)
      RAVE_INFO0("[radarcomp_c] compositing.generate: Generating cartesian composite");
    // Composite_applyRadarIndexMapping(Composite_t* composite, RaveObjectHashTable_t* mapping);
    Composite_applyRadarIndexMapping(generator, radar_index_mapping);
    //generator.applyRadarIndexMapping(self.radar_index_mapping)
    Cartesian_t* result = Composite_generate(generator, the_area, 0);
    
    if (applyctfilter) {
      if (verbose)
        RAVE_DEBUG0("[{radarcomp_c}] compositing.generate: Applying ct filter");
      RAVE_INFO1("[{radarcomp_c}] compositing.generate: Applying ct filter for %s not implemented", quantity.c_str());
    }
    
    if (applygra) {
      if (qfields.find("se.smhi.composite.distance.radar")== std::string::npos) {
        RAVE_INFO0("[radarcomp_c] compositing.generate: Trying to apply GRA analysis without specifying a quality plugin specifying the se.smhi.composite.distance.radar q-field, disabling...");
      } else {
        if (verbose)
          RAVE_INFO2("[radarcomp_c] compositing.generate: Applying GRA analysis (ZR A = %f, ZR b = %f)",zr_A,zr_b);
        CartesianParam_t  * grafield = _apply_gra(result, dd, dt);
        if (grafield) {
          Cartesian_addParameter(result, grafield);
        } else {
          RAVE_WARNING0("[radarcomp_c] compositing.generate: Failed to generate gra field....");
        }
      }
    }
    
    //Hack to create a BRDR field if the qfields contains se.smhi.composite.index.radar
    if (qfields.find("se.smhi.composite.index.radar")!= std::string::npos) {
      RAVE_INFO0("[radarcomp_c] compositing.generate: Trying to create a BRDR field not implemented yet!");
      /*
      bitmapgen = _bitmapgenerator.new()
      brdr_field = bitmapgen.create_intersect(result.getParameter(self.quantity), "se.smhi.composite.index.radar")
      brdr_param = result.createParameter("BRDR", _rave.RaveDataType_UCHAR)
      brdr_param.setData(brdr_field.getData())
      */
    }
    RaveAttribute_t* nodes_attr = RaveAttributeHelp_createString("how/nodes", nodes.c_str());
    if (nodes_attr != NULL) {
      if (!Cartesian_addAttribute(result, nodes_attr)) {
        RAVE_ERROR0("Failed to add attribute how/nodes to composite");
      }
    }
    RAVE_OBJECT_RELEASE(nodes_attr);
    if (how_tasks.length()) {
      RaveAttribute_t* tasks_attr = RaveAttributeHelp_createString("how/tasks", how_tasks.c_str());
      if (tasks_attr != NULL) {
        if (!Cartesian_addAttribute(result, tasks_attr)) {
          RAVE_ERROR0("Failed to add attribute how/tasks to composite");
        }
      }
      RAVE_OBJECT_RELEASE(tasks_attr);
    }
    
    if (verbose)
      RAVE_DEBUG0("[radarcomp_c] compositing.generate: Returning resulting composite image");
   
    RAVE_OBJECT_RELEASE(radar_index_mapping);
    RAVE_OBJECT_RELEASE(the_area);
    RAVE_OBJECT_RELEASE(proj_registry);
    RAVE_OBJECT_RELEASE(area_registry);
    RAVE_OBJECT_RELEASE(generator);
    return result;
  }
  /*#
   # Dumps the objects on the ingoing po*lar objects onto the file system. The names will contain a unique identifier
   # to allow for duplicate versions of the same object.
   # @param objects the objects to write to disk */
  void Compositing::_dump_objects(std::vector<RaveCoreObject*> & vobjects){
    // Implement later
  }
  /*#
   # Apply gra coefficient adjustment.  *
   # @param result: The cartesian product to be adjusted
   # @param d: the date string representing now (YYYYmmdd)
   # @param t: the time string representing now (HHMMSS)
   # @return the gra field with the applied corrections */
  CartesianParam_t * Compositing::_apply_gra(Cartesian_t* result, std::string d, std::string t) {
    return 0;
  }
  /*#
   # @return the string representation o*f the selection method
   */
  std::string Compositing::_selection_method_repr() {
    if (selection_method == CompositeSelectionMethod_t::CompositeSelectionMethod_NEAREST)
      return "NEAREST_RADAR";
    else if (selection_method == CompositeSelectionMethod_t::CompositeSelectionMethod_HEIGHT)
      return "HEIGHT_ABOVE_SEALEVEL";
    return "Unknown";
  }
  /*#
   # @return the string representation o*f the interpolation method */
  std::string Compositing::_interpolation_method_repr() {
    if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_NEAREST)
      return "NEAREST_VALUE";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_HEIGHT)
      return "LINEAR_HEIGHT";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_RANGE)
      return "LINEAR_RANGE";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_AZIMUTH)
      return "LINEAR_AZIMUTH";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_RANGE_AND_AZIMUTH)
      return "LINEAR_RANGE_AND_AZIMUTH";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_LINEAR_3D)
      return "LINEAR_3D";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_QUADRATIC_HEIGHT)
      return "QUADRATIC_HEIGHT";
    else if (interpolation_method == CompositeInterpolationMethod_t::CompositeInterpolationMethod_QUADRATIC_3D)
      return "QUADRATIC_3D";
    return "Unknown";
  }
  /*#
   # @return the string representation o*f the product type   */
  std::string Compositing::_product_repr() {
    
    if (product == Rave_ProductType::Rave_ProductType_PPI)
      return "ppi";
    else if (product== Rave_ProductType::Rave_ProductType_CAPPI)
      return "cappi";
    else if (product== Rave_ProductType::Rave_ProductType_PCAPPI)
      return "pcappi";
    else if (product== Rave_ProductType::Rave_ProductType_PMAX)
      return "pmax";
    else if (product== Rave_ProductType::Rave_ProductType_MAX)
      return "max";
    else {
      return "unknown";
    }
  }
  
  void Compositing::_update_generator_with_prodpar(Composite_t* generator) {
    if (prodpar.length() != 0) { 
      if ((product == Rave_ProductType::Rave_ProductType_CAPPI) || (product==Rave_ProductType::Rave_ProductType_PCAPPI)) {
        generator->height = _strToNumber(prodpar);
      }
      else if (product ==  Rave_ProductType::Rave_ProductType_PMAX) {
          std::vector<std::string> pp;
          std::istringstream f(prodpar);
          std::string s;    
          while (getline(f, s, ',')) {
            pp.push_back(s);
          }
          // FIXME: Do we need to strip withspaces?
          if (pp.size() == 2) {
            generator->height = _strToNumber(pp[0]);
            generator->range = _strToNumber(pp[1]);
          }
          else if (pp.size() == 1) {
            generator->height = _strToNumber(pp[0]);
          }
      }
      else if (product == Rave_ProductType::Rave_ProductType_PPI) {
            float v = _strToNumber(prodpar);
            generator->elangle = v * M_PI / 180.0;
      }
    }
  }
  
  float Compositing::_strToNumber(std::string sval) {
    return std::stof(sval);
  }
