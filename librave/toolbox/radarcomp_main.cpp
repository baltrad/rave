/* --------------------------------------------------------------------
 C *opyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,
 
 This is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This software is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License
 along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
 ------------------------------------------------------------------------*/
#include "compositing.h"
extern "C" {
#include "rave_object.h"
#include "rave_debug.h"
#include "composite.h"
#include "rave_io.h"
}

#include "optparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <hlhdf.h>

/** Main function for a binary for running KNMI's sun scanning functionality
 * @file
 * @author Yngve Einarsson, SMHI
 * @date 2024-12-09
 */
int Radarcomp(optparse::OptionParser & parser) {
  Compositing comp = Compositing();
  // Split infiles and detectors
  std::vector<std::string> fparts;
  std::istringstream f(parser.get_option("infiles"));
  std::string s;    
  while (getline(f, s, ',')) {
    fparts.push_back(s);
  }
  comp.filenames = fparts;
  std::vector<std::string> dparts;
  std::istringstream d(parser.get_option("qc"));
  while (getline(d, s, ',')) {
    dparts.push_back(s);
  }
  comp.detectors = dparts;
  comp.quantity = parser.get_option("quantity");
  comp.set_product_from_string(parser.get_option("product"));
  comp.range = std::stof(parser.get_option("range"));
  comp.gain = std::stof(parser.get_option("gain"));
  comp.offset = std::stof(parser.get_option("offset"));
  comp.minvalue = std::stof(parser.get_option("minvalue"));
  comp.prodpar = parser.get_option("prodpar");
  comp.set_method_from_string(parser.get_option("method"));
  comp.qitotal_field = parser.get_option("qitotal_field");
  comp.pcsid = parser.get_option("pcsid");
  comp.xscale = std::stof(parser.get_option("scale"));
  comp.yscale = std::stof(parser.get_option("scale"));
  comp.set_interpolation_method_from_string(parser.get_option("interpolation_method"));
  comp.use_azimuthal_nav_information == !((bool)std::stoi(parser.get_option("disable_azimuthal_navigation")));
  comp.zr_A = std::stof(parser.get_option("zr_A"));
  comp.zr_b = std::stof(parser.get_option("zr_b"));
  
  if ((bool)std::stoi(parser.get_option("gf")))
    comp.applygapfilling = true;
  if ((bool)std::stoi(parser.get_option("ctfilter")))
    comp.applyctfilter = true;
  if ((bool)std::stoi(parser.get_option("grafilter")))
    comp.applygra = true;
  if ((bool)std::stoi(parser.get_option("ignore_malfunc")))
    comp.ignore_malfunc = true;
  if ((bool)std::stoi(parser.get_option("verbose")))
    comp.verbose = true;
            
  if (comp.verbose)
    Rave_setDebugLevel(Rave_Debug::RAVE_SPEWDEBUG);
    //comp.logger = rave_pgf_logger.rave_pgf_stdout_client("debug")
    
  else
    Rave_setDebugLevel(Rave_Debug::RAVE_INFO);
    //comp.logger = rave_pgf_logger.rave_pgf_stdout_client("info")
                
  comp.reprocess_quality_field = false;
  if ((bool)std::stoi(parser.get_option("reprocess_quality_fields")))
     comp.reprocess_quality_field = true;
                  
  if (!(parser.get_option("area").empty()) && !((bool)std::stoi(parser.get_option("noMultiprocessing")))) {
    RAVE_INFO0("Tiled_compositing not implemented yet, using standard compositing");
    /*
    if rave_tile_registry.has_tiled_area(options.area):
      preprocess_qc = False
      mp_process_qc = False
      mp_process_qc_split_evenly = False
      if options.preprocess_qc:
        preprocess_qc = True
      if options.preprocess_qc_mp:
        preprocess_qc = True
        mp_process_qc = True
      if options.mp_process_qc_split_evenly:
        mp_process_qc_split_evenly = True
      comp = tiled_compositing(comp, preprocess_qc, mp_process_qc, mp_process_qc_split_evenly)
      comp.number_of_quality_control_processes = options.number_of_quality_control_processes
      */
  }
  // result is Cartesian_t*

  Cartesian_t* result = comp.generate(parser.get_option("date"), parser.get_option("time"), parser.get_option("area"));
  
  if ((bool)std::stoi(parser.get_option("imageType")))
    Cartesian_setObjectType(result, Rave_ObjectType::Rave_ObjectType_IMAGE);
  
  RaveIO_t* rio = (RaveIO_t*)RAVE_OBJECT_NEW(&RaveIO_TYPE);
  if (rio == 0) {
    RAVE_CRITICAL0("Failed to allocate memory for raveIO.");
    RAVE_OBJECT_RELEASE(result);
    return 1;
  }
  RaveIO_setObject(rio, (RaveCoreObject*)result);
  RaveIO_setFilename(rio, parser.get_option("outfile").c_str());
 if (comp.verbose)
    RAVE_INFO1("Saving %s",RaveIO_getFilename(rio));
  RaveIO_save(rio, 0);
  RaveIO_close(rio);
  RAVE_OBJECT_RELEASE(result);
  RAVE_OBJECT_RELEASE(rio);
/*                              
  rio = _raveio.new()
  rio.object = result
  rio.filename = options.outfile
                              
  if comp.verbose:
    print("Saving %s"%rio.filename)
  rio.save()*/
  return 0;
}



int main(int argc,char *argv[]) {
  
  Rave_initializeDebugger();
  Rave_setDebugLevel(Rave_Debug::RAVE_SPEWDEBUG);
  
  std::string usage("usage: radarcomp_c -i <infile(s)> -o <outfile> [-a <area>] [args] [h]");
  usage += "\nGenerates weather radar composites directly from polar scans and volumes. If area is omitted, a best fit will be performed.";
  usage += "\nIn that case, specify pcs, xscale and yscale to get an appropriate image.";
  
  optparse::OptionParser parser(usage);
  
  parser.add_option("-i", "--input", "infiles",
                    "Name of input file(s) to composite, comma-separated in quotations.");
  
  parser.add_option("-o", "--output", "outfile",
                    "Name of output file to write.");
  
  parser.add_option("-a", "--area", "area",
                    "Name of Cartesian area to which to generate the composite. If not specified, a best fit composite will be created.");
 
  parser.add_option("-c", "--pcsid", "pcsid",
                    "Name of the pcsid to use if the area should be automatically generated from a best fit. Default is 'gmaps'.",
                    optparse::STORE, optparse::STRING,"gmaps");
  
  parser.add_option("-s", "--scale", "scale",
                    "The x/y-scale to use if the area should be automatically generated from a best fit. Default is 2000.0.",
                    optparse::STORE, optparse::DOUBLE, "2000.0");
  
  parser.add_option("-q", "--quantity", "quantity",
                    "The radar parameter to composite. Default=DBZH.",
                    optparse::STORE, optparse::STRING, "DBZH");
  
  parser.add_option("-p", "--product", "product",
                    "The type of Cartesian product to generate [PPI, CAPPI, PCAPPI, PMAX]. Default=PCAPPI.",
                    optparse::STORE, optparse::STRING,"PCAPPI");
  
  parser.add_option("-P", "--prodpar", "prodpar",
                    "Product parameter. For (P)CAPPIs it is the height of the desired layer. For PPIs, it is the elevation angle. Default=1000.0 (meters).",
                    optparse::STORE, optparse::DOUBLE, "1000.0");
  
  parser.add_option("-r", "--range", "range",
                    "Maximum range to apply PMAX algorithm. Applies only to PMAX algorithm. Defaults to 200 km.",
                    optparse::STORE, optparse::DOUBLE, "200000.0");
  
  //GAIN = 0.4
  //OFFSET = -30.0
  
  parser.add_option("-g", "--gain", "gain",
                    "Linear gain applied to output data. Default=as defined in rave_defines.py.",
                    optparse::STORE, optparse::DOUBLE, "0.4");
  
  parser.add_option("-O", "--offset","offset",
                    "Linear offset applied to output data. Default=as defined in rave_defines.py.",
                    optparse::STORE, optparse::DOUBLE, "-30.0");
  
  parser.add_option("-mv", "--minvalue", "minvalue",
                    "Minimum value that can be represented in composite. Relevant when interpolation is performed. Default=-30.0",
                    optparse::STORE, optparse::DOUBLE, "-30.0");
  
  parser.add_option("-im", "--interpolation_method", "interpolation_method",
                    "Interpolation method to use in composite generation. Default=NEAREST_VALUE",
                    optparse::STORE, optparse::STRING, "NEAREST_VALUE",
                    "NEAREST_VALUE,LINEAR_HEIGHT,LINEAR_RANGE,LINEAR_AZIMUTH,LINEAR_RANGE_AND_AZIMUTH,LINEAR_3D,QUADRATIC_HEIGHT,QUADRATIC_3D");
  
  parser.add_option("-d", "--date", "date",
                    "Nominal date of the composite to be written. Defaults to the nominal date of the last input file.");
  
  parser.add_option("-t", "--time", "time",
                    "Nominal time of the composite to be written. Defaults to the nominal time of the last input file.");
  
  parser.add_option("-m", "--method", "method",
                    "Compositing algorithm to apply. Current choices are NEAREST_RADAR or HEIGHT_ABOVE_SEALEVEL. Default=NEAREST_RADAR.",
                    optparse::STORE, optparse::STRING,"NEAREST_RADAR");
  
  parser.add_option("-Q", "--qc", "qc",
                    "Which quality-controls to apply. Comma-separated, no white spaces. Default=None");
  
  parser.add_option("-G", "--gap-fill", "gf",
                    "Gap-fill small holes in output composite. Default=False",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-C", "--ctfilter", "ctfilter",
                    "Filter residual non-precipitation echoes using SAF-NWC cloud-type product. Default=False",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-A", "--applygra", "grafilter",
                    "Applies the GRA correction coefficients. Default=False",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-y", "--zr_A", "zr_A",
                    "The ZR A coefficient to use for the gra correction. Default=200.0",
                    optparse::STORE, optparse::DOUBLE, "200.0");
  
  parser.add_option("-z", "--zr_b", "zr_b",
                    "The ZR b coefficient to use for the gra correction. Default=1.6",
                    optparse::STORE, optparse::DOUBLE, "1.6");
  
  parser.add_option("-F", "--qitotal_field", "qitotal_field",
                    "The QI-total field to use when creating the composite from the qi-total Default=Not used.");
  
  parser.add_option("-I", "--ignore-malfunc", "ignore_malfunc",
                    "If scans/volumes contain malfunc information. Don't use them in the composite. Default is to always use everything.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-V", "--verbose", "verbose",
                    "If the different steps should be displayed. I.e. verbose information.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-T", "--imageType", "imageType",
                    "If the stored file should be saved as an IMAGE instead of a COMP (osite).",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-M", "--no-multiprocessing", "noMultiprocessing",
                    "Disable multiprocessing even if an entry exists in the tile_registry",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-pq", "--preprocess_qc", "preprocess_qc",
                    "Preprocesses the quality fields and stores these as temporary files. This is really only useful when performing tiled processing.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-pm", "--preprocess_qc_mp", "preprocess_qc_mp",
                    "Preprocesses the quality fields in the provided files and uses multiprocessing to do this.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-np","--number_of_quality_control_processes", "number_of_quality_control_processes",
                    "Number of processes that should be used for performing the quality control. Default 4. Requires that --preprocess_qc_mp is used.",
                    optparse::STORE, optparse::INT, "4");
  
  parser.add_option("-me","--mp_process_qc_split_evenly", "mp_process_qc_split_evenly",
                    "Splits the incomming files evenly among the quality control processes. Requires that --preprocess_qc_mp is used.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-rq", "--reprocess_quality_fields", "reprocess_quality_fields",
                    "Reprocessed the quality fields even if they already exist in the object.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  
  parser.add_option("-dn", "--disable_azimuthal_navigation", "disable_azimuthal_navigation",
                    "If this flag is set, then azimuthal navigation won't be used when creating the composite.",
                    optparse::STORE_TRUE, optparse::BOOL, "0");
  try {
    parser.parse_args(argc,argv);
  }
  catch (optparse::OptionError &e) {
    RAVE_ERROR1("Option error, terminate: %s", e.what());
    exit(1);
  }
  catch (...) {
    RAVE_ERROR0("Unexpected exception, terminate");
    exit(1);
  }
  if ((parser.get_option("infiles").length() != 0) && (parser.get_option("outfile").length() != 0)) {
    printf("Radarcomp--->\n");
    int result = Radarcomp(parser); 
    printf("Radarcomp <---\n");
  }
  else {
      std::ostringstream o;
      parser.help(o);
      std::cerr << o.str();
  }
  
  exit(0);
}
