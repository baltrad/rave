/**

    Copyright 2001 - 2010  Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Rack library for C++.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "Rack.h"

#include <iostream>
#include <map>
#include <exception>

#include <drain/util/Debug.h>
#include <drain/util/Data.h>
#include <drain/util/StringMapper.h>
#include <drain/util/TreeNode.h>

#include <drain/image/CoordinateHandler.h>
#include <drain/image/File.h>
#include <drain/image/Drain.h>
#include <drain/image/CatenatorOp.h>
#include <drain/image/Cumulator.h>
#include <drain/image/PaletteOp.h>

#include <drain/radar/Coordinates.h>
#include <drain/radar/PolarToCartesian.h>
#include <drain/radar/PolarToGeographical.h>
#include <drain/radar/SubComposite.h>

using namespace std;
using namespace drain::image;
using namespace drain::radar;

extern "C"
{
#include <polarvolume.h>
#include <rave_debug.h>
}

namespace rack
{

PolarScan_t* Rack::getPolarScan()
{
	currentView.properties = currentImage->properties;	
	return raveConv.createPolarScan(*currentImage, currentView, (const string &)options["defaultQuantity"]);
}

Rack::Rack ()
{
}

Rack::~Rack()
{
}


void Rack::init(){

	// These are the "true", memory-allocating images
	inputImage.setName("input");
	polarProduct.setName("polar");
	cartesianProduct.setName("cartesian");
	colourProduct.setName("coloured");

	currentImage = &inputImage;
	currentPolarImage = &inputImage;

	//Initialize options
	options.reader.setKeySyntax("[^\\-\\=]+");

	// The options should be declared here (even with empty values)
	// so that they can be read in command line options in
	// --KEY VALUE format, not only as --KEY=VALUE

	// TODO: readonly
	options["version"].description = "Display version information.";

	options["help"] = "General help.";
	options.setAlias("help", 'h');

	options["man"] = "Help on a command or command set.";
	options["man"].syntax = "<command>|drain|andre";
	options.setAlias("man", 'H');

	options["status"] = "Dump information on current images.";

	options["dumpMap"].description = "Dump variable map, filtered by keys, to std or file.";
	options["dumpMap"].separators = ":";
	options["dumpMap"].syntax = "<regexp>[:<file>]";
	options["dumpMap"] = ".*:out.txt";

	options["view"] = "f";
	options["view"].description
	= "Current view to point to full image, image channels, alpha channels, n'th channel or flat (concatenaed) image.";
	options["view"].syntax = "[f|i|a|<n>|F]";
	options.setAlias("view",'v');

	options["cartesian"].description
	= "Maps polar to cartesian (Azimuthal equidistant). Adds alpha channel, if applied.";
	options["cartesian"].syntax = PolarToCartesian<>().getParameterNames();

	// Anomaly detection and removal
	options["aClear"].description = "Frees memory by clearing the feature cache. Clears detection results as well.";

	options["a<DETECTOR>"].description = "Syntax for AnDRe anomaly detectors, see '--man andre'.";
	options["a<DETECTOR>"].syntax = "<class,parameters>";

	options["r<DETECTOR>"].description = "Syntax for old RoPo anomaly detectors, see '--man ropo'.";
	options["r<DETECTOR>"].syntax = "<class,parameters>";


	options["aCumulated"].description = "Moves current image ptr to cumulated anomaly field.";

	options["aGapFill"].description = "Cleans data by mixing good-quality pixels to low-quality pixels. Call after 'aPaste'.";
	options["aGapFill"].syntax = DistanceTransformExponentialOp<>().getParameterNames();

	options["aPaste"].description = "Inverts and stores anomaly field to input volume.";

	options["aSave"].description = "Saves cumulated detection field.";
	options["aSave"].syntax = "<filename>";

	options["aSaveCurrent"].description = "Saves current detetion field.";
	options["aSaveCurrent"].syntax = "<filename>";

	//  COMPOSITING COMMANDS
	options["cSize"].description = "Prepares to a create a composite of this size. Does not allocate memory yet.";
	options["cSize"].syntax = "<width>,<height>";

	options["cFade"].description = "Weight (and forecast) for previous results. Set before calling 'cLoad'.";
	options["cFade"].syntax = "<fade,dx,dy>";
	options["cFade"] = "0.75,0,0";

	options["cLoad"].description = "Loads a (previous) composite, to serve as initial image. See 'cFade'.";
	options["cLoad"].syntax = "<file>";

	options["cLoadTile"].description = "Loads a tile image to be added onto a composite with 'cAddTile'.";
	options["cLoadTile"].syntax = "<file>";

	options["cTileConfOut"].description = "Saves current tile information, especially offsets, to <file>. ";
	options["cTileConfOut"].syntax = "<file>";

	options["cMethod"].description
	= "Formula to be used in cumulating the (weighted) values.";
	options["cMethod"].syntax = "<MAX|MAXW|AVG|WAVG,p,r>";

	options["cClose"].description = "Resets compositing image to zero size";

	options["cExtract"].description = "From resulting composite, saves channel(s) to cartesian product.";
	options["cExtract"].syntax = "<dwcsDWCS*>  (data,weight,count,stddev; scaled and UNSCALED)";

	options["cCreateTile"].description
	= "Creates a (minimum) tile ie. subimage from current polar product to Cartesian product. Remember to set ORIGIN.";
	options["cCreateTile"].syntax = "<dwcsDWCS*>  (data,weight,count,stddev; scaled and UNSCALED)";
	options["cCreateTile"] = "dw";

	options["cClear"].description = "Clears the current composite values, keeps the geometry.";

	options["cAddTile"].description
	= "Adds current polar product to composite using intermediate image. Extracts channels like [cExtract]. Remember ORIGIN.";
	options["cAddTile"].syntax = "<weight>";

	options["cAdd"].description
	= "Adds current polar product to composite using intermediate image. Extracts channels like [cExtract]. Remember ORIGIN.";
	options["cAdd"].syntax = "<weight>";

	options["cInterpolation"].description
	= "Sets the method of interpolation: distance transform or recursive weighted averaging.";
	options["cInterpolation"].syntax
	= "d[,<rect>,<diag>] | w[,<width>,<height>,<rounds>,<decay>]";

	options["cBBox"].description = "Bounding box of the composite. In degrees.";
	options["cBBox"].syntax = "<lonLL,latLL,lonUR,latUR>";
	options["cBBox"] = "19.02,58.01,32.04,71.03";

	options["debug"].description = "Debug (verbosity) level.";
	options["debug"].syntax = "<level>";
	options["debug"] = 0;

	options["palette"].description = "Load and apply palette (1x256 image file for example).";
	options["palette"].syntax = "<file>";

	options["gainInt"].description = "Internally applied gain for: dBZ = gainInt*byte + offsetInt.";
	options["gainInt"] = 0.5;

	options["offsetInt"].description = "Internally applied offset for: dBZ = gainInt*byte +offsetInt.";
	options["offsetInt"] = -32;

	options["defaultQuantity"].description = "The default quantity. This quantity is read from the PolarScan";
	options["defaultQuantity"] = "DBZH";

	options["format"].description = "Sets a format string.";
	options["format"].syntax = "<string contaning ${VARIABLES}>";

	options["formatFile"].description = "Reads a format string form a file.";
	options["formatFile"].syntax = "<filename>";

	options["formatOut"].description = "Dumps the formatted string to a file or stdout.";
	options["formatOut"].syntax = "<filename>|-";

	options["h5Data"].description = "Data from h5 file is selected with this pattern. Wildcards like ?,*,[abc], [0-9] and ** allowed";
	options["h5Data"].syntax = "<pattern>";
	options["h5Data"] = "/**1/data";

	options["detectBBox"].description = "Based on 'rscale','lat','lon' and 'nbins', updates 'BBox{Lat,Lon}{Min,Max}'";
	options["detectBBox"].syntax = "";

	options["inputAlpha"].description = "Input alpha (quality) channel, inverted clutter map for example";
	options["inputAlpha"].syntax = "<filename>";

	options["outputFile"] = "";
	options["outputFile"].description = "Output current image to a file";
	options["outputFile"].syntax = "<filename>";
	options.setAlias("outputFile", 'o');

	options["proj"].description = "(Future extension.) Sets the projection of a composite product.";
	options["proj"].syntax = "<proj4 syntax>";
	options["proj"] = "proj=stere lon_0=25.00e lat_0=60.00n";

	proj.setProjection((const string) options["proj"]);

	options["quality"].description = "Associate quality to input volume";
	options["quality"].syntax = "<file|a|>";

	// Todo, is "size" used?
	vector<unsigned int> size;
	options["size"] = "300,500";
	options["size"].splitTo(size);
	options["size"].syntax = "<width>,<height>";

	options["SITE"].description = "A shorthand for giving coordinates.";
	options["SITE"].syntax = "<string>";
	options["SITE"] = "VAN";

	options["RANGE"].description = "A variable automatically set to rscale*nbins read last.";
	options["RANGE"].syntax = "<kms>";
	options["RANGE"] = 250000;

	options["/<path...>/<attribute>"].description = "Sets ODIM attribute of the current image. May also start with @.";
	options["/<path...>/<attribute>"].syntax = "<value>";
	options["/<path...>/<attribute>"] = "";

	// Non-ODIM variables
	options["beamwidth"].description = "Width of the radar beam in degrees. Non-standard.";
	options["beamwidth"] = "0.9";
	options["beamwidth"].syntax = "<degrees>";

	options["origin"] = "25,60";
	options["origin"].syntax = "<lon>,<lat>";
	options["origin"].description = "Shorthand for --/where/lon <lon> --/where/lat <lat> .";

	localConf["ANJ"] = "27.10,60.90"; //6054 2706
	localConf["IKA"] = "23.07,61.77"; //6146 2304
	localConf["KER"] = "22,60.12"; //6007 2138
	localConf["KOR"] = "21.63,60.12"; //6007 2138
	localConf["KUO"] = "27.38,62.85"; //6251 2723
	localConf["LUO"] = "26.90,67.13"; //6708 2654
	localConf["UTA"] = "26.27,64.77"; //6446 2619
	localConf["VAN"] = "24.87,60.27"; //6016 2452
	localConf["VIM"] = "23.82,63.10"; //6306 2349

	localConf.reader.read("locations.cnf");

	drainHandler.addDefaultOps();
	polarProductHandler.addOperator("cappi", cappi);
	polarProductHandler.addOperator("maxEcho", maxEcho);
	polarProductHandler.addOperator("echoTop", echoTop);
	polarProductHandler.addOperator("attenuation", attenuation);

	andre.addDefaultOps();
	andre.prefixAll("a");

	ropo.addDefaultOps();
	ropo.prefixAll("r");
}

void Rack::getDefaultQuantity(std::vector<string>& argv)
{
	for (int i = 0; i < argv.size(); i++)
	{
		if (argv[i] == "--quantity" && i < (argv.size() - 1))
		{
			options["defaultQuantity"] = argv[i+1];
			if (drain::Debug > 0)
 			{			
				cout << "Setting default quantity to " << argv[i+1] << endl; 				
			}
			argv.erase(argv.begin()+i+1);
			argv.erase(argv.begin()+i);
			break; //only allow one output file per call
		}
	}	
}

void Rack::convertScanToImage(PolarScan_t* scan)
{
	bool ok = raveConv.convertToRack(scan, inputImage, options);
	RAVE_ASSERT(ok,"Failed to convert from RAVE format to RACK format.");

	const size_t channels = inputImage.properties["@where/elangle"].getVector().size();

	if (channels == 0)
		cerr << "ERROR: rack: No channels/scans found\n";
	else
		inputImage.setGeometry(inputImage.getWidth(), inputImage.getHeight() / channels, channels);

	if (drain::Debug > 1){
		cout << "properties" << '\n';
		cerr << inputImage.properties << '\n';
		if (drain::Debug > 2)
			inputImage.debug(cerr);
	}

	currentPolarImage = &inputImage;
	currentImage = &inputImage;
	currentView.viewImage(*currentImage);
}

double Rack::main(PolarScan_t* scan, std::vector<string> argv){
	
	drain::Debug = 0;
	
	//Initialize data
	init();

	getDefaultQuantity(argv);

	int argc = argv.size();

	bool firstSweeps = true;
	bool inputOk = false;

	if (argc == 0) {
		argc = 1;
		argv.push_back("--help");
	}

	if (scan != NULL) {
	  convertScanToImage(scan);
	  inputOk = true;
	  firstSweeps = false;
	}
	else {
		cerr << "PolarScan was NULL" << endl;
		return -1;
	}

	const drain::RegExp h5FileExtension(".*\\.(h5|hdf5|hdf)$");

	// TODO: start with '@' or '/' if option should be stored in image properties?


	// MAIN LOOP
	string key;
	try {
		for (int i = 0; i < argc; i++) {

			if (drain::Debug > 1)
				cout << "---------------------------------------------------------------\n";
			//cout << argv[i] << '\n';
			options.reader.readCommandLineEntry(argv[i]);

			key = options.reader.getKey();
			key = options.getAlias(key);

			if (drain::Debug > 2)
				cout << "KEY=" << key << endl;



			// If not of form --KEY=VALUE  but --KEY VALUE
			// and map entry exists, treat next argument as VALUE.
			if (!options.reader.hasArgument()) {
				//cerr << "? Reading pending argument of '" << key << "'\n";
				if ((!options[key].isFlag())
						||andre.hasOperator(key)
						||ropo.hasOperator(key)
						||drainHandler.hasOperator(key)
						||polarProductHandler.hasOperator(key)) { //if (options.hasKey(key)){
					// Read next.
					i++;
					if (i == argc) {
						cerr
						<< "Error: unexpected end of command line options."
						<< endl;
						exit(-1);
					}
					options.reader.readPendingCommandLineArgument(argv[i]);
				}
			}

			drain::Data value(options.reader.getValue());
			value.separators = options[key].separators;

			vector<int> valuesInt;
			value.splitTo(valuesInt);

			vector<float> valuesF;
			value.splitTo(valuesF);

			if (key == "") {
				cerr << "Unknown argument: " << argv[i] << endl;
				return -1;				
			} 
			else { // HANDLE COMMAND LINE OPTIONS

				firstSweeps = true;  // TODO

				vector<drain::Data> values; // TODO:same with key if no args?
				value.splitTo(values);

				if (drain::Debug > 1)
					cout << key << " = '" << value << '\'' << endl;

				if (key == "help") {

					// General help
					cout << __RACK__ << '\n';
#ifdef RACK_ANORACK
					cout << ANORACK_H_ << '\n';
#endif
					options.help(cout,"\n\t");
					//polarProductHandler.help(cout, "", "--", "\n\t");
					//andre.help(cout, "", "--", "\n\t");
					cout << " \n";
					//cout << "Type '--help examples' for usage examples.\n";
					cout << "Type '--man <cmd>' for a command.\n";
					cout << "Type '--man prod'  for polar products.\n";
					cout << "Type '--man drain' for general image processing commands.\n";
					cout << "Type '--man andre' for help with new anomaly detectors.\n";
					cout << "Type '--man ropo' for help with old anomaly detectors.\n";
					//cout << drain.help();
					return 0;
				}
				else if (key == "man") {

					if (value == "drain")
						drainHandler.help(cout,"", "--", "\n\t");
					//else if (key == "examples")
					//	examples(cout);
					else if (value == "andre")
						andre.help(cout,"", "--", "\n\t");
					else if (value == "ropo")
						ropo.help(cout,"", "--", "\n\t");
					else if (value == "prod")
						polarProductHandler.help(cout,"", "--", "\n\t");
					else {
						// TODO search
						if (drainHandler.hasOperator(value)) {
							drainHandler.help(cout, value);
						}
						else if (andre.hasOperator(value)) {
							andre.help(cout, value); //,"--","\n\t");
						}
						else if (ropo.hasOperator(value)) {
							ropo.help(cout, value); //,"--","\n\t");
						}
						else if (polarProductHandler.hasOperator(value)) {
							polarProductHandler.help(cout, value); //,"--","\n\t");
						}
						else {
							cerr << "Sorry, does not exist: " << value << '\n';
						}
					}
					return 0;
				}
				else if (key == "status") {
					for (map<string, drain::Data>::const_iterator it =
							options.begin(); it != options.end(); it++) {
						cout << it->first << '=' << it->second << '\n';
					}
					cout << "Image geometry: " << currentImage->getGeometry()
																	<< '\n';
					cout << "View  geometry: " << currentView.getGeometry()
																	<< '\n';
					cout << "Image comments:\n" << currentImage->properties
							<< '\n';
				} else if (key == "cmd") {
					ostringstream s(value);
					int i = std::system(s.str().c_str());
					s << value;
					cout << s << '\n' << i << '\n';
				} else if (key == "quantity") {
					options["defaultQuantity"] = (const string &)value;
					cout << "--setting default quantity to" << (const string &)value << endl; 	
				} else if (key == "proj") {

					proj.setProjection(value);
					if (!proj.ok) {
						cerr << proj.getProjectionString() << endl;
						cerr << proj.getErrorString() << endl;
						exit(-1);
					}
					double x = 25;
					double y = 62;
					//
					proj.project(y, x);
					cerr << x << ' ' << y << endl;

				}
				else if (key == "aCumulated") {
					currentImage = & andre.cumulatedDetection;
					currentView.viewImage(*currentImage);
				}
				else if (key == "aEnhance") {
					andre.enhance(value[0],valuesInt[1]);
					andre.cumulate();
					currentImage = & andre.lastDetection;
					currentView.viewImage(*currentImage);
				}
				else if (key == "aGapFill") {
					DistanceTransformFillOp<> op(value);
					op.filter(inputImage,inputImage);
					currentImage = & inputImage;
					currentView.viewImage(*currentImage);
				}
				else if (key == "aPaste") {
					const unsigned int a = andre.cumulatedDetection.getChannelCount();
					inputImage.setAlphaChannelCount(a);
					currentView.viewChannels(inputImage,a,inputImage.getImageChannelCount());
					NegateOp<>().filter(andre.cumulatedDetection,currentView);
					inputImage.properties["andre"] = andre.cumulatedDetection.properties["detections"];
					andre.reset();
					currentImage = & inputImage;
					// currentPolarImage = ?
					currentView.viewImage(*currentImage);
				}
				// Andre operators
				else if (andre.hasOperator(key)){
					string andreKey = key;
					//if (drain::Debug>2)
					//	cout << "AnDRe:" << andreKey << '\n';
					if (drain::Debug> 0)					
						inputImage.debug();
					andre.computeDetection(inputImage,andreKey,value);
					const string a = andreKey+':'+value;
					andre.lastDetection.properties["detections"] = a;
					andre.cumulate();

					andre.cumulatedDetection.properties["detections"] << a;

					if (drain::Debug>6)
						File::write(andre.cumulatedDetection, andre.cumulatedDetection.properties["detections"]+"-cumul.png");

					currentImage = & andre.lastDetection;
					currentView.viewImage(*currentImage);
					//}
				}
				// Ropo operators
				else if (ropo.hasOperator(key)){
					ropo.computeDetection(inputImage,key,value);
					const string a = key+':'+value;
					ropo.lastDetection.properties["detections"] = a;

					andre.cumulate(ropo.lastDetection);
					andre.cumulatedDetection.properties["detections"] << a;

					if (drain::Debug>6)
						File::write(andre.cumulatedDetection,
								andre.cumulatedDetection.properties["detections"]+"-cumul.png");

					currentImage = & ropo.lastDetection;
					currentView.viewImage(*currentImage);
					//}
				}
				//
				//
				// COMPOSITING
				//
				else if (key == "cAddNoise") {  // later add test
					for (unsigned int j = 0; j < composite.getHeight(); ++j) {
						for (unsigned int i = 0; i < composite.getWidth(); ++i) {
							composite.add(i,j,(j+i)&255,(j&128)+(i&64));
						}
					}
				}
				else if (key == "cAddTile") {  // later add
					if (!inputOk) {
						cerr << "Last volume read unsuccessful, skipping operation\n";
						continue;
					}
					const vector<drain::Data> & offset = options["cTileOffset"].getVector();
					if (offset.size() != 2)
						throw runtime_error(" cAddTile: cTileOffset must have 2 elements.");
					//cartesianProduct.debug();
					cout << offset[0] << ',' << offset[1] << '\n';

					composite.addImage(cartesianProduct.getChannel(0),cartesianProduct.getChannel(1),
							value,offset[0],offset[1]);
					composite.properties["/what/product"] = cartesianProduct.properties["/what/product"];
					composite.properties["/what/prodpar"] = cartesianProduct.properties["/what/prodpar"];
					composite.properties["/what/date"]    = cartesianProduct.properties["/what/date"];
					composite.properties["/what/time"]    = cartesianProduct.properties["/what/time"];
				}
				else if (key == "cBBox") {  // for COMPOSITE
					composite.setBoundingBox(values[0], values[3], values[2], values[1]);
					//}
					//else if (key == "cBBoxLocations") {
					//const int range = value;
					vector<double> coords;
					options["cBBoxLocations"].clear();
					for (map<string, drain::Data>::const_iterator it =
							localConf.begin(); it != localConf.end(); it++) {
						//const vector<drain::Data> &coords = it->second.getVector();

						it->second.splitTo(coords);
						double & lat = coords[1];
						double & lon = coords[0];

						const bool withinVert = (lat > composite.getYLowerLeft()) &&
								(lat < composite.getYUpperRight());

						const bool withinHorz = (lon > composite.getXLowerLeft()) &&
								(lon < composite.getXUpperRight());

						cerr << it->first << ' ' << withinHorz << withinVert << '\n';

						// Is radar center inside?
						if (withinHorz && withinVert)
							options["cBBoxLocations"] << it->first;
						else {
							drain::radar::Coordinates radar;
							radar.setOriginDeg(lat,lon);
							double latMin;
							double lonMin;
							double latMax;
							double lonMax;
							radar.getBoundingBox(250000,latMin,lonMin,latMax,lonMax);
							if (withinHorz) {
								if ((latMax > composite.getYLowerLeft())
										|| (latMin < composite.getYUpperRight()))
									options["cBBoxLocations"] << it->first;
							}
							else if (withinVert) {
								if ((lonMax > composite.getXLowerLeft())
										|| (lonMin < composite.getXUpperRight()))
									options["cBBoxLocations"] << it->first;
							}
							// else corners
						}
					}
				}
				else if (key == "cCreateTile"){ // || (key=="cSaveTile") || (key=="cAddTile")){

					/// CREATE
					if (!inputOk) {
						cerr << "Last volume read unsuccessful, skipping operation\n";
						continue;
					}

					SubComposite subComposite(composite);

					const double lat = currentPolarImage->properties["/where/lat"];
					const double lon = currentPolarImage->properties["/where/lon"];
					const double rscale = currentPolarImage->properties["@where/rscale"];
					const int nbins = currentPolarImage->properties["@where/nbins"];

					//composite.properties["/what/prodpar"] = cartesianProduct.properties["/what/prodpar"];

					if ((lat==0.0)&&(lat==0.0)){
						cerr << "Warning: lat,lon=0,0.\n";
					}

					subComposite.setRadarLocation(lon,lat,rscale,nbins);
					subComposite.detectBoundingBox();
					options["cTileOffset"].clear();
					options["cTileOffset"] << subComposite.getXOffset() << subComposite.getYOffset();
					if (drain::Debug > 2){
						cerr << " lat,lon=" << lat << ',' << lon << '\n';
						cerr << " rscale=" << rscale << '\n';
						cerr << " nbins=" << nbins << '\n';
						cerr << " offset=" << options["cTileOffset"] << '\n';
						//cerr << " lowerLeft=" << subComposite.get << ',' << subComposite.getYLowerLeft() << '\n';
						//cerr << " upperLeft=" << subComposite.getXUpperRight() << ',' << subComposite.getYUpperRight() << '\n';
						//cerr << " lowerLeft=" << subComposite.getXLowerLeft() << ',' << subComposite.getYLowerLeft() << '\n';
						//cerr << " upperLeft=" << subComposite.getXUpperRight() << ',' << subComposite.getYUpperRight() << '\n';
					}

					if (currentPolarImage->getAlphaChannelCount() > 0) {
						subComposite.addPolar(*currentPolarImage,currentPolarImage->getAlphaChannel(),lon, lat);
					} else {
						subComposite.addPolar(*currentPolarImage,*currentPolarImage, lon, lat);
					}

					if (drain::Debug > 2)
						cerr << " Extracting subcomposite: " << value << '\n';

					subComposite.extractTo(cartesianProduct, value);

					drain::Options &p = cartesianProduct.properties;
					setCompositeAttributesODIM(subComposite,p);
					p["/what/object"] = currentPolarImage->properties["/what/object"];
					p["/what/product"] = currentPolarImage->properties["/what/product"];
					p["/what/prodpar"] = currentPolarImage->properties["/what/prodpar"];
					p["/what/gain"] = 0.0;  // TODO
					p["/what/offset"] = 0.0; // TODO
					p["cTileOffset"] = options["cTileOffset"];
					p["cTileCreate"] = value;
					//cartesianProduct.properties[""] = "";
					p["cBBox"] = options["cBBox"];
					p["cSize"] = options["cSize"];
					p["andre"] = currentPolarImage->properties["andre"];


					if (drain::Debug > 2){
						cartesianProduct.debug();
					}

					//cartesianProduct.setGeometry(subComposite.getWidth(),subComposite.getHeight(),1,1);
					//subComposite.extractTo(tile,"d");
					//subComposite.extractTo(tileAlpha,"w");
					if (!options["cInterpolation"].empty()){
						cartesianProduct.properties["cInterpolation"] = options["cInterpolation"];
						Image<> & tile = cartesianProduct.getChannel(0);
						Image<> & tileAlpha = cartesianProduct.getChannel(1);
						const vector<drain::Data> & v = options["cInterpolation"].getVector();
						if (v[0][0] == 'd') {  // == "dtf"
							drain::image::DistanceTransformFillExponentialOp<> op;
							// Auto mode
							switch (v.size()) {
							case 3:
								op.parameters["vert"] = v[2];
							case 2:
								op.parameters["horz"] = v[1];
								if (v[1] != "0")
									break;
							case 1:
								op.parameters["horz"] = 0.5 * subComposite.getDxMax();
								op.parameters["vert"] = 0.5 * subComposite.getDyMax();
								break;
							default:
								break;  // eror
							}
							cerr << op.parameters << '\n';
							op.filter(tile, tileAlpha, tile, tileAlpha);
						}
						else {
							throw runtime_error("cInterpolation: unsupported method");
						}
						//drain::image::RecursiveRepairerOp<double,double> op2;
					}; // end interpolation
					currentImage = &cartesianProduct;
					currentView.viewImage(cartesianProduct);
				}
				else if (key == "cClose") {
					composite.setGeometry(1, 1); // TODO .reset();
				}
				else if (key == "cExtract") {
					composite.extractTo(cartesianProduct, value);
					//cartesianProduct.properties["ODIM_PATH"] = "/data";
					drain::Options &p = cartesianProduct.properties;

					setCompositeAttributesODIM(composite,p);

					p["/dataset1/what/product"] = polarProduct.properties["/dataset1/what/product"];
					p["productParameter"] = value;
					p["cMethod"] =  options["cMethod"];
					p["sources"] = composite.properties["sources"];

					currentImage = &cartesianProduct;
					currentView.viewImage(*currentImage);


				}
				else if (key == "cInterpolation") {
				}
				else if (key == "cClear") {
					composite.clear();
				}
				else if (key == "cLoad") {   // FULL COMBO BASE
					try {
						//inputOk = false;
						vector<drain::Data> cFade = options["cFade"].getVector();
						cFade.resize(3);
						float fade = cFade[0];
						if (fade == 0.0)
							fade = 0.9;
						int dx = cFade[1];
						int dy = cFade[2];

						drain::image::File::read(cartesianProduct, value);
						composite.setGeometry(cartesianProduct.getWidth(),cartesianProduct.getHeight());
						composite.addImage(cartesianProduct.getChannel(0),cartesianProduct.getAlphaChannel(),
								fade,dx,dy);
						currentImage = &cartesianProduct;
						currentView.viewImage(cartesianProduct);

						//inputOk = true;
					}
					catch (exception& e) {
						cerr << e.what() << '\n';
					}
				}
				else if (key == "cLoadTile") {
					try {
						inputOk = false;
						cartesianProduct.reset();
						drain::image::File::read(cartesianProduct, value);
						if (cartesianProduct.properties.hasKey("cTileOffset")){
							options["cTileOffset"] = cartesianProduct.properties["cTileOffset"];
						}
						else {
							// if !isSet cTileOffset
						}
						currentImage = &cartesianProduct;
						currentView.viewImage(cartesianProduct);
						inputOk = true;
					}
					catch (exception& e) {
						cerr << e.what() << '\n';
					}
				}
				else if (key == "cMethod") {
					if (values.size() > 1)
						composite.setMethod(values[0], valuesF[1], valuesF[2]);
					else
						composite.setMethod(values[0]);
				}
				else if (key == "cSize") {
					composite.setGeometry(values[0], values[1]);
				}
				// deprecen
				else if (key == "cTileConfOut") {
					// Todo range?
					SubComposite subComposite(composite);
					// options["ORIGIN"] = currentPolarImage->properties.get("ORIGIN",options["ORIGIN"]);
					//const vector<float> & origin = options["ORIGIN"].getFloatVector();
					const vector<drain::Data> & origin = currentPolarImage->properties.get("ORIGIN").getVector();
					//cerr << "*** originD " << originD << '\n';
					//vector<float> origin;
					//originD.splitTo(origin);
					subComposite.setRadarLocation(origin[0], origin[1], 250000);
					subComposite.detectBoundingBox();

					ofstream ofstr;
					const float R2D = 180.0/M_PI;
					const string & site = options["SITE"];
					ofstr.open(value.c_str(),ios::out);
					ofstr << "cSize=" << composite.getWidth() << ',' << composite.getHeight() << '\n';
					ofstr << "cBBox=" << R2D*composite.getXLowerLeft() << ',' << R2D*composite.getYUpperRight() << ','
							<< R2D*composite.getXUpperRight() << ',' << R2D*composite.getYLowerLeft() << '\n';
					ofstr << "SITE=" << site << '\n';
					ofstr << "cTileOffset_"<< site <<"=" << subComposite.getXOffset() << ',' << subComposite.getYOffset() << '\n';
					ofstr << "ORIGIN_"<< site <<"=" <<  origin[0] << ',' << origin[1] << '\n';
					ofstr << "SIZE_"<< site <<"=" << subComposite.getWidth() << ',' << subComposite.getHeight() << '\n';
					// TODO +/- sign

				}
				else if (key == "cTileOffset") {
					//
				}
				else if (key == "dumpMap") {
					string filename = "";
					string filter = "*";
					switch (values.size()) {
					case 2:
						filename = values[1];
					case 1:
						filter = values[0];
						break;
					default:
						cerr << options[key].syntax << '\n';
						exit(-1);
						break;
					}
					// TODO convert.replace
					drain::RegExp r(filter);
					ostream *ostr = &cout;
					ofstream ofstr;
					if (filename != ""){
						//string filename = options.get("outputFile","out")+extension;
						cout << "opening " << filename << '\n';
						ofstr.open(filename.c_str(),ios::out);
						ostr = & ofstr;
					}
					for (map<string, drain::Data>::const_iterator it =
							options.begin(); it != options.end(); it++) {
						if (r.test((const string &)it->first)){
							*ostr << it->first << '=' << it->second << '\n';
						}
					}
					ofstr.close();
				}

				//else if (key == "cSubimage") {
				//}
				/*
				 else if (key == "composite.add2"){
				 composite.addPolar(*currentPolarImage,*currentPolarImage,origin[0],origin[1]);
				 }
				 */
				// SELECTION
				else if (key == "view") {
					currentView.properties = currentImage->properties;

					const char c = value.at(0);
					switch (c) {
					case 'F':
						currentView.viewImageFlat(*currentImage);
						cerr << "viewImageFlat " << currentView.getGeometry()
																		<< endl;
						break;
					case 'f': // full image
						currentView.viewImage(*currentImage);
						//currentView.properties["TEST"] = 1;

						break;
					case 'i': // image channels (ie. excluding alpha)
						currentView.viewChannels(
								*currentImage,
								currentImage->getGeometry().getImageChannelCount());
						break;
					case 'a':
						currentView.viewImage(currentImage->getAlphaChannel());
						break;
					default:
						currentView.viewChannel(*currentImage,
								(unsigned int) value);
						break;
					}

				} else if (key == "palette") {
					PaletteOp<> op(value);
					op.filter(currentView, colourProduct);
					currentImage = &colourProduct;
					currentView.viewImage(colourProduct);
				}
				// INPUT/OUTOUT
				else if (key == "outputFile") {
					cerr << "Writing to file not supported by RACK" << endl;
					return -1;
				}
				else if (key == "inputAlpha") {
					// Input alpha
					//cout << "Input alpha\n";
					/*
					 drain::image::BufferedImage<> inputImage2;
					 drain::image::File::read(inputImage2,value);
					 */
					//cout << "pick alpha \n" << flush;
					if (currentImage->getAlphaChannelCount() == 0)
						currentImage->setAlphaChannelCount(1);
					drain::image::Image<> &alpha =
							currentImage->getAlphaChannel();
					//cout << "alpha there\n" << flush;
					//drain::image::File::write(*currentImage,"alphaed.png");
					cout << "alpha here\n" << flush;
					drain::image::File::read(alpha, value);
					// currentImage intact...
					// todo currentAlpha =
				}
				// GENERAL IMAGE PROCESSING
				// To be
				else if (key == "copy") {
					CopyOp<> (value).filter(*currentImage, *currentImage);
					currentView.viewImage(*currentImage); //
				}
				// RADAR
				else if (key == "cartesian") { // TODO == "aeqd"
					drain::radar::PolarToCartesian<> op;
					/*
					if (!inputVolumeConverted){  // = raw data, not a polar product
						convertVolumeToImage(inputVolume,options["h5Data"],inputImage);
						currentPolarImage = &inputImage;
					}
					*/
					op.filter(*currentPolarImage, cartesianProduct);
					currentImage = &cartesianProduct;
					currentView.viewImage(*currentImage);
				}
				else if (key == "debug"){
					drain::Debug = value;
				}
				else if (key == "SITE") {
					cerr << "Setting SITE=" << value << '\n';
					if (localConf.hasKey(value)) {
						// currentPolarImage->properties["ORIGIN"] = locations[value];
						// currentPolarImage->properties["_debug"] = "test";
						vector<double> origin;
						localConf[value].splitTo(origin);
						if (origin.size()==2){
							options["/where/lat"] = origin[0];
							options["/where/lon"] = origin[1];
						}
						else {
							// throw invalid_argument(" ORIGIN needs exactly two arguments <lat>,<lon>.");
						}
						//options["ORIGIN"] = locations[value];
					}
					//cerr << "Now ORIGIN=" << options["ORIGIN"] << '\n';
					//cerr << "Now ORIGIN=" << currentPolarImage->properties["ORIGIN"] << '\n';
				}
				else if (key == "ORIGIN") {
					vector<double> origin;
					value.splitTo(origin);
					if (origin.size()==2){
						options["/where/lat"] = origin[0];
						options["/where/lon"] = origin[1];
					}
					else {
						throw invalid_argument(" ORIGIN needs exactly two arguments <lat>,<lon>.");
					}
				}
				else if (key == "detectBBox"){

					drain::radar::Coordinates coordinates;
					coordinates.setOriginDeg(options["/where/lat"],options["/where/lon"]);

					const float range = options["RANGE"];

					double latMin,lonMin,latMax,lonMax;
					coordinates.getBoundingBox(range,latMin,lonMin,latMax,lonMax);

					options["BBoxLonMin"] = lonMin;
					options["BBoxLatMin"] = latMin;
					options["BBoxLonMax"] = lonMax;
					options["BBoxLatMax"] = latMax;
				}
				else if (key == "version") {
					cout << __RACK__ << endl;
					cout << IMAGE_H_ << endl;
					cout << " Magick++ support: ";
#ifdef DRAIN_MAGICK_yes
					cout << "yes\n"; //MAGICKCORE_PACKAGE_VERSION << '\n';
#else
					cout << "(no)\n";
#endif

					//cerr << options["version"] << '\n';
					// TODO drain Handler version
				}
				else if (key == "format") {

				}
				else if (key == "formatFile") {
					stringstream sstr;
					ifstream ifstr;
					ifstr.open(value.c_str(),ios::in);
					for (int c = ifstr.get(); !ifstr.eof(); c = ifstr.get()){
						sstr << (char)c;
					}
					ifstr.close();
					options["format"] = sstr.str();

				}
				else if (key == "formatOut") {
					options["RANGE"] = options.get("@what/rscale",500) * options.get("@what/nbins",500) ;
					drain::StringMapper<drain::Data> strm(options.getMap(),"@a-zA-Z0-9_/");
					strm.parse(options["format"]);
					if ((value == "")||(value == "-"))
						cout << strm << "\n";
					else {
						ofstream ofstr;
						ofstr.open(value.c_str(),ios::out);
						ofstr << strm;
						ofstr.close();
					}
				}
				// Finally, try drainHandler and rackHandler
				// (Andre handlers are above)
				else {
					int result = 0;

					if (drainHandler.hasOperator(key)) {
						//cout << "drainHandler OK \n";
						if (drain::Debug > 0)
							cout << "drainHandler found for " << key << "\n";
						result = drainHandler.executeOperator(key, value,
								currentView, currentView);
					} else if (polarProductHandler.hasOperator(key)) {
						if (drain::Debug > 0)
							cout << "rackHandler found for " << key << "\n";
						polarProduct.properties.clear();
						polarProduct.properties = currentImage->properties;
						result = polarProductHandler.executeOperator(key, value,
								*currentImage, polarProduct);
						//op.convert(*currentImage,polarProduct);
						currentImage = &polarProduct;
						currentPolarImage = &polarProduct;
						currentView.viewImage(*currentImage);
					} else {
						if (drain::Debug > 0)
							cout << "Notice. Undefined variable or command.\n";
					}
					// } CATCH
					if (result != 0)
						if (!options.hasKey(key)) {
							cerr << "Error: unrecognized option '" << key
									<< "'" << endl;
							exit(-1);
						}
				}

			}
			//cout << "End  of main loop" << endl;

		} // End of main loop
	}
	catch (string s) {
		cerr << s << endl;
	}
	catch (const char *s) {
		cerr << s << endl;
	}
	catch (runtime_error &e) {
		cerr << e.what() << endl;
	}
	catch (exception &e) {
		cerr << e.what() << endl;
	}
	catch (...) {
		cerr << "Error: in handling option '" << key << "'" << endl;
		exit(-1);
	}
	return 0;
}

void Rack::setCompositeAttributesODIM(const Composite &composite,drain::Options &p){
	const long xsize = composite.getWidth();
	const long ysize = composite.getHeight();
	const double LL_lon = composite.getXLowerLeft();
	const double LL_lat = composite.getYUpperRight();
	const double UR_lon = composite.getXUpperRight();
	const double UR_lat = composite.getYLowerLeft();

	//p.clear();
	p["/what/object"] = "COMP";

	stringstream sstr;
	sstr << "+proj=longlat  +R=" << EARTH_RADIUS << " +no_defs";
	p["/dataset1/where/projdef"] = sstr.str();
	//"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs";

	//string prefix="/dataset1";
	p["/where/xsize"] = xsize;
	p["/where/ysize"] = ysize;

	// xscale is the m/pixel resolution at the centerpoint of the image (hence cos).
	p["/where/xscale"] =
			cos((UR_lat+LL_lat)/2.0)*(UR_lon-LL_lon)*EARTH_RADIUS / static_cast<double>(xsize);
	p["/where/yscale"] = (UR_lat-LL_lat)*EARTH_RADIUS / static_cast<double>(ysize);

	p["/where/LL_lon"] = LL_lon * RAD_TO_DEG;
	p["/where/LL_lat"] = LL_lat * RAD_TO_DEG;
	p["/where/UR_lon"] = UR_lon * RAD_TO_DEG;
	p["/where/UR_lat"] = UR_lat * RAD_TO_DEG;
}

} //end namspace rack
