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

#include "RaveConvert.h"

#include <iostream>
#include <map>

using namespace std;

using namespace drain::image;

#include <drain/image/CatenatorOp.h>
#include <drain/util/Debug.h>

#include "DataScaleOp.h"

extern "C"
{
#include <polarvolume.h>
#include <rave_debug.h>
}

namespace rack
{

void RaveConvert::getParameterData(PolarScan_t* scan, drain::image::Image<>& image, drain::Options& options)
{
	PolarScanParam_t* param = PolarScan_getParameter(scan, ((const string &)options["defaultQuantity"]).c_str());
	unsigned char* data = (unsigned char*)PolarScanParam_getData(param);
	RAVE_ASSERT(data != 0, "Failed to get data from PolarScanParam");
	long nBins = PolarScan_getNbins(scan);
	long nRays = PolarScan_getNrays(scan);

	if (drain::Debug > 0) {
		cout << "nbins: " << nBins << " nRays: " << nRays << " totalSize: " << nBins*nRays << endl;
	}

	drain::image::Image<unsigned char> tmp;
	vector<unsigned char> & vec = tmp.getBuffer();
	vec.resize (nBins*nRays);
	std::copy (data, data + nBins*nRays, vec.begin());
	tmp.setGeometry(nBins,nRays);
	RAVE_OBJECT_RELEASE(param);

	//Set gain etc
	//TODO, this won't work unless the parameter chosen as CLI arg is /dataset1/data1
	double gain = image.properties["@what/gain"].getVector().front();
	double offset = image.properties["@what/offset"].getVector().front();
	double noData = image.properties["@what/nodata"].getVector().front();

	double gainInt = options["gainInt"];
	double offsetInt = options["offsetInt"];

	// y = ax + b   orig data x   (a,b in file)
	// y = AX + B   mapped data X (local A,B)
	// X = (ax + b -B)/A  = (a/A)x + (b-B)/A
	const float gain2 = gain/gainInt;
	const float offset2 = (offset-offsetInt)/gainInt;
	drain::radar::DataScaleOp<unsigned char,unsigned char> scale;
	scale.setScale(gain2,offset2);
	scale.setNoDataCode(noData);
	scale.filter(tmp,tmp);
	drain::image::CatenatorOp<unsigned char,unsigned char> cat("vert");
	cat.filter(tmp,image);

	image.properties["gain2"] << gain2;
	image.properties["offset2"] << offset2;
	image.properties["gainInt"] << gainInt;
	image.properties["offsetInt"] << offsetInt;

}

void RaveConvert::getAttributes(drain::image::Image<>& image, RaveObjectList_t* attrs, string scanIndex, string paramIndex)
{
	int numAttributes = RaveObjectList_size(attrs);
	RaveAttribute_t* attr;
	string mainPath = "";
	if (scanIndex != "")
	{
		mainPath = "/dataset"+scanIndex;
		if (paramIndex != "")
		{
			mainPath = mainPath+"/data"+paramIndex;
		}
	}

	for (int i = 0; i < numAttributes; i++)
	{
		attr = (RaveAttribute_t*)RaveObjectList_get(attrs, i);

		if (attr != NULL)
		{
			const char* name = RaveAttribute_getName(attr);
			std::ostringstream ostr;
			ostr << mainPath << "/" << name;
			string path = ostr.str();
			ostr.str("");
			if (drain::Debug > 0) {
				cout << "\tattribute " << name << ": ";
			}
			RaveAttribute_Format format = RaveAttribute_getFormat(attr);
			switch (format)
			{
			case RaveAttribute_Format_String:
			{
				char* value = NULL;
				RaveAttribute_getString(attr, &value);
				if (drain::Debug > 0) {
					cout << value << endl;
				}
				ostr << value;
				break;
			}
			case RaveAttribute_Format_Long:
			{
				long value;
				RaveAttribute_getLong(attr, &value);
				if (drain::Debug > 0) {
					cout << value << endl;
				}
				ostr << value;
				break;
			}
			case RaveAttribute_Format_Double:
			{
				double value;
				RaveAttribute_getDouble(attr, &value);
				if (drain::Debug > 0) {
					cout << value << endl;
				}
				ostr << value;
				break;
			}
			default:
				if (drain::Debug > 0) {
					cout << "undefined";
				}				
			}
			image.properties[path] = ostr.str();
		}
		RAVE_OBJECT_RELEASE(attr);
	}
}



void RaveConvert::getQualityFields(RaveObjectList_t* fields)
{
	int numQualityFields = RaveObjectList_size(fields);
	RaveField_t* field;
	RaveValueType type;
	double value;

	if (drain::Debug > 1) {
		for (int i = 0; i < numQualityFields; i++)
		{
			field = (RaveField_t*)RaveObjectList_get(fields, i);
			if (drain::Debug > 0) {
				cout << "Quality field " << i << endl;
			}

			if (field != NULL)
			{
				long xSize = RaveField_getXsize(field);
				long ySize = RaveField_getYsize(field);
				for (int x = 0; x < xSize; x++)
				{
					for (int y = 0; y < ySize; y++)
					{
						type = RaveField_getValue(field, x, y, &value);
						if (type == RaveValueType_DATA)
						{
							cout << value << ",";
						}
						else
						{
							cout << "x,";
						}
					}
				}

			}
			RAVE_OBJECT_RELEASE(field);
		}
	}
}


bool RaveConvert::getPolarScanParam(drain::image::Image<>& image, PolarScanParam_t* param, string scanIndex, string paramIndex)
{
	std::ostringstream ostr;
	bool passed = true;
	if (drain::Debug > 0) {
		cout << "Reading PolarScanParam " << PolarScanParam_getQuantity(param) << endl;
	}


	double gain = PolarScanParam_getGain(param);
	double offset = PolarScanParam_getOffset(param);
	double noData = PolarScanParam_getNodata(param);
	double undetect = PolarScanParam_getUndetect(param);

	string path = "/dataset"+scanIndex+"/data"+paramIndex;

	const string dataset = path+"/data";
	image.properties["datasets"] << dataset;

	image.properties[path+"/what/quantity"] = PolarScanParam_getQuantity(param);
	image.properties["@what/quantity"] << string(PolarScanParam_getQuantity(param));
	ostr.str(""); ostr << PolarScanParam_getGain(param);
	image.properties[path+"/what/gain"] = ostr.str();
	image.properties["@what/gain"] << ostr.str();
	ostr.str(""); ostr << PolarScanParam_getOffset(param);
	image.properties[path+"/what/offset"] = ostr.str();
	image.properties["@what/offset"] << ostr.str();
	ostr.str(""); ostr << PolarScanParam_getNodata(param);
	image.properties[path+"/what/nodata"] = ostr.str();
	image.properties["@what/nodata"] << ostr.str();
	ostr.str(""); ostr << PolarScanParam_getUndetect(param);
	image.properties[path+"/what/undetect"] = ostr.str();
	image.properties["@what/undetect"] << ostr.str();

	if (drain::Debug > 0) {
		cout << "\tgain: " << gain << " offset: " << offset << " noData: " << noData << " undetect: " << undetect << endl;
	}	

	long nBins = PolarScanParam_getNbins(param);
	long nRays = PolarScanParam_getNrays(param);

	if (drain::Debug > 0) {
		cout << "\tnBins: " << nBins << " nRays: " << nRays << endl;
	}

	if (drain::Debug > 2) {
		RaveValueType type;
		double value = 0;
		for (int ray = 0; ray < nRays; ray++)
		{
			cout << "--ray: " << ray << endl;
			for (int bin = 0; bin < nBins; bin++)
			{
				type = PolarScanParam_getValue(param, bin, ray, &value);
				if (type == RaveValueType_DATA)
				{
					cout << value << ",";
				}
				else
				{
					cout << "x,";
				}
			}
			cout << endl;

		}
	}

	//Polar scan param attributes
	if (drain::Debug > 0) {
		cout << "Polar scan param attributes:" << endl;
	}
	RaveObjectList_t* attrs = PolarScanParam_getAttributeValues(param);
	getAttributes(image, attrs, scanIndex, paramIndex);
	RAVE_OBJECT_RELEASE(attrs);

	//Polar scan param quality fields
	if (drain::Debug > 0) {
		cout << "Polar scan param quality fields:" << endl;
	}
	RaveObjectList_t* fields = PolarScanParam_getQualityFields(param);
	getQualityFields(fields);
	RAVE_OBJECT_RELEASE(fields);

	return passed;
}

bool RaveConvert::getPolarScan(drain::image::Image<>& image, PolarScan_t* scan, int scanIndex, string quantity)
{
	if (drain::Debug > 0) {
		cout << "getPolarScan" << endl;	
	}

	std::ostringstream ostr;
	ostr << scanIndex;
	std::string index = ostr.str();

	RAVE_ASSERT((scan != NULL), "scan == NULL");
	if (drain::Debug > 0) {
		cout << "Reading polar scan: " << "/dataset"+index << endl;
	}

	bool passed = true;
	image.properties["/dataset"+index+"/what/source"] = PolarScan_getSource(scan);
	image.properties["@what/source"] << string(PolarScan_getSource(scan));
	image.properties["/dataset"+index+"/what/date"] = PolarScan_getDate(scan);
	image.properties["@what/date"] << string(PolarScan_getDate(scan));
	image.properties["/dataset"+index+"/what/time"] = PolarScan_getTime(scan);
	image.properties["@what/time"] << string(PolarScan_getTime(scan));

	ostr.str(""); ostr << PolarScan_getLongitude(scan)*RAD_TO_DEG;
	image.properties["/dataset"+index+"/where/lon"] = PolarScan_getLongitude(scan)*RAD_TO_DEG;
	image.properties["@where/lon"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getLatitude(scan)*RAD_TO_DEG;
	image.properties["/dataset"+index+"/where/lat"] = PolarScan_getLatitude(scan)*RAD_TO_DEG;
	image.properties["@where/lat"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getHeight(scan);
	image.properties["/dataset"+index+"/where/height"] = PolarScan_getHeight(scan);
	image.properties["@where/height"] << ostr.str();



	image.properties["/dataset"+index+"/what/starttime"] = PolarScan_getStartTime(scan);
	image.properties["@what/starttime"] << string(PolarScan_getStartTime(scan));

	image.properties["/dataset"+index+"/what/endtime"] = PolarScan_getEndTime(scan);
	image.properties["@what/endtime"] << string(PolarScan_getEndTime(scan));

	image.properties["/dataset"+index+"/what/startdate"] = PolarScan_getStartDate(scan);
	image.properties["@what/startdate"] << string(PolarScan_getStartDate(scan));
	image.properties["/dataset"+index+"/data1/what/enddate"] = PolarScan_getEndDate(scan);
	image.properties["@what/enddate"] << string(PolarScan_getEndDate(scan));

	ostr.str(""); ostr << PolarScan_getElangle(scan)*RAD_TO_DEG;
	image.properties["/dataset"+index+"/where/elangle"] = PolarScan_getElangle(scan)*RAD_TO_DEG;
	image.properties["@where/elangle"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getNbins(scan);
	image.properties["/dataset"+index+"/where/nbins"] = PolarScan_getNbins(scan);
	image.properties["@where/nbins"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getNrays(scan);
	image.properties["/dataset"+index+"/where/nrays"] = PolarScan_getNrays(scan);
	image.properties["@where/nrays"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getRscale(scan);
	image.properties["/dataset"+index+"/where/rscale"] = PolarScan_getRscale(scan);
	image.properties["@where/rscale"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getRstart(scan);
	image.properties["/dataset"+index+"/where/rstart"] = PolarScan_getRstart(scan);
	image.properties["@where/rstart"] << ostr.str();
	ostr.str(""); ostr << PolarScan_getA1gate(scan);
	image.properties["/dataset"+index+"/where/a1gate"] = PolarScan_getA1gate(scan);
	image.properties["@where/a1gate"] << ostr.str();

	long nBins = PolarScan_getNbins(scan);
	long nRays = PolarScan_getNrays(scan);
	double rScale = PolarScan_getRscale(scan);
	double rStart = PolarScan_getRstart(scan);

	if (drain::Debug > 0) {
		cout << "\tnBins: " << nBins << " nRays: " << nRays << " rScale: " << rScale << " rStart: " << rStart << endl;
	}

	//Polar scan params
	RaveObjectList_t* params = PolarScan_getParameters(scan);
	int numParamValues = RaveObjectList_size(params);
	int numQualityFields = PolarScan_getNumberOfQualityFields(scan);

	if (drain::Debug > 0) {
		cout << "\tnumParams: " << numParamValues << " numQualityFields: " << numQualityFields << endl;
	}

	for (int i = 0; i < numParamValues; i++)
	{
		PolarScanParam_t* param = (PolarScanParam_t*)RaveObjectList_get(params, i);
		if (param != NULL && string(PolarScanParam_getQuantity(param)) == quantity) //Rack only uses the quantity param
		{
			std::ostringstream ostr2;
			ostr2 << i+1;
			std::string paramIndex = ostr2.str();
			getPolarScanParam(image, param, index, "1");// paramIndex); //TODO, use when rack supports several parameters
			RAVE_OBJECT_RELEASE(param);
		}
		else //store the rest of the parameters until rack is finished
		{
			paramStorage.push_back(param);
		}
	}
	RAVE_OBJECT_RELEASE(params);

	//Polar scan attributes
	if (drain::Debug > 0) {
		cout << "Polar scan attributes:" << endl;
	}	
	RaveObjectList_t* attrs = PolarScan_getAttributeValues(scan);
	getAttributes(image, attrs, index);
	RAVE_OBJECT_RELEASE(attrs);

	//Polar scan quality fields
	if (drain::Debug > 0) {
		cout << "Polar scan quality fields:" << endl;
	}	
	RaveObjectList_t* fields = PolarScan_getQualityFields(scan);
	getQualityFields(fields);
	RAVE_OBJECT_RELEASE(fields);

	return passed;
}

bool RaveConvert::getPolarVolume(drain::image::Image<>& image, PolarVolume_t* volume)
{	
	int numScans = PolarVolume_getNumberOfScans(volume);
	bool passed = true;
	
	if (drain::Debug > 0) {
		cout << "Number of scans: " << numScans << endl;
		cout << "Polar volume params:" << endl;

		//The following data is handled in the PolarScan instead since the HDF5 file
		//may contain just a SCAN and no PVOL.
		cout << "\t time: " << PolarVolume_getTime(volume) << endl;
		cout << "\t date: " << PolarVolume_getDate(volume) << endl;
		cout << "\t source: " << PolarVolume_getSource(volume) << endl;
		cout << "\t lon: " << PolarVolume_getLongitude(volume) << endl;
		cout << "\t lat: " << PolarVolume_getLatitude(volume) << endl;
		cout << "\t height: " << PolarVolume_getHeight(volume) << endl;
	}

	//Polar volume attributes
	if (drain::Debug > 0) {		
		cout << "Polar volume attributes:" << endl;
	}
	RaveObjectList_t* attrs = PolarVolume_getAttributeValues(volume);
	getAttributes(image, attrs);
	RAVE_OBJECT_RELEASE(attrs);

	return passed;
}

bool RaveConvert::convertToRack(PolarVolume_t* volume, drain::image::Image<>& image, drain::Options& options)
{
	bool passed = true;
	PolarScan_t* scan = NULL;
	if (drain::Debug > 0) {		
		cout << "Reading polar volume." << endl;
	}	

	getPolarVolume(image, volume); //Get all params/data except for the polar scans

	int numScans = PolarVolume_getNumberOfScans(volume);
	for (int i = 0; i < numScans; i++)
	{
		scan = PolarVolume_getScan(volume, i);
		passed &= getPolarScan(image, scan, i+1, ((const string &)options["defaultQuantity"]).c_str());
		getParameterData(scan, image, options);
		RAVE_OBJECT_RELEASE(scan);
	}

	return passed;
}

bool RaveConvert::convertToRack(PolarScan_t* scan, drain::image::Image<>& image, drain::Options& options)
{
	bool passed = getPolarScan(image, scan, 1, ((const string &)options["defaultQuantity"]).c_str());
	getParameterData(scan, image, options);
	return passed;
}

bool RaveConvert::addPolarScanParamAttribute(string scanIndex, string paramIndex, string node, string attribute, const drain::Data &value)
{
	bool passed = true;
	if (drain::Debug > 0) {		
		cout << "\t\taddPolarScanParamAttribute" << endl;
	}
	
	const type_info &t = value.getType();

	RAVE_ASSERT((scanMap[scanIndex] != NULL), "Failed to get scan");
	PolarScanParam_t* param = scanMap[scanIndex]->getParam();
	if (param == NULL)
	{
		param = (PolarScanParam_t*)RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
		scanMap[scanIndex]->setParam(param);
	}

	istringstream iss(value);
	double val;
	iss >> val;

	if (attribute == "gain")
	{
		PolarScanParam_setGain(param, val);
	}
	else if (attribute == "nodata")
	{
		 PolarScanParam_setNodata(param, val);
	}
	else if (attribute == "offset")
	{
		 PolarScanParam_setOffset(param, val);
	}
	else if (attribute == "undetect")
	{
		 PolarScanParam_setUndetect(param, val);
	}
	else if (attribute == "quantity")
	{
		 PolarScanParam_setQuantity(param, ((const string &)value).c_str());
	}
	else if (t == typeid(string) || t == typeid(char *) || !value.typeIsSet())
	{
		size_t pos = node.rfind('/');
		string attrib = node.substr(pos+1)+"/"+attribute;
		if (drain::Debug > 0) {		
			cout << "Adding attrib " << attrib << "=" << (const string &)value << " to PolarScanParam" << endl;
		}	
		RaveAttribute_t* attr = RaveAttributeHelp_createString(attrib.c_str(), ((const string &)value).c_str());
		RAVE_ASSERT((attr != NULL), "Failed to create attribute");
		int ok = PolarScanParam_addAttribute(param, attr);
		passed &= (ok != 0);
		RAVE_ASSERT(ok != 0, "Failed to add attribute to param");
		RAVE_OBJECT_RELEASE(attr);
	}
	else
	{
		if (drain::Debug > 0) {		
			cout << "<--PolarScanParam attribute (param)" << attribute << " not handled!" << endl;
		}
	}
	return passed;
}

bool RaveConvert::addPolarScanAttribute(string scanIndex, string node, string attribute, const drain::Data &value)
{
	bool passed = true;
	const type_info &t = value.getType();

	PolarScan_t* scan = scanMap[scanIndex]->getScan();


	if (attribute == "nbins" || attribute == "nrays")
	{
		//These are set inside RAVE HDF5 drivers so they can be ignored here.
	}
	else if (attribute == "date")
	{
		passed = PolarScan_setDate(scan, ((const string &)value).c_str());
	}
	else if (attribute == "time")
	{
		passed = PolarScan_setTime(scan, ((const string &)value).c_str());
	}
	else if (attribute == "enddate")
	{
		passed = PolarScan_setEndDate(scan, ((const string &)value).c_str());
	}
	else if (attribute == "endtime")
	{
		passed = PolarScan_setEndTime(scan, ((const string &)value).c_str());
	}
	else if (attribute == "startdate")
	{
		passed = PolarScan_setStartDate(scan, ((const string &)value).c_str());
	}
	else if (attribute == "starttime")
	{
		passed = PolarScan_setStartTime(scan, ((const string &)value).c_str());
	}
	else if (attribute == "source")
	{
		passed = PolarScan_setSource(scan, ((const string &)value).c_str());
	}
	else if (attribute == "a1gate")
	{
		vector<long> values;
		value.splitTo(values);
		PolarScan_setA1gate(scan, values[0]);
	}
	else if (attribute == "elangle")
	{
		vector<double> values;
		value.splitTo(values);
		PolarScan_setElangle(scan, values[0]*DEG_TO_RAD);
	}
	else if (attribute == "lat")
	{
		vector<double> values;
		value.splitTo(values);
		PolarScan_setLatitude(scan, values[0]*DEG_TO_RAD);
	}
	else if (attribute == "lon")
	{
		vector<double> values;
		value.splitTo(values);
		PolarScan_setLongitude(scan, values[0]*DEG_TO_RAD);
	}
	else if (attribute == "rscale")
	{
		vector<double> values;
		value.splitTo(values);
		PolarScan_setRscale(scan, values[0]);
	}
	else if (attribute == "rstart")
	{
		vector<double> values;
		value.splitTo(values);
		PolarScan_setRstart(scan, values[0]);
	}
	else if (attribute == "height")
	{
		vector<double> values;
		value.splitTo(values);
		PolarScan_setHeight(scan, values[0]);
	}
	else if (t == typeid(string) || t == typeid(char *) || !value.typeIsSet())
	{
		size_t pos = node.rfind('/');
		string attrib = node.substr(pos+1)+"/"+attribute;
		if (drain::Debug > 0) {		
			cout << "Adding attrib " << attrib << "=" << (const string &)value << " to scan" << endl;
		}
		RaveAttribute_t* attr = RaveAttributeHelp_createString(attrib.c_str(), ((const string &)value).c_str());
		RAVE_ASSERT((attr != NULL), "Failed to create attribute");
		int ok = PolarScan_addAttribute(scan, attr);
		passed &= (ok != 0);
		RAVE_ASSERT(ok != 0, "Failed to add attribute to scan");
		RAVE_OBJECT_RELEASE(attr);
	}
	else
	{
		if (drain::Debug > 0) {		
			cout << "<--PolarScan attribute " << attribute << " not handled!" << endl;
		}		
		passed = false;
	}
	return passed;
}

bool RaveConvert::addPolarVolumeAttribute(PolarVolume_t* volume, string node, string attribute, const drain::Data &value)
{
	bool passed = true;
	const type_info &t = value.getType();
	string attrib = node.substr(1)+"/"+attribute;

	if (attribute == "date")
	{
		PolarVolume_setDate(volume, ((const string &)value).c_str());
	}
	else if (attribute == "time")
	{
		PolarVolume_setTime(volume, ((const string &)value).c_str());
	}
	else if (attribute == "source")
	{
		PolarVolume_setSource(volume, ((const string &)value).c_str());
	}
	else if (attribute == "lat")
	{
		PolarVolume_setLatitude(volume, ((const double)value)*DEG_TO_RAD);
	}
	else if (attribute == "lon")
	{
		PolarVolume_setLongitude(volume, ((const double)value)*DEG_TO_RAD);
	}
	else if (attribute == "height")
	{
		PolarVolume_setHeight(volume, ((const double)value));
	}
	else if (attribute == "beamwidth")
	{
		PolarVolume_setBeamwidth(volume, ((const double)value));
	}
	else if (t == typeid(string) || t == typeid(char *) || !value.typeIsSet())
	{
		if (drain::Debug > 0) {		
			cout << "Adding attrib " << attrib << "=" << (const string &)value << " to volume" << endl;
		}
		RaveAttribute_t* attr = RaveAttributeHelp_createString(attrib.c_str(), ((const string &)value).c_str());
		RAVE_ASSERT((attr != NULL), "Failed to create attribute");
		int ok = PolarVolume_addAttribute(volume, attr);

		RAVE_ASSERT(ok != 0, "Failed to add attribute to volume");
		passed &= (ok != 0);
		RAVE_OBJECT_RELEASE(attr);
	}
	else
	{
		cerr << "<-- Couldn't handle " << node << "/" << attribute << endl;
	}
	return passed;
}

void RaveConvert::addScans(drain::image::ImageView<>& view)
{
	if (drain::Debug > 0) {		
		cout << "Adding " << view.getImageChannelCount() << " scans to volume" << endl;
	}	
	for (unsigned int i = 0; i < view.getImageChannelCount(); i++)
	{
		std::ostringstream ostr;
		ostr << i+1;
		PolarScan_t* scan = (PolarScan_t*)RAVE_OBJECT_NEW(&PolarScan_TYPE);
		RAVE_ASSERT((scan != NULL), "Failed to allocate mem for PolarScan");

		ScanData* data = new ScanData();
		data->setScan(scan);
		scanMap[ostr.str()] = data;
	}
}


bool RaveConvert::addAttributes(PolarVolume_t* volume, drain::image::Image<>& image)
{
	bool passed = true;
	const drain::RegExp scanRegExp("^/dataset");
	const drain::RegExp scanQualityRegExp("^/dataset[0-9]*/quality");
	const drain::RegExp paramRegExp("^/dataset[0-9]*/data");
	const drain::RegExp paramQualityRegExp("^/dataset[0-9]*/data[0-9]*/quality");

	for (map<string, drain::Data>::const_iterator it =
			image.properties.begin(); it != image.properties.end(); it++) {

		const string &key = it->first;
		const drain::Data &value = it->second;
		string node;
		string attribute;
		if (key[0] != '/'){
			continue;
		}
		else {
			size_t pos = key.rfind('/');
			node = key.substr(0,pos);
			attribute = key.substr(pos+1);
		}

		if (drain::Debug > 0) {		
			cout << node << "/" << attribute << endl;
		}

		if (scanRegExp.test(node)) //If begins with "/dataset...."
		{
			string scanIndex = getScanIndex(node);

			if (paramRegExp.test(node))
			{
				if (paramQualityRegExp.test(node)) //Begins with "/dataset[0-9]*/data[0-9]*/quality"
				{
					cout << "Not implemented param quality attributes handling yet!" << endl;
				}
				else //Begins with "/dataset[0-9]*/data"
				{
					string paramIndex = getParamIndex(node);
					passed &= addPolarScanParamAttribute(scanIndex, paramIndex, node, attribute, value);
				}
			}
			else if (scanQualityRegExp.test(node))
			{
				cout << "Not implemented scan quality attributes handling yet!" << endl;
			}
			else // Something of type /dataset1/what,where etc
			{
				if (drain::Debug > 0) {		
					cout << "\taddPolarScanAttribute" << endl;
				}
				passed &= addPolarScanAttribute(scanIndex, node, attribute, value);
			}
		}
		else // It is /what, /where, etc
		{
			if (volume != NULL)
			{
				passed &= addPolarVolumeAttribute(volume, node, attribute, value);
			}
			else
			{
				//passed = false;//TODO, should this one be set to false here?
				if (drain::Debug > 0) {		
					cout << "<--" << node << "/" << attribute << " is not set to volume since we're working with a PolarScan" << endl;
				}			
			}
		}
	}

	if (!passed)
	{
		cerr << "RaveConvert::addAttributes failed" << endl;
	}
	return passed;
}

bool RaveConvert::addDataToVolume(PolarVolume_t* volume, drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity)
{
	bool result = true;
	for (map<string, ScanData*>::iterator iter = scanMap.begin(); iter != scanMap.end(); iter++)
	{
		result &= addDataToScan(iter->first, image, view, quantity);
		int ok = PolarVolume_addScan(volume, iter->second->getScan());
		result &= (ok != 0);
		RAVE_ASSERT(ok != 0, "Failed to add scan to volume");
		RAVE_OBJECT_RELEASE(iter->second);
	}
	return result;
}

bool RaveConvert::addDataToScan(string scanIndex, drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity)
{
	bool result = true;

	PolarScan_t* scan = scanMap[scanIndex]->getScan();

	RAVE_ASSERT((scanMap[scanIndex] != NULL), "scanData == NULL");
	PolarScanParam_t* param = scanMap[scanIndex]->getParam(); //Rack can only handle one param per scan
	if(param != NULL) //This is the parameter matching the default quantity
	{
		RAVE_ASSERT((string(PolarScanParam_getQuantity(param)) == quantity), "Wrong parameter processed by RACK!");
		//Create data for all parameters, but only fill it with data for the parameter
		//specified by quantity since we don't have any data for the others.
		long nBins = view.getWidth();
		long nRays = view.getHeight();
		if (drain::Debug > 0) {
			cout << "<---------Using nbins, rays: " << nBins << "," << nRays << " when setting data" << endl;
		}
		PolarScanParam_createData(param, nBins, nRays, RaveDataType_UCHAR); //TODO, add support for other than uchar?

		unsigned char* data = (unsigned char*)PolarScanParam_getData(param);
		vector<unsigned char> & vec = image.getBuffer();
		for (int i = 0; i < nBins*nRays; i++)
		{
			*(data+i) = vec[i];
		}

		int ok = PolarScan_addParameter(scan, param);
		result &= (ok != 0);
		RAVE_ASSERT(ok != 0, "Failed to add param to scan");
		RAVE_OBJECT_RELEASE(param);
	}

	//Add the rest of the parameters
	for (int i = paramStorage.size()-1; i >= 0; i--)
	{
		int ok = PolarScan_addParameter(scan, paramStorage[i]);
		RAVE_OBJECT_RELEASE(paramStorage[i]);
		result &= (ok != 0);
	}

	if (drain::Debug > 0) {		
		cout << "width: " << view.getWidth() << " height: " << view.getHeight() << endl;
	}
	//Add quality
	unsigned int imgChCount = view.getImageChannelCount();
	unsigned int chCount = view.getChannelCount();
	for (unsigned int k = imgChCount; k < chCount; ++k)
	{
		RaveField_t* field = (RaveField_t*)RAVE_OBJECT_NEW(&RaveField_TYPE);
		unsigned char* char_p = &(*view.getChannel(k).begin());
		//std::copy(vec.begin(), vec.end(), a) ;
		int ok = RaveField_setData(field, view.getWidth(), view.getHeight(), char_p, RaveDataType_UCHAR); //TODO, add support for other than uchar?
		result &= (ok != 0);
		ok = PolarScan_addQualityField(scan, field);
		result &= (ok != 0);
		RAVE_OBJECT_RELEASE(field);
	}
	return result;
}

string RaveConvert::getScanIndex(string path)
{
	string indexStr = path.substr(8); //Remove "/dataset" from beginning of string
	//cout << "scan indexStr(1): " << indexStr << endl;
	size_t indexEndPos = indexStr.find_first_of('/'); //Get first pos after last index digit
	indexStr = indexStr.substr(0,indexEndPos);
	//cout << "scan indexStr(2): " << indexStr << endl;
	return indexStr;
}

string RaveConvert::getParamIndex(string path)
{
	//Get pos for first digit in param index.
	// 9 => first char after "/dataset?". In most cases it will be a "/".
	// 5 => first char after "/data" which is the first digit in param index.
	size_t startPos = path.find("/", 9) + 5;
	string indexStr = path.substr(startPos); //Remove "/dataset[0-9]*/data" from beginning of string
	//cout << "param indexStr(1): " << indexStr << endl;
	size_t indexEndPos = indexStr.find_first_of('/'); //Get first pos after last index digit
	indexStr = indexStr.substr(0,indexEndPos);
	//cout << "param indexStr(2): " << indexStr << endl;
	return indexStr;
}

PolarVolume_t* RaveConvert::createPolarVolume(drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity)
{
	PolarVolume_t* volume = (PolarVolume_t*)RAVE_OBJECT_NEW(&PolarVolume_TYPE);
	if (volume == NULL) {
		cerr << "Failed to allocate mem for PolarVolume" << endl;
		return NULL;
	}

	//fill with data
	bool passed = true;
	addScans(view);
	passed &= addAttributes(volume, image);
	passed &= addDataToVolume(volume, image, view, quantity);

	cleanup();
	if (!passed)
	{
		RAVE_OBJECT_RELEASE(volume);
		return NULL;
	}
	return volume;
}

PolarScan_t* RaveConvert::createPolarScan(drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity)
{
	PolarScan_t* scan = NULL;

	//fill with data
	bool passed = true;
	addScans(view);
	passed &= addAttributes(NULL, image);
	if (passed)
	{
		passed &= addDataToScan("1", image, view, quantity);
	}

	scan = scanMap["1"]->getScan();

	cleanup();
	if (!passed)
	{
		RAVE_OBJECT_RELEASE(scan);
		return NULL;
	}
	return scan;
}

void RaveConvert::cleanup()
{
	for (map<string, ScanData*>::iterator iter = scanMap.begin(); iter != scanMap.end(); iter++)
	{
		delete iter->second;
	}	
}


RaveConvert::RaveConvert()
{
}

RaveConvert::~RaveConvert()
{
}

void RaveConvert::copyVolumeAttributes(PolarVolume_t* sourceVolume, PolarVolume_t* destVolume)
{
	PolarVolume_setTime(destVolume, PolarVolume_getTime(sourceVolume));
	PolarVolume_setDate(destVolume, PolarVolume_getDate(sourceVolume));
	PolarVolume_setSource(destVolume, PolarVolume_getSource(sourceVolume));
	PolarVolume_setLongitude(destVolume, PolarVolume_getLongitude(sourceVolume));
	PolarVolume_setLatitude(destVolume, PolarVolume_getLatitude(sourceVolume));
	PolarVolume_setHeight(destVolume, PolarVolume_getHeight(sourceVolume));
	PolarVolume_setBeamwidth(destVolume, PolarVolume_getBeamwidth(sourceVolume));
	
	RaveObjectList_t* attrs = PolarVolume_getAttributeValues(sourceVolume);
	int numAttributes = RaveObjectList_size(attrs);
	RaveAttribute_t* attr;

	for (int i = 0; i < numAttributes; i++)
	{
		attr = (RaveAttribute_t*)RaveObjectList_get(attrs, i);
		PolarVolume_addAttribute(destVolume, attr);
		RAVE_OBJECT_RELEASE(attr);
	}
	RAVE_OBJECT_RELEASE(attrs);
}


} //end namspace rack
