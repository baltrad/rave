/**

    Copyright 2001 - 2010

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
#ifndef __RAVECONVERT__
#define __RAVECONVERT__

#include <drain/image/Image.h>

///// Included for accessing polarvolume and subtypes
extern "C"
{
#include <polarvolume.h>
#include <rave_debug.h>
}
/////

using namespace std;

namespace rack
{

///   Class which converts:
///   * From RAVE types PolarVolume_t/PolarScan_t to RACK internal Image type.
///   * From RACK internal Image type to RAVE types PolarVolume_t/PolarScan_t.
/**
 *
 */
class RaveConvert
{
public:
	RaveConvert();
	virtual ~RaveConvert();

	/*
	 * Creates a PolarVolume_t using data in the Image.
	 * @param[in] image - The image
	 * @param[in] view - The view
	 * @param[in] quantity - The default quantity
	 * @return The PolarVolume_t*
	 */
	PolarVolume_t* createPolarVolume(drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity);

	/*
	 * Creates a PolarScan_t using data in the Image.
	 * @param[in] image - The image
	 * @param[in] view - The view
	 * @param[in] quantity - The default quantity
	 * @return The PolarScan_t*
	 */
	PolarScan_t* createPolarScan(drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity);

	/*
	 * Copies data from PolarScan to Image.
	 * @param[in] scan - The PolarScan_t
	 * @param[out] image - The Image
	 * @param[in] options - Options
	 * @return bool - Whether operation was successful
	 */
	bool convertToRack(PolarScan_t* scan, drain::image::Image<>& image, drain::Options& options);

	/*
	 * Copies data from PolarVolume to Image.
	 * @param[in] volume - The PolarVolume_t
	 * @param[out] image - The Image
	 * @param[in] options - Options
	 * @return bool - Whether operation was successful
	 */
	bool convertToRack(PolarVolume_t* volume, drain::image::Image<>& image, drain::Options& options);

	/*
	 * Copies all attributes from one PolarVolume to another.
	 * @param[in] sourceVolume - The PolarVolume_t to copy from.
	 * @param[in/out] destVolume - The PolarVolume_t to copy to.
	 */
	void copyVolumeAttributes(PolarVolume_t* sourceVolume, PolarVolume_t* destVolume);
    
private:

	/**
	 * Storage class used for temporary storing data when converting from RACK image to RAVE format.
	 */	
	class ScanData
	{
	public:

		ScanData()
		{
			scanParam = NULL;
		}

		void setScan(PolarScan_t* polarScan/*, int index*/)
		{
			//this.index = index;
			scan = polarScan;
		}

		PolarScan_t* getScan()
		{
			return scan;
		}

		void setParam(PolarScanParam_t* param)
		{
			RAVE_ASSERT((scanParam == NULL), "Cannot add two default quantitys per scan"); //RAck can only process one parameter per scan.
			scanParam = param;
		}

		PolarScanParam_t* getParam()
		{
			return scanParam;
		}

		void setScanParamQuality(RaveField_t* quality, string paramIndex, string qualityIndex)
		{
			pair<string,string> key(paramIndex,qualityIndex);
			paramQualityMap[key] = quality;
		}

		RaveField_t* getScanParamQuality(string paramIndex, string qualityIndex)
		{
			pair<string,string> key(paramIndex,qualityIndex);
			return paramQualityMap[key];
		}

		void setScanQuality(RaveField_t* quality, string qualityIndex)
		{
			scanQualityMap[qualityIndex] = quality;
		}

		RaveField_t* getScanQuality(string qualityIndex)
		{
			return scanQualityMap[qualityIndex];
		}

		//Data
		PolarScan_t* scan;
		PolarScanParam_t* scanParam; //The parameter with default quantity, i.e. the one processed by RACK.
		std::map<pair<string,string>, RaveField_t*> paramQualityMap; //Map with params. The key is a pair<sparam index, quality index>.
		std::map<string, RaveField_t*> scanQualityMap; //Map with scan quality. The key is string with quality index.
	};
	
	////////////////////////////////////////////////////////////////
	// Functions used for converting from RAVE format to RACK format
	////////////////////////////////////////////////////////////////

	/**
	 * Fetches the attributes from a list and stores them in the Image.
	 * @param[in/out] image - the image to store the attributes in
	 * @param[in] attr - list of attributes
	 * @param[in] scanIndex - string containing index for a specific scan.
	 * @param[in] paramIndex - string containing index for a specific parameter.
	 */
	void getAttributes(drain::image::Image<>& image, RaveObjectList_t* attrs, string scanIndex = "", string paramIndex = "");

	/**
	 * Fetches the quality fields from a list of field. However, since RACK doesn't store quality fields from an input
	 * file yet, the fields aren't stored but printed at the moment.
	 * @param[in] fields - the list of quality fields
	 */
	void getQualityFields(RaveObjectList_t* fields);
	
	/**
	 * Fetches the attributes from a PolarVolume and stores them in the Image. However, since RACK doesn't fully support
	 * PolarVolumes yet, this function is not used at the moment. 
	 * @param[in/out] image - the image to store the attributes in
	 * @param[in] volume - the polar volume to fetch the attributes from
	 * @return True if passed, false otherwise
	 */
	bool getPolarVolume(drain::image::Image<>& image, PolarVolume_t* volume);

	/**
	 * Fetches the attributes and quality fields from a PolarScanParam and stores them in the Image. However, since RACK
	 * doesn't store quality fields from an input file yet, the fields aren't stored but printed at the moment.
	 * @param[in/out] image - the image to store the attributes in
	 * @param[in] param - the PolarScanParam
	 * @param[in] scanIndex - string containing index for a specific scan.
	 * @param[in] paramIndex - string containing index for a specific parameter.
	 * @return True if passed, false otherwise
	 */
	bool getPolarScanParam(drain::image::Image<>& image, PolarScanParam_t* param, string scanIndex, string paramIndex);

	/**
	 * Fetches the attributes and quality fields from a PolarScan and stores them in the Image. However, since RACK
	 * doesn't store quality fields from an input file yet, the fields aren't stored but printed at the moment. The
	 * attributes for the PolarScanParams in the PolarScan are only stored for the PolarScanParam with the default
	 * quantity. The other PolarScanParams are stored untouched to be copied back when RACK is finished and the
	 * resulting PolarScan shall be produced.
	 * @param[in/out] image - the image to store the attributes in
	 * @param[in] scan - the PolarScan
	 * @param[in] quantity - the default quantity indicating which PolarScanParam RACK shall apply the algorithms upon.
	 * @return True if passed, false otherwise
	 */
	bool getPolarScan(drain::image::Image<>& image, PolarScan_t* scan, int scanIndex, string quantity);

	/**
	 * Fetches the data array for the PolarScanParam with the default quantity and stores it in the Image.
	 * @param[in/out] image - the image to store the attributes in
	 * @param[in] options - storage class containing information needed when converting to Image
	 */
	void getParameterData(PolarScan_t* scan, drain::image::Image<>& image, drain::Options& options);

	////////////////////////////////////////////////////////////////
	// Functions used for converting from RACK format to RAVE format
	////////////////////////////////////////////////////////////////
	
	/**
	 * Add a specific attribute to a PolarScanParam.
	 * @param[in] scanIndex - string indicating in which PolarScan the PolarScanParam is located.
	 * @param[in] paramIndex - string indicating in which PolarScanParam to add the attribute to
	 * @param[in] node - ODIM path where attribute is stored. E.g. "/dataset1/data1/what"
	 * @param[in] attribute - the name of the attribute. E.g. elangle
	 * @param[in] value - storage class containing the value of the attribute
	 * @return True if passed, false otherwise
	 */
	bool addPolarScanParamAttribute(string scanIndex, string paramIndex, string node, string attribute, const drain::Data &value);

	/**
	 * Add a specific attribute to a PolarScan.
	 * @param[in] scanIndex - string indicating in which PolarScan the attribute is located.
	 * @param[in] node - ODIM path where attribute is stored. E.g. "/dataset1/what"
	 * @param[in] attribute - the name of the attribute. E.g. elangle
	 * @param[in] value - storage class containing the value of the attribute
	 * @return True if passed, false otherwise
	 */
	bool addPolarScanAttribute(string scanIndex, string node, string attribute, const drain::Data &value);

	/**
	 * Add a specific attribute to a PolarVolume.
	 * @param[in/out] volume - the volume.
	 * @param[in] node - ODIM path where attribute is stored. E.g. "/what"
	 * @param[in] attribute - the name of the attribute. E.g. elangle
	 * @param[in] value - storage class containing the value of the attribute
	 * @return True if passed, false otherwise
	 */
	bool addPolarVolumeAttribute(PolarVolume_t* volume, string node, string attribute, const drain::Data &value);

	/**
	 * Add all attribute in an Image to a PolarVolume or a PolarScan if the volume is NULL.
	 * @param[in/out] volume - the volume.
	 * @param[in] image - the image to fetch attributes from
	 * @return True if passed, false otherwise
	 */
	bool addAttributes(PolarVolume_t* volume, drain::image::Image<>& image);
	
	/**
	 * Get index for scan as a string from an ODIM path. E.g. "/dataset2/data3/data" would return "2". 
	 * @param[in] path - the ODIM path.
	 * @return The index as a string
	 */
	string getScanIndex(string path);

	/**
	 * Get index for param as a string from an ODIM path. E.g. "/dataset2/data3/data" would return "3". 
	 * @param[in] path - the ODIM path.
	 * @return The index as a string
	 */
	string getParamIndex(string path);

	/**
	 * Create and temporary store empty PolarScan matching  the number of channels in the image. The scans are created here
	 * to avoid NULL pointers when trying to att attributes to PolarScans later. 
	 * @param[in] view - The view containing the channels. The view is sort of a wrapper for an Image, but Markus Peura can
                             probably explain/rewrite this better later.
	 */
	void addScans(drain::image::ImageView<>& view);

	/**
	 * Class with 3 main purposes:
	 * 1) Add the data array to the PolarScanParam with default quality and then add the PolarScanParam to the PolarScan.
	 * 2) All other parameters, which were stored untouched when converting from PolarScan to Image, are added
         *    to the PolarScan.
	 * 3) Add the calculated quality fields to the PolarScan.
	 * @param[in] scanIndex - Index indicating which scan to fill with data
	 * @param[in] image - The image to fetch data from
	 * @param[in] view - The view containing the channels.
	 * @param[in] quantity - the default quantity
	 */
	bool addDataToScan(string scanIndex, drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity);
	
	/**
	 * Wrapper function which calls the addDataToScan function for each PolarScan in the PolarVolume. However, since RACK
	 * doesn't fully support PolarVolumes yet, this function is not used at the moment. 
	 * @param[in] volume - The PolarVolume to fill with data
	 * @param[in] image - The image to fetch data from
	 * @param[in] view - The view containing the channels.
	 * @param[in] quantity - the default quantity
	 */
	bool addDataToVolume(PolarVolume_t* volume, drain::image::Image<>& image, drain::image::ImageView<>& view, string quantity);

	/**
	 * Clean up after converting. Free allocated memory etc.
	 */
	void cleanup();


	//Data

	//Since PolarVolume_getScan(...), PolarScan_getParameter(...) etc return a copy instead of the element
	//actually stored we must store all elements to be added/modified in a temporary container first.
	//When all data have been added to the container the PolarVolume will be filled with final data.
	map<string,ScanData*>  scanMap; //Map with scan data. Key is scan index as a string.

	vector<PolarScanParam_t*> paramStorage; //Parameters not used by rack, but still needs to be stored in resulting scan.

};

} //end namespace rack

#endif /*__RAVECONVERT__*/
