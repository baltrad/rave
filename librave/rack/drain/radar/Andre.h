/**

    Copyright 2001 - 2010  Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Drain library for C++.

    Drain is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    Drain is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Drain.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifndef ANDRE_H_
#define ANDRE_H_

#include <map>

#include "Geometry.h"

#include "../image/Image.h"
#include "../image/ImageView.h"
#include "../image/ImageOp.h"
#include "../image/File.h"

#include "../image/DistanceTransformOp.h"
#include "../image/FuzzyPeakOp.h"
#include "../image/FuzzyThresholdOp.h"
#include "../image/GammaOp.h"
#include "../image/HighBoostOp.h"
#include "../image/MathOpPack.h"
#include "../image/RunLengthOp.h"
#include "../image/SegmentAreaOp.h"

#include "../image/Drain.h"

/*
#include "../image/Image.h"
#include "../image/CatenatorOp.h"
#include "../image/CopyOp.h"
#include "../image/DistanceTransformOp.h"
#include "../image/DistanceTransformLogOp.h"
#include "../image/DistanceTransformFillOp.h"
#include "../image/FloodFillOp.h"
#include "../image/FuzzyPeakOp.h"

#include "../image/HighPassOp.h"
#include "../image/MathOpPack.h"
#include "../image/MarginalStatisticOp.h"
#include "../image/PaletteOp.h"
#include "../image/QuantizatorOp.h"
#include "../image/RecursiveRepairerOp.h"
#include "../image/RunLengthOp.h"
#include "../image/SegmentAreaOp.h"
#include "../image/SlidingStripeOp.h"
#include "../image/SlidingStripeAverageOp.h"
#include "../image/SlidingStripeAverageWeightedOp.h"
#include "../image/SlidingWindowMedianOp.h"
#include "../image/ThresholdOp.h"
 */
namespace drain
{

using namespace image;

namespace radar
{


/**
 *    \p
 */
template <class T = unsigned char,class T2 = unsigned char>
class AndreDetector : public image::ImageOp<T,T2>
{
public:

	AndreDetector(const string &name = "AndreDetector",
				const string &description="Radar data processor.",
				const string & parameterNames="",
				const string & defaultValues = "") :
					ImageOp<T,T2>(name,description,parameterNames,defaultValues) {};


	virtual void makeCompatible(const Image<T> &src, Image<T2> &dst) const {
		unsigned int channels = dst.getChannelCount();
		if (channels == 0){
			channels = src.getChannelCount();
		}
		dst.setGeometry(src.getWidth(),src.getHeight(),channels);
	};

protected:
	static Image<T> tmp;
};

template <class T,class T2>
Image<T> AndreDetector<T,T2>::tmp;



/// Detects speckle noise.
/**
 *  Segments of intensity > 'threshold' and area=1 get intensity 255, and larger ones
 *  smaller values with descreasing slope 'slope'.
 */
template <class T = unsigned char,class T2 = unsigned char>
class AndreSpeckle : public AndreDetector<T,T2>
{
public:

	AndreSpeckle(const string p="1,3") :
		AndreDetector<T,T2>("aSpeckle","Detects speckle noise.","threshold,slope",p)
	{
		//this->setInfo("Detects speckle noise.","threshold,size",p);
	};

	void filter(const image::Image<T> &src,image::Image<T2> &dst) const {

		makeCompatible(src,dst);

		const int threshold = this->getParameter("threshold",1);
		const int size = this->getParameter("size",3);

		SegmentAreaOp<T> area;  //min,max,mapping,mSlope,mPos"
		area.setParameter("mapping","f");  // bilinear inverse
		area.setParameter("mPos",1);
		area.setParameter("mSlope",size);
		area.setParameter("min",threshold);
		area.setParameter("max",250);
		area.filter(src,dst);
		//if (drain::Debug > 4)
		//	File::write(dst,"andre-speckle-0.png");

	};
};

/// Detects line segments caused by electromagnetic interference.
/**
 *  First, computes vertical run lengths and inverts values so that the shortest segments become the brightest.
 *  Then, continues by computing horizonal run lengths. Finally, masks
 */
template <class T = unsigned char,class T2 = unsigned char>
class AndreEmitter : public AndreDetector<T,T2>
{
public:

	AndreEmitter(const string p="1,8"){
		this->setInfo("Detects line segments caused by electromagnetic interference. Parameters: max vertical width, min horizontal length.","thickness,length",p);
	};

	void filter(const image::Image<T> &src,image::Image<T2> &dst) const {

		makeCompatible(src,dst);

		const int thickness = this->getParameter("thickness",1);
		const int length = this->getParameter("length",8);

		Image<T> vert;

		RunLengthOp<T> rle;
		rle.setParameter("mode","vert");
		rle.setParameter("threshold",1);  // yes
		rle.filter(src,vert);
		if (drain::Debug > 2)
			File::write(vert,"andre-rle-0-vert0.png");

		RemapOp<T> remap;
		remap.setParameter("from",0);
		remap.setParameter("to",255);
		remap.filter(vert,vert);
		if (drain::Debug > 4)
			File::write(vert,"andre-rle-0-vert1.png");

		FuzzyPeakOp<T> peak;
		peak.setParameter("location",1);
		peak.setParameter("width",thickness);
		peak.filter(vert,vert);
		if (drain::Debug > 3)
			File::write(vert,"andre-rle-0-vert2.png");

		//Image<T> horz;
		//RunLengthOp<T> rle;
		rle.setParameter("mode","horz");
		rle.setParameter("threshold",thickness);  // yes
		rle.filter(vert,dst);
		if (drain::Debug > 4)
			File::write(dst,"andre-rle-1-horz0.png");

		FuzzyThresholdOp<> threshold;
		threshold.setParameter("location",length);
		threshold.setParameter("width",length/8);
		threshold.filter(dst,dst);
		if (drain::Debug > 3)
			File::write(dst,"andre-rle-1-horz1.png");

		// NOTE lastDetection contains vertical mask
		//MaximumOp<>().filter(vert,dst,dst);
		MultiplicationOp<>("255,0").filter(vert,dst,dst);
		if (drain::Debug > 3)
			File::write(dst,"andre-emitter.png");

	}
};

/// Detects ships.
/**
 *  Computes segment sizes and scales them fuzzily such that segment of area 'size' get intensity 128.
 */
template <class T = unsigned char,class T2 = unsigned char>
class AndreShip : public AndreDetector<T,T2>
{
public:

	AndreShip(const string p="64,4,5"){
		this->setInfo("Detects ships.","threshold,size,sidelobe",p);
	};

	void filter(const image::Image<T> &src,image::Image<T2> &dst) const {

		makeCompatible(src,dst);

		const int threshold = this->getParameter("threshold",64);
		const int size = this->getParameter("size",4);
		const int sidelobe = this->getParameter("sidelobe",5);

		/*
		HighBoostOp<T> high;
		high.setParameter("width",3);
		high.setParameter("height",3);
		//high.filter(src,this->tmp);
		high.filter(src,dst);
		 */

		SegmentAreaOp<T,T> area;
		area.setParameter("mapping","f");  // bilinear inverse
		area.setParameter("min",threshold);
		area.setParameter("targetSize",size);
		area.setParameter("halfWidth",size/2);
		area.filter(src,dst);
		MultiplicationOp<T,T>().filter(src,dst,dst);

		if (sidelobe > 0){
			DistanceTransformExponentialOp<T,T> dist;
			//DistanceTransformLinearOp<T,T> dist;
			dist.setParameter("horz",1);
			dist.setParameter("vert",sidelobe);
			dist.setParameter("diag",1);
			dist.filter(dst,dst);
		}

	};
};

/** AnDRe is a radar image processing module that detects and removes
 *
 */
template <class T = unsigned char,class T2 = unsigned char>
class Andre : public Drain<T>
{

public:

	AndreEmitter<T> emitter;
	AndreSpeckle<T> speckle;
	AndreShip<T> ship;

	Andre(){};

	virtual void addDefaultOps(){

		addOperator("emitter",emitter);
		addOperator("speckle",speckle);
		addOperator("ship",ship);
		reset();

	};

	// TODO: initialize()


	/*
	void setSource(const image::Image<T> &src){
		this->src.viewChannels(src,src.getImageChannelCount(),0);
	};
	*/

	/// Computes or retrieves a primary feature image from cache, if available, or generates it if needed.
	/** Primary features are further used by getDetection
	 *
	 */
	/*
	const image::Image<T> &getFeature(const string &op,const string &parameters){
			string key = op + ':' + parameters;
			if (features.find(key) == features.end()){
				drain.executeOperator(op,parameters,src,features[key]);
				features[key].setName(key);
			}
			return features[key];
	};
	 */

	void computeDetection(const Image<T> &src, const string &op,const string &parameters){
		executeOperator(op,parameters,src,lastDetection);
	};

	void computeDetection(Image<T> &src, const string &op,const string &parameters){
			executeOperator(op,parameters,src,lastDetection);
	};


	/// Enhances last detection field with beamwise, spread indications. Finally, calls cumulate.
	/**
	 *   \param a - statistic to be calculated; first letter of: average, sum, std.dev, max,... See MarginStatistics
	 */
	void enhance(char statistic='a',int radius='8'){

		Image<float> stat; // small tmp image

		MarginalStatisticOp<T,float> s;
		s.setParameter("mode","horz");
		s.setParameter("stat",statistic);
		//cerr << "MarginalStatisticOp:" << parametersS[0];
		s.help();
		s.filter(lastDetection,stat);

		//CopyOp<float,T> copy;
		//copy.filter(stat,tmp); File::write(tmp,"stat1.png");

		DistanceTransformExponentialOp<float,float> dist;
		dist.setParameter("horz",0);
		dist.setParameter("vert",radius);
		dist.filter(stat,stat);
		//copy.filter(stat,tmp); File::write(tmp,"stat2.png");
		const unsigned int width  = lastDetection.getWidth();
		const unsigned int height = lastDetection.getHeight();
		const unsigned int iChannels = lastDetection.getImageChannelCount();

		for (unsigned int k=0; k<iChannels; k++)
			for (unsigned int j=0; j<height; j++){
				const float coeff = stat.at(0,j,k);
				for (unsigned int i=0; i<width; i++){
					T &f = lastDetection.at(i,j,k);
					f = static_cast<T>(f + (coeff/255.0)*(255.0-f));
				}
			}

	}


	/// Adds a detection to the overall detection field by using maximum principle.
	/** This is useful for adding external detection results.
	 *
	 */
	void cumulate(){
		cumulate(lastDetection);
	};

	/// Adds (the latest) detection to the overall detection field by using maximum principle.
	void cumulate(const Image<T> & detection){
		//MaximumOp<T>().filter(lastDetection,cumulatedDetection,cumulatedDetection);

		cumulatedDetection.setGeometry(detection.getGeometry());

		// ???? YLim.?
		typename std::vector<T>::const_iterator s  = detection.begin();
		typename std::vector<T>::const_iterator s2 = cumulatedDetection.begin();
		typename std::vector<T>::iterator d;
		typename std::vector<T>::iterator dEnd = cumulatedDetection.end();

		for (d=cumulatedDetection.begin(); d!=dEnd; d++,s++, s2++)
			*d = *s > *s2 ? *s : *s2;

	}

	//map<string,image::Image<T> > features;
	void clear(){
		lastDetection.clear();
		lastDetection.properties.clear();
		cumulatedDetection.clear();
		cumulatedDetection.properties.clear();
		cumulatedDetection.properties["detections"].separators = "_";
	};

	void reset(){
		lastDetection.reset();
		lastDetection.properties.clear();
		cumulatedDetection.reset();
		cumulatedDetection.properties.clear();
		cumulatedDetection.properties["detections"].separators = "_";
	};

	Image<T> lastDetection;
	Image<T> cumulatedDetection;



protected:
	//image::Drain<T> drain;


	//ImageView<T> src;
	//Image<T2> dst;


};



} // ::image

} // ::drain

#endif /* ANDRE_H_*/
