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
#ifndef DRAIN_H_
#define DRAIN_H_

#include <string>
#include <list>
#include <map>
#include <set>

#include <exception>


#include "Image.h"
#include "CatenatorOp.h"
#include "CopyOp.h"
#include "DoubleSmootherOp.h"
#include "DistanceTransformOp.h"
//#include "DistanceTransformLogOp.h"
#include "DistanceTransformFillOp.h"
#include "FastAverageOp.h"
#include "FastOpticalFlowOp.h"
#include "FloodFillOp.h"
#include "FuzzyPeakOp.h"
#include "FuzzyThresholdOp.h"
#include "GammaOp.h"
#include "GradientOp.h"
#include "HighPassOp.h"
#include "MarginalStatisticOp.h"
#include "MathOpPack.h"
#include "PaletteOp.h"
#include "PixelVectorOp.h"
#include "QuadraticSmootherOp.h"
#include "QuantizatorOp.h"
#include "RecursiveRepairerOp.h"
#include "RunLengthOp.h"
#include "SegmentAreaOp.h"
#include "SlidingStripeOp.h"
//#include "SlidingStripeAverageOp.h"
//#include "SlidingStripeAverageWeightedOp.h"
#include "SlidingWindowMedianOp.h"
#include "ThresholdOp.h"

//#include ""

namespace drain
{

namespace image
{

///   A class that wraps various image processing operators into a service instance.
/**
 *   
 * 
 *  \section Examples
 * 
 *  \subsection Scaling intensities
 * 
 *  \code
 *   drainage test-grey.png --threshold 128 -o threshold.png
 *  \endcode
 *  
 *  \code
 *   drainage test-grey.png --fuzzyThreshold 128,5 -o fuzzyThreshold.png
 *  \endcode 
 *  
 *  Correspondingly, intensities around a given intensity can be highlighted 
 *  with a \em fuzzy \em peak.
 *    
 *  \code
 *   drainage test-grey.png --fuzzyPeak 128,32 -o fuzzyPeak.png
 *  \endcode 
 *
 *  \subsection Window operations
 * 
 *   
 *
 */
template <class T=unsigned char>  
class Drain : private map<string,ImageOp<T,T> *> {
	
protected:
	
	AdditionOp<T,T> add;
	CatenatorOp<T,T> cat;
	CopyOp<T,T> copy;
	DoubleSmootherOp<T,T> doubleSmoother;
	FastAverageOp<T,T> fastAverage; //wavgWindow;   // window only
	GradientHorizontalOp<T,T> gradientHorz;
	GradientVerticalOp<T,T>   gradientVert;
	DistanceOp<T,T> distance;
	DistanceTransformLinearOp<> distanceTransform;
 	DistanceTransformExponentialOp<> distanceTransformLog;
	DistanceTransformFillOp<> distanceTransformFill;
	DistanceTransformFillExponentialOp<> distanceTransformFillExp;
	DivisionOp<T,T> div;
	GammaOp<T,T> gamma;
	HighPassOp<T,T> highPass;
	FloodFillOp<T,T> floodFill;
	FuzzyPeakOp<T,T> fuzzyPeak;
	FuzzyThresholdOp<T,T> fuzzyThreshold;
	MaximumOp<T,T> max;
	MagnitudeOp<T,T> magnitude;
	MarginalStatisticOp<T,T> marginStat;
	MinimumOp<T,T> min;
	MixerOp<T,T> mix;
	MultiplicationOp<T,T> mul;
	NegateOp<T,T> negate;
	// Palette?
	ProductOp<T,T> product;
	QuantizatorOp<T,T> quantize;
	QuadraticSmootherOp<T,T> quadSmooth;
	RecursiveRepairerOp<T,T> rec;
	RemapOp<T,T> remap;
	RunLengthOp<T,T> runLength;
	ScaleOp<T,T> rescale;
	SegmentAreaOp<T,T> segmentArea;
   	SlidingWindowMedianOp<T,T> median;
	SubtractionOp<T,T> sub;
	ThresholdOp<T,T> threshold;
 	
public:

 	//: average(avgWindow) , averageWeighted(wavgWindow), median(medianWindow) {
	Drain(){};
	
	virtual void addDefaultOps(){
		// Simple math ops
		// addOperator("",);
		addOperator("add",add);
		addOperator("average",fastAverage);
		addOperator("cat",cat);
		addOperator("copy",copy);
		addOperator("gradientHorz",gradientHorz);
		addOperator("gradientVert",gradientVert);
		addOperator("distance",distance);
		addOperator("distanceTransform",distanceTransform);
		addOperator("distanceTransformLog",distanceTransformLog);
		addOperator("distanceTransformFill",distanceTransformFill);
		addOperator("distanceTransformFillExp",distanceTransformFillExp);
		addOperator("div",div);
		addOperator("doubleSmoother",doubleSmoother);
		addOperator("gamma",gamma);
		addOperator("highPass",highPass);
		addOperator("floodFill",floodFill);
		addOperator("fuzzyPeak",fuzzyPeak);
		addOperator("fuzzyThreshold",fuzzyThreshold);
		addOperator("max",max);
		addOperator("magnitude",magnitude);
		addOperator("marginalStatistic",marginStat);
		addOperator("min",min);
		addOperator("mix",mix);
		addOperator("median",median);
		addOperator("mul",mul);
		addOperator("negate",negate);
		addOperator("product",product);
		addOperator("quantize",quantize);
		addOperator("quadSmooth",quadSmooth);
		addOperator("rescale",rescale);
		addOperator("remap",remap);
		addOperator("rec",rec);
		addOperator("runLength",runLength);
		addOperator("segmentArea",segmentArea);
		addOperator("sub",sub);
		addOperator("threshold",threshold);
	};
	
	virtual ~Drain(){};
	
	void prefixAll(const string &prefix, bool upperCaseFirst=true){

		set<string> keys;

		for (typename Drain<T>::iterator it = this->begin(); it != this->end(); it++)
			keys.insert(it->first);

		for (typename set<string>::iterator it = keys.begin();it != keys.end(); it++){
			string newKey = *it;
			if (upperCaseFirst)
				if (!newKey.empty())
					newKey[0] = newKey[0]+'A'-'a';
			(*this)[prefix+newKey] = (*this)[*it];
			this->erase(*it);
		}
	}

	/// Adds an image processing operator to the internal map. 
	/// Thereafter the operator can be executed with executeOperator().
	void addOperator(const string &name,ImageOp<T,T> &op){
		(*this)[name] = & op;
	}
	
	void removeOperator(const string &name){
		this->erase(name);
	}

	/// 
 	bool hasOperator(const string &key) const {
		return (this->find(key) != this->end());
	}
	
	
	/// An interface for single-image processing  
	int executeOperator(const string & name, const string & parameters,
		const Image<T> & src, Image<T> & dst){
		if (hasOperator(name)){
			if (drain::Debug > 0)
				cerr << "executing: '" << name << "'\n";
			ImageOp<T,T> & op = *(*this)[name];
			op.parameters.set(parameters); 
			op.filter(src,dst);
		}
		else {
		  throw runtime_error("drain::Drain: Could not find operator " + name);
		}		
		return 0;
	};
	

	/// Lists the installed operator names, parameters, and descriptions. If key is given, returns only help for that command.
	void help(ostream &ostr, const string & key = "", const string & prefix = "--", 
			const string & postfix = "\n\t") const {
		if (key == ""){
			for (typename map<string,ImageOp<T,T> *>::const_iterator it = this->begin();
					it != this->end(); it++){
				ostr << prefix << it->first;
				if ( !it->second->parameters.getParameterNames().empty() ) //it->second->parameters.isFlag )
					//	ostr << it->second->parameters.info << '\n';
					//else
					ostr << " <" << it->second->parameters.getParameterNames() << '>';
				ostr << postfix;
				ostr << it->second->parameters.getDescription() << '\n';
			//it->second->help(ostr);
		  }
		}
		else {
			if (hasOperator(key)){
				typename map<string,ImageOp<T,T> *>::const_iterator it = this->find(key);
				if (it != this->end()){
					it->second->help(ostr);
				}
			}
		}
    } 
	
	/// Convenience
    string help() const {
    	stringstream s;
    	help(s);
    	return s.str();	
    }
	
		
	//int handle(const string & command, const string & parameters,
	//	const list<BufferedImage<T> > & srcList, list<BufferedImage<T> > & dstList);
};

/*
template <class T>  
int Drain<T>::handleImage(const string & command, const string & parameters,
		const BufferedImage<T> & src, BufferedImage<T> & dst){
	list<BufferedView<T> > srcList(1);
	list<BufferedView<T> > dstList(1);
	srcList.front().viewImage(src);
	dstList.front().viewImage(dst);
	return handle(command,parameters,srcList,dstList);
}
*/
		

}

}

#endif /*DRAIN_H_*/
