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
#ifndef IMAGE_OP_H_
#define IMAGE_OP_H_

//#include <exception>
#include <stdexcept>
#include <list>

#include "Image.h"

namespace drain
{

namespace image
{

/**  
 *  Contract.
 * 
 *  If (dst == src), and computation involves writing to dst in a way that affects
 *  succeeding computation, operator should apply an internal temporary image.   
 * 
 *  Experimental, optional: Using (Data) parameters only.
 *  The allowed parameters are defined through the list parameterNames, which also
 *  states the order of parameters, if given through setParameters(const Data & d). 
 * 
 */
template <class T,class T2>
class ImageOp  
{
public:

	// \bug TODO: problem default value must be given in this call.
   ImageOp(const string &name = "ImageOp",const string &description="Image processing operator.",
		   const string & parameterNames="", const string & defaultValues = "") : name(name) {
	   setInfo(description,parameterNames,defaultValues);
   };
   
   virtual ~ImageOp(){};

   virtual void makeCompatible(const Image<T> &src,Image<T2> &dst) const  {
    	//cerr << "ImageOp\n";
		dst.setGeometry(src.getGeometry());
    	//coordinateHandler->setBounds(src.getWidth(),src.getHeight());  // ?? both
    	// TODO: change handler to similar  
    	dst.getCoordinateHandler().setBounds(src.getWidth(),src.getHeight());  // ??
	};
   
   	/*
	void setCoordinateHandler(CoordinateHandler &handler){
    	coordinateHandler = &handler;
	};
	*/
   

   /// TODO: const
   virtual void filter(const Image<T> &src,Image<T2> &dst) const = 0;
   /*
   {
   	  throw runtime_error("(ImageOp unimplemented.)");
   };
    */


   /// Container for storing operator parameters as KEY,VALUE pairs.  
   /*! I am still hesitating with this---
    *   Advantages:
    *   The parameters can be more easily...
    *   # read from a command file
    *   # dumped, for debugging & reflection 
    *   Disadvantages:
    *   # using this hides the actual parameters from the interface
    *   # type checks?  
    */
    //void setParameter(const string  
    //Data info;
    
    //const string & getHelp(){ return parameters; };
    
    /// Throws exception if the number of parameters exceed the number.d 
    void setParameters(const string & p){
    	parameters.set(p);
    }
    
    bool hasParameter(const string & key) const {
    	 return parameters.hasKey(key);
    }
	
    void setParameter(const string & key,const Data & value){
   		parameters[key] = value;
    }
	

    //template <>
    const char * getParameter(const string & key,const char *defaultValue) const {
    		return parameters.get(key,defaultValue).c_str();
   	}


	template <class T3>
	T3 getParameter(const string & key,T3 defaultValue) const {
   		return parameters.get(key,defaultValue);
    }

	inline
	const string & getParameterNames() const {
		return parameters.getParameterNames();
    }

	inline
	const string &getParameterValues() const {
		return parameters.getParameterValues();
	}
	/*
	inline
	const string getParameterValues() const {
		stringstream sstr;
		string separator = "";
		for (map<string,Data>::const_iterator it = opt.begin(); it != opt.end(); it++){
			sstr << separator << it->second;

		}
		return sstr.str();
		//return parameters.getParameterNames();
	}
	*/

	inline
	const Options & getParameters() const {
		return parameters;
	}

	const string & getName() const { return this->name; };

	// protect
	Options parameters;
	
	/// Sets the applicable parameters, their order for setParameters() calls as well as usage information to be retrieved by help calls.
	void setInfo(const string & description, const string & parameterNames, const string & defaultValues = ""){
		parameters.clear();
    	parameters.setDescription(description);
    	parameters.setParameterNames(parameterNames);
    	parameters.set(defaultValues);
    };
   
    void help(ostream &ostr) const {
    	ostr << name << ": ";
    	ostr << parameters.getDescription() << '\n';
    	ostr << "Parameters: " << parameters.getParameterNames() << '\n';
    	ostr << "Current values:\n" << parameters << '\n'; 	
    } 
    
    string help() const {
    	stringstream s;
    	help(s);
    	return s.str();	
    }
	// comma-separated list of parameters in the order specified by parameters.getAllowedParameters()
	// void setParameters(const string & s);
	


protected:

    /// Utility for re-calling filter() with a temporary dst image, when dst==src would corrupt the computation.
    /**  Experimental. Place this in the first line of your filter() as simply as:
     *  \code
     *  if (filterWithTmp(src,dst))
     *  	return;
     *  \endcode
     */
    bool filterWithTmp(const Image<T> &src,Image<T2> &dst) const {

    	if (src.hasOverlap(dst)){
    		Image<T2> tmp;
    		filter(src,tmp);
    		///
    		dst.setGeometry(tmp.getGeometry());
    		/// Copy
    		typename std::vector<T2>::const_iterator si = tmp.begin();
            typename std::vector<T2>::iterator di = dst.begin();
            const typename std::vector<T2>::const_iterator dEnd = dst.end();
    		while (di!=dEnd){
    			*di = *si;
    			si++;
    			di++;
            }
    		return true;
    	}
    	else
    		return false;
    }

    /// Should the operation be carried out to each channel
    bool filterEachChannel(const Image<T> &src,Image<T2> &dst) const {

    	const unsigned int c = src.getImageChannelCount();
    	dst.setGeometry(src.getWidth(), src.getHeight(), c, dst.getAlphaChannelCount());

    	if (c == 1)
    		return true;

    	for (unsigned int i = 0; i < c; ++i)
    		filter(src.getChannel(i),dst.getChannel(i));

    	return false;
    }

    // todo: make it const
    string name;
   
    //drain::MapWrapper<string,Data> parameterWrapper;
    /*
	CoordinateHandler *coordinateHandler;
	CoordinateHandler defaultCoordinateHandler;
	Mirror mirror;
	Wrapper wrapper;
	*/
};

//static unsigned int ImageOpDebug;
//template <class T,class T2>
//unsigned int ImageOp<T,T2>::debug = 0;
   
   
   
}
}

#endif /* IMAGE_OP_H_ */
