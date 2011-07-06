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
#ifndef PROJ4_H_
#define PROJ4_H_


#include <string>
#include <iostream>

#include <proj_api.h>

using namespace std;


namespace drain
{

/*! A wrapper for the famous Proj4 class.
 *  
 *
 */ 
class Proj4
{
public:
	Proj4();
	virtual ~Proj4(); 
    
    /// Sets destination projection.
    void setProjection(const string &str);
   
    /// Sets destination projection according to string presentation of proj.
    inline
    void setProjection(const Proj4 &proj)
    {
    	setProjection(proj.getProjectionString());
    };
    
    
    
    /// Unscaled projection, ie. that directly produced by proj4 library. Input in radians.
    inline
    void project(double & x, double & y) const   // todo: phi,theta
    {
    	if (projSrc == NULL){
    		cerr << "project() warning: projSrc NULL" << endl; 	
    	}
    	if (projDst == NULL){
    		cerr << "project() warning: projDst NULL" << endl; 	
    	}
    	pj_transform(projSrc, projDst, 1, 1, &x, &y, NULL );
    };
    
    
   	/// Returns the projection string applied by the last setProjection call.
   	/**! Hence, does not reconstruct it from the allocated proj structure.
   	 */
    inline
    string getProjectionString() const {
    	//return projStr;
    	return string(pj_get_def( projDst,0));
    };
    
    inline
    string getErrorString() const {
    	//projErrorStr = pj_strerrno(*pj_get_errno_ref());
    	return string(pj_strerrno(*pj_get_errno_ref()));
    };
        
    bool ok;
        
protected:

	    
	//string projStr;  // obsolete?
	//mutable string projErrorStr;
	
    projPJ projSrc;
    projPJ projDst;
    
 
	
};

}

#endif /*PROJ4_H_*/
