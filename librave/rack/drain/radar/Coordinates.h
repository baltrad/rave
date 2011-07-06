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
#ifndef RADAR__COORDINATES_ 
#define RADAR__COORDINATES_ "radar__coordinates 0.1, Jun 21 2006 Markus.Peura@fmi.fi"

#include <math.h>

#include <iostream>
#include <ostream>
#include <sstream>

#include "Constants.h"
//#include "radar.h"


namespace drain {

namespace radar {

  /// Site dependent coordinate computations. Does not handle projections, but earth coords.
  /*! 
     \image latex radar-coordinates-fig.pdf
     \image html  radar-coordinates-fig.png
    
    \section Variablenames Variable names
   
      - \f$\phi\f$, phi:  longitude of radar site (in radians)
      - \f$\theta\f$, theta: latitude of radar site (in radians)
      - \f$\alpha\f$, alpha: azimuth angle of the radar beam, \f$+\pi/2\f$=North
      - \f$r\f$:  distance to the surface point
      - \f$\boldmath{e}_{i}=(e_{i1}e_{i1}e_{i1})\f$: unit vectors at the site
     

     Note. This is a model for ideal sphere, defined as
     \code
     "+proj=longlat +R=6371000 +no_defs";
     \endcode
     where R is the radius of the Earth. Use getDatumString() to get the actual string.

     So strictly speaking, this is not the standard proj
     \code
     EPSG:4326 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs";
     \endcode
   */
  class Coordinates {  // consideder : public Proj?
  public:

	  virtual
	 ~Coordinates(){};

    //    coordinator();

    /// Radar site latitude and longitude in radians.
    void origin(const double &theta,const double &phi); 

    // inline
    void setOriginDeg(const double &lat,const double &lon);
      //      origin(lat*M_PI/180.0 , lon*M_PI/180.0);
      //    }; 
  //const double &lat,const double &lon);

    /// \par alpha is azimuth in radians, \par range in metres.
    void setBinPosition(const double &alpha, const float &range);
   
    // \par alpha is azimuth in radians, \par range in metres.
    //void setBinPosition(double alpha, float range);
   
	/// The bounding box (in degrees) of the circular radar measurement area.  
    void getBoundingBox(float range,double &latMin,double &lonMin,double &latMax,double &lonMax);

    /// Info
    void info(std::ostream &ostr = std::cout);

    // site positional
    // double cos_theta;
 
    /// Radar position vector (from Earth center to surface
    // (Site normal unit vector not needed as such)
    double p01, p02, p03;

    /// Earth centered coordinates [p_1 p_2 p_3] of the current bin position.
    double p1, p2, p3;

    /// Elelements of the East pointing site unit vector [e11 e12 e13].
    double e11, e12, e13;

    ///  Elelements of the North pointing site unit vector [e21 e22 e23].
    double e21, e22, e23;

    /// Bin latitude in radians after calling bin_position().
    mutable double thetaBin;

    /// Bin longitude in radians after calling bin_position().
    mutable double phiBin;

    ///  Bin latitude in degrees after calling bin_position().
    inline
    double binLatitudeDeg(){ return thetaBin/M_PI*180.0;};
				//phi_bin/M_PI*180.0;};
				//

    ///  Bin longitude in degrees after calling bin_position().
    inline
    double binLongitudeDeg(){ return phiBin/M_PI*180.0;};
    //    theta_bin/M_PI*180.0;};;
    //
    inline
    virtual std::string getProjectionString(){
    	std::stringstream sstr;
    	sstr << "+proj=longlat +R=" << EARTH_RADIUS << std::string(" +no_defs");
    	return sstr.str();
    };
  };

} // ::radar


} // ::drain

#endif
