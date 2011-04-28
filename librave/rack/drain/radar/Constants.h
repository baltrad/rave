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
#ifndef RADAR__CONSTANTS 
#define RADAR__CONSTANTS "radar__constants 1.0, May 27 2006 Markus.Peura@fmi.fi"

//#include <math>


/*! \file
 *  This file contains radar constants...
 */
// 6,370
namespace drain {

namespace radar {

  const int EARTH_RADIUSi = 6371000;
  const double EARTH_RADIUS = static_cast<double>(EARTH_RADIUSi);

  const double EARTH_RADIUS_43 = EARTH_RADIUS * 4.0 / 3.0;

  const int     earthRadiusMI = 6371000;  // WHY THIS AS WELL
  const int   earthRadius43MI = (earthRadiusMI * 4) / 3;
  const float earthRadius43MF = static_cast<float>(earthRadiusMI) * 4.0 / 3.0;


  /*! Converts Z from logarithmic to linear scale. 
   *  \f[                                          
   *      Z = 10^{dBZ/10}                          
   *  \f]                                          
   */
  template <class T>                               
  inline double dbzToZ(const T &dBZ){              

	  return pow(10.0, static_cast<double>(dBZ)/10.0 );
  }                                                  

  /*! Converts Z from logarithmic to linear scale.   
   *  \f[                                            
   *      dBZ = 10\log_{10}Z = 10 \ln Z / \ln 10     
   *  \f]                                            
   */
  template <class T>                                 
  inline double zToDbz(const T &Z){                  
	  static const double coeff = 10.0 / log(10.0);
	  return coeff * log( static_cast<double>(Z) ) ;
  }                                                  


} // ::radar


} // ::drain

#endif
