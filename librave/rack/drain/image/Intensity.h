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
#ifndef INTENSITY_H_
#define INTENSITY_H_


#include <limits>
#include <cmath>

namespace drain
{

namespace image
{


//template <class T>
class Intensity
{
public:

	/// For floating-point values, returns 0.0, otherwise the lower limit of the range.
	template <class T>
	inline
	static T min(){ return std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : 0; };

	/// For floating-point values, returns 1.0, otherwise the upper limit of the range.
	template <class T>
	inline
	static T max(){ return std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : 1; };

	/// Value to be used in inverting numbers; returns 0.0 for floating-point values.
	template <class T>
	inline
	static T maxOrigin(){ return std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : 0; };


	/// Ensures that the value stays within the numeric range of T2.
	template <class T>
	static T limit(double x){
		static const double xMin = static_cast<double>(
				numeric_limits<T>::is_integer ?  std::numeric_limits<T>::min() : -std::numeric_limits<T>::max() );
		static const double xMax = static_cast<double>( std::numeric_limits<T>::max() );
		x = std::max(xMin,x);
		x = std::min(x,xMax);
		return static_cast<T>(x);
	};

};



}

}

#endif /*INTENSITY_H_*/
