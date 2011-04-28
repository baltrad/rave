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
#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <limits>

namespace drain
{

/// Class for computing a histogram and some statistics: average, min, max, mean, std.dev, sum.
/**
 *  T  input type, eg. unsigned char for 8-bit images.
 *  T2 output type for values not within the set of input values, like std.dev.
 */
template <class T,class T2=float>
class Histogram : public vector<T> {
public:
	Histogram(typename vector<T>::size_type size=256) : vector<T>(size) { delimiter = ", "; };
	virtual ~Histogram(){};

	string delimiter;
	// void setMedianPercentage(double p);
	// inline float getMedianPercentage(){ return medianPercentage; };
	typedef typename vector<T>::size_type size_type;
	/// 
	/*!
	 *  @param p applies to weighted median; for standard median, p = 0.5.
	 */

	T getMax() const {
		const size_type bins = vector<T>::size();
		for (size_type i = bins-1; i > 0; i--)
			if ((*this)[i] > 0)
				return i;
		return 0;
		// cerr << "Warning: Histogram empty.\n";
	};

	T getMin() const {
		//T result =  std::numeric_limits<T>::max();
		const size_type bins = vector<T>::size();
		for (size_type i = 0; i < bins; i++)
			if ((*this)[i] > 0)
				return i;
		// TODO: EXCEPTION if empty?
		cerr << "Warning: Histogram empty.\n";
		return bins-1;
	};

	T getMedian(float p=0.5) const {
		if ((p < 0.0) || (p>1.0)){
			throw runtime_error("Histogram<T>::getMedian: median point <0 or >1.0 .");
		}
		const long int limit = static_cast<unsigned int>(p * sampleCount);
		long int sum = 0;
		const size_type bins = vector<T>::size();
		for (size_type i = 0; i < bins; i++){
			sum += (*this)[i];
			if (sum >= limit){
				return i;
			}
		}
		return bins;
	};


	//template <class T2>
	T2 getSum() const {
		double sum = 0;
		const size_type bins = vector<T>::size();
		for (size_type i = 0; i < bins; i++){
			sum += (*this)[i]*i;
		}
		return static_cast<T2>(sum);
	};

	//template <class T2>
	T2 getMean() const {
		double sum = 0;
		const size_type bins = vector<T>::size();
		for (size_type i = 0; i < bins; i++){
			sum += (*this)[i]*i;
		}
		return static_cast<T2>(sum/sampleCount);
	};

	//template <class T2>
	T2 getStdDeviation() const {
		const size_type bins = vector<T>::size();
		double n = 0;
		double sum = 0;
		double sum2 = 0;
		for (size_type i = 0; i < bins; i++){
			n = static_cast<double>((*this)[i]);
			sum  += n*i;
			sum2 += n*(i*i);
		}
		sum = sum/sampleCount;
		return static_cast<T2>(sum2/sampleCount - sum*sum);
	};

	void clearBins(){
		const size_type bins = vector<T>::size();
		for (size_type i = 0; i < bins; i++)
			(*this)[i] = 0;
	};

	inline void setSampleCount(long int n){ sampleCount = n; };

	inline void setSize(typename vector<T>::size_type s){ resize(s,0); };

	inline const vector<T> getVector() const {return *this;};


protected:
	long int sampleCount;
	//float medianPercentage;  
};

}

//template <class T>


#endif /*HISTOGRAM_H_*/
