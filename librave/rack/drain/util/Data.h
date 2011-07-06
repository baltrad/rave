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
#ifndef Data_H_
#define Data_H_

#include <iostream> 
#include <string>
#include <sstream> 

#include <typeinfo>

//#include <list>
#include <list>
#include <vector>

using namespace std;

namespace drain {

///  A multi-purpose container designed for storing configuration data.
/*!  Data are internally stored as a string but handy conversions are supported.
 *   
 *   Example:
 *   \code
 *   Data data;
 *   data = 123.456;   
 *
 *   int i = data;    // i = 123
 *   float f = data;  // f = 123.456
 *   \endcode
 *   
 *  If data string contains separator characters, the string is split in when applicable.
 *  Example:
 *  \code
 *   data = "123.456, test, 789";
 *  
 *   vector<float> vectorF;
 *   data.splitTo(vectorF); //  123.456  NaN 769
 * 
 *   vector<Data> vectorD;
 *   data.splitTo(vectorD); // "123.456" "test" "769"
 *
 *   float f = vectorD[0];
 *  \endcode
 *  The separator characters are stored as a string.
 *
 *  The data can be appended with << operator, which uses the \i first character in separator string
 *  as a separator:
 *  \code
 *  data = 123.456;  // "123.456"
 *  data << "test";  // "123.456,test"
 *  data << 769;     // "123.456,test,769"
 *  \endcode
 *
 *  The content is always stored as a string, but type information
 *  is stored in parallel. It is automatically set upon the first assignment but can be changed by the user.
 *	\code
 *	data = 3.12;  // double
 *	if (data.getType == typeof(float))
 *	    ...
 *
 *	data.setType(float);
 *  \endcode
 *
 *  Remark. One should be careful when assigning unsigned chars.
 *  Consider:
 *  \code
 *  unsigned char n;
 *  Data data;
 *
 *  data = 'a'; // stored as "a"
 *  n = data;   // n = 65  (ASCII number of 'a')
 *
 *  data = 3;   // stored as "3"
 *  n = data;   // n = 51 ! (ASCII number of '3')
 *  \endcode
 *
 */
class Data: public string {
public:

	//bool lenient;
	/// NEW: may also be empty.
	string separators;

	string trimChars;

	Data();
	//Data(const string &s = "");

	Data(const Data &d);

	template<class T>
	Data(const T &t) {
		separators = ",";
		trimChars = " \t\n";
		type = &typeid(T);
		(*this) = t;
		//type = &typeid(T);
	};

	template<class T>
	Data &operator =(const T &x);

	Data &operator =(const Data &d);

	Data &operator =(const char *str);

	Data &operator =(const string &str);

	template<class T>
	Data &operator <<(const T &x);

	Data &operator <<(const Data &d);

	/// Conversion to any type is carried out through sstr.
	template<class T>
	operator T() const;


	//template <class T>
	//Data operator()(int i) const;

	/// Splits the contents to elements and exports them to a list
	template<class T>
	void splitTo(list<T> &l, const string &separator = "") const;

	/// Splits the contents to elements and exports them to a vector
	template<class T>
	void splitTo(vector<T> &v, const string &separator = "") const;

	/// Implicit assignment of a type. Remember to update syntax, if needed.
	/** In Data, the content is always stored as a string, but type information
	 *  can be stored in parallel.
	 *  The stored type is either the original type obtained at the first assignment or
	 *  a type in which the contents are desired to be converted.
	 *
	 */
	template<class T>
	inline
	void setType() {
		if (typeid(T) == typeid(bool)){
			syntax.clear();
		}
		else {
			if (syntax.empty())
				syntax = "<value>";
		}
		type = &typeid(T);
	};

	/// Returns false if type is undefined (void).
	inline
	bool typeIsSet() const { return type != &typeid(void); };

	/// Returns the type of data.
	inline
	const type_info &getType() const {
		return *type;
	};

	inline
	bool isFlag() const { return syntax.empty(); };


	// Under constr
	/*
	template<class T>
	bool isType() const;
	 */

	string description;
	string syntax;



	/// Removes all trimChars from both ends of the string.
	inline Data & trim() {
		trim(*this);
		return *this;
	};



	/// For convenience Data has three internal vectors.
	//inline const & vector<int>     getIntVector() const { splitTo(intVector); return intVector; };
	//inline const & vector<float>   getFloatVector() const { splitTo(floatVector); return floatVector; };
	//inline const & vector<string>  getStringVector() const { splitTo(stringVector); return stringVector; };
	/*
	 inline vector<int> getIntVector() const {
	 vector<int> v;
	 splitTo(v);
	 return v;
	 };
	 */

	inline const vector<Data> & getVector() const {
		splitTo(dataVector);
		return dataVector;
	}
	;

	//type_info type;

protected:
	void trim(string &s) const;

	const type_info *type;


	// experimental
	mutable vector<Data> dataVector;
	//vector<int> intVector;
	//mutable vector<float> floatVector;
	//mutable vector<string> stringVector;

};

template<class T>
Data &Data::operator =(const T &x) {
	std::stringstream sstr;
	sstr << x;
	//if (lenient  || !sstr.fail())
	this->assign(sstr.str());
	//if (type == &typeid(void))
	if (!typeIsSet())
		setType<T>();
	return (*this);
}

template<class T>
Data &Data::operator <<(const T &x) {
	std::stringstream sstr;
	if (!empty() && !separators.empty()){
		sstr << separators[0];
	}
	else
		setType<T>(); // not sure about this
	sstr << x;
	this->append(sstr.str());
	return (*this);
}

template<class T>
Data::operator T() const {

	if (empty())
		return T();

	T x;
	stringstream sstr;
	sstr << (*this);
	sstr >> x;

	/// If conversion fails, assigns default value.
	if (!sstr.fail()) // if (lenient  || !sstr.fail())
		return x;
	else
		return T();
}

/// Can be used to check the type of current data.
/*
template<class T>
bool Data::isType() const {
	T x;
	stringstream sstr;
	sstr.precision(2);
	sstr << (*this);
	sstr >> x;

	 cout << "" << x << "\t";
	 cout << sstr.good() << sstr.bad() << sstr.fail() << sstr.eof();
	 cout << '=' << sstr.rdstate() << "\n";
	//cout << "flags=" << sstr.flags() << "\n";
	// a good combination seems to be:
	// !sstr.fail() = read successful, but perhaps something remaing
	//  sstr.eof()  = full success
	return !sstr.fail() && sstr.eof();
}
*/
/*
 drain::Data d;
 d = "keo";
 d.isType<double>();
 d.isType<int>();
 d.isType<bool>();
 d.isType<string>();

 d = "123";
 d.isType<double>();
 d.isType<int>();
 d.isType<bool>();
 d.isType<string>();

 d = "123uwuq";
 d.isType<double>();
 d.isType<int>();
 d.isType<bool>();
 d.isType<string>();

 d = "123.456";
 d.isType<double>();
 d.isType<int>();
 d.isType<bool>();
 d.isType<string>();
 */

/// " 2,3,4,5 "
/// " 2,3,4,,5 "
/// "      "
/// "  ,,, "
/// ",,,,,,"
template<class T>
void Data::splitTo(list<T> &l, const string &sep) const {

	const string & separators = sep.empty() ? this->separators : sep;

	const string::size_type n = size();

	l.clear();
	if (n == 0)
		return;

	// koe
	string::size_type pos = 0;
	string::size_type pos2 = 0;

	Data x;
	do {

		pos2 = find_first_not_of(trimChars, pos);
		pos2 = find_first_of(separators, pos2);
		if (pos2 == string::npos)
			pos2 = n; // last

		// Pick string segment, notice +1 missing because separator
		x = substr(pos, pos2 - pos);
		trim(x);
		l.push_back(x);
		pos = pos2 + 1;
	} while (pos2 != n);

}

template<class T>
void Data::splitTo(vector<T> &v, const string & sep) const {

	const string & separators = sep.empty() ? this->separators : sep;

	list<T> l;
	splitTo(l, separators);

	v.resize(l.size());

	typename list<T>::const_iterator li = l.begin();
	typename vector<T>::iterator vi = v.begin();
	while (li != l.end()) {
		//cerr << " split " << *li << endl;
		*vi = *li;
		vi++;
		li++;
	}

}

}

#endif /* DATA_H_ */
