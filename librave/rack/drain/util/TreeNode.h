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

 * Tree.h
 *
 *  Created on: Nov 25, 2010
 *      Author: mpeura
 */

#ifndef TREE_H_
#define TREE_H_

#include <iterator>
#include <string>
#include <map>

#include "String.h"
#include "RegExp.h"
#include "Options.h"

namespace drain {

using namespace std;

template <class T>
class TreeNode;

/// Unordered tree.
/**
 *  Child nodes can be addressed with operator[] with string-valued keys:
 *  \code
 *  TreeNode<float> t;
 *  t["firstChild"];
 *  \endcode
 *  The keys can be retrieved to a list with getKeys().
 *
 *  Alternatively, one may iterate the children directly:
 *  \code
 *  map<string,TreeNode<int> >::iterator it = tree.begin();
 *  while (it != tree.end()){
 *    cout << it->first << ' ' << it->second.value << '\n';
 *    ++it;
 *  }
 *  \endcode
 *
 *  As the operator[] returns a reference, it may be used recursively:
 *  \code
 *  t["firstChild"]["a"]["bb"];
 *  \endcode
 *
 *  Tree also recognizes a path separator. The node created in the above example could be likewise
 *  created or addresses as:
 *  \code
 *  t["firstChild/a/bb"];
 *  \endcode
 *
 *  Each node contains data of type T. It may be unused. There are no setData() or getData() members,
 *  but direct assignments to and from the node data instead:
 *  \code
 *  TreeNode<string> t;
 *  t = "Hello world";
 *  string s = t;  // Now contains "Hello world".
 *  \endcode
 *
 *  Also direct referencing of the node data is supported:
 *  \code
 *  TreeNode<string> t;
 *  t = "Hello world";
 *  string &s = t;  // Refers to data, string("Hello world").
 *  \endcode
 *
 */
template <class T>
class TreeNode {
public:

	/// Default constructor.
	TreeNode(char separator = '/') : separator(separator) {};

	/// Copy constructor
	TreeNode(const TreeNode &t) : value(t.value), separator(t.separator)  {};

	/// Contents (data) of the node.
	T value;

	/// Assigns value to contents.
	inline
	TreeNode &operator=(const T &v){
		value = v;
		return *this;
	};

	/// Assigns a value to contents.
	template <class T2>
	inline
	TreeNode &operator=(const T2 &v){
		value = v;
		return *this;
	};

	/// Copies the data of another node.
	// TODO: what to do with the separator?
	inline
	TreeNode &operator=(const TreeNode &t){
		value = t.value;
		//separator = t.separator;
		return *this;
	};

	/// Returns a copy of the data of a node.
	inline
	operator T() const {
		return value;
	};

	/// Returns the reference to the contents of a node.
	inline
	operator T &(){
		return value;
	};

	/// Returns the reference to the contents of a node.
	inline
	operator const T &() const {
			return value;
	};



	/// Child iterator pointing to the first child.
	inline
	typename map<string,TreeNode<T> >::const_iterator begin() const { return children.begin(); };

	/// Child iterator end.
	inline
	typename map<string,TreeNode<T> >::const_iterator end() const { return children.end(); };

	/// Child iterator pointing to the first child.
	inline
	typename map<string,TreeNode<T> >::iterator begin(){ return children.begin(); };

	/// Child iterator end.
	inline
	typename map<string,TreeNode<T> >::iterator end(){ return children.end(); };


	/// Returns a child.
	inline
	TreeNode<T> &operator[](const char *key){
		return operator[](string(key));
	}

	/// Returns a child.
	inline
	TreeNode<T> &operator[](const string &key){

		// Self-reference
		if (key.length() == 0)
			return *this;

		// Skip separator at root
		if (key.at(0) == separator)
			return (*this)[key.substr(1)];

		const size_t i = key.find(separator);

		// Leaves
		if (i == string::npos)
			return getChild(key);
		// Subtrees
		else
			return getChild(key.substr(0,i))[key.substr(i+1)];

	};

	/// Debugging utility - dumps the structure of the tree (not the contents).
	void dump(ostream &ostr = std::cout,int depth = 0) const {
		//if (depth==0)
		ostr << ' ' << depth << '\n';
		for (typename map<string,TreeNode<T> >::const_iterator it = begin(); it != end(); it++){
			for (int i = 0; i < depth; ++i)
				ostr << "  ";
			ostr << "¦--";
			ostr << it-> first; // << '\n'; //' ' << depth << '\n';
			it->second.dump(ostr,depth+1);
		};
	};

	/// Debugging utility - dumps the structure of the tree (not the contents).
	void dumpContents(ostream &ostr = std::cout,int depth = 0) const {
			//if (depth==0)
			ostr << ' ' << value << '\n';
			for (typename map<string,TreeNode<T> >::const_iterator it = begin(); it != end(); it++){
				//ostr << "¦";
				for (int i = 0; i < depth; ++i)
					ostr << "  ";
				ostr << "¦--";
				ostr << it->first;
				//ostr << it-> first << '\n'; //' ' << depth << '\n';
				it->second.dumpContents(ostr,depth+1);
			};
		};

	/// Returns the node names matching a pattern.
	/*  Following wildcards can be used:
	 *  - ?  - matches any single character, excluding the path separator
	 *  - *  - matches any string not containing the path separator
	 *  - ** - matches any string
	 */
	template <class L>
	void getKeys(L & list,const string &filter = "") const {

		if (filter.empty())
			return getKeys(list,RegExp());

		string s;
		s += "^";
		s += filter;  //this->getParameter("data","/**/data");
		s += "$";

		const string nonSeparator = string("[^\\")+separator+string("]");

		s = String::replace(s,"**","#"); // tmp marker
		s = String::replace(s,"?",nonSeparator+"?");
		s = String::replace(s,"*",nonSeparator+"*");
		s = String::replace(s,"#",".*"); // tmp marker

		return getKeys(list,RegExp(s));
	};

	/// Returns a list of the node names matching a pattern.
	template <class L>
	void getKeys(L & list,const RegExp &r,const string &path = "") const {
		for (typename map<string,TreeNode<T> >::const_iterator it = begin(); it != end(); it++){
			string p = path + separator + it->first;
			if (r.test(it->first))
				list.push_back(p);
			it->second.getKeys(list,r,p);
		};
	};


	//char separator;

protected:

	char separator;

	map<string,TreeNode<T> > children;

	TreeNode<T> &getChild(const string &key){

		const typename map<string,TreeNode<T> >::iterator it = children.find(key);

		if (it != children.end()){
			return it->second;
		}
		else {
			children[key].separator = separator;
			return children[key];
		}

	};


	//static char separator;
	// cooperator[]
};

//template <class T>
//char TreeNode<T>::separator('/');

}

#endif /* TREE_H_ */
