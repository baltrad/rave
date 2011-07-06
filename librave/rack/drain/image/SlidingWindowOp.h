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
#ifndef SLIDINGWINDOWOP_H_
#define SLIDINGWINDOWOP_H_

#include "CopyOp.h"
#include "WindowOp.h"
#include "SlidingWindow.h"

namespace drain
{

namespace image
{

template <class T=unsigned char,class T2=unsigned char>
class SlidingWindowOp : public WindowOp<T,T2>
{

public:

	// For inherited classes
	SlidingWindowOp(SlidingWindow<T,T2> & w,const string &name, const string &description,
				const string & parameterNames, const string & defaultValues) :
					WindowOp<T,T2>(name,description,parameterNames,defaultValues), window(w) {
	};

	SlidingWindowOp(SlidingWindow<T,T2> & w) :
		WindowOp<T,T2>("SlidingWindowOp","Fast implementation of SlidingWindowOp (base class).","width,height","0,0"), window(w) {
	};

	/*
	SlidingWindowOp(SlidingWindow<T,T2> & w,const string & parameterNames = "",
			const string & defaultValues = "") : window(w) {
		this->setInfo("desc",parameterNames,defaultValues);
	};
	*/

	virtual void initialize() const {
		int w = this->getParameter("width",window.width);
		int h = this->getParameter("height",window.height);
		window.setSize(w,h);	
	}


	// TODO: who has coordhandler?
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		initialize();

		Image<T2> tmp;
		const bool tmpNeeded = dst.hasOverlap(src);

		Image<T2> &d = tmpNeeded ? tmp : dst; 

		makeCompatible(src,d);
		window.setSrc(src);
		window.setDst(d);

		filter();
		if (tmpNeeded)
			CopyOp<T2,T2>().filter(tmp,dst);
	};

	// TODO: who has coordhandler?
	virtual void filter(const Image<T> &src,const Image<T> &srcWeight,
			Image<T2> &dst,Image<T2> &dstWeight) const {
		initialize();

		Image<T2> tmp;
		const bool tmpNeeded = dst.hasOverlap(src);

		Image<T2> tmpWeight;
		const bool tmpWeightNeeded = dstWeight.hasOverlap(srcWeight);

		Image<T2> &d = tmpNeeded ? tmp : dst; 
		makeCompatible(src,d);
		window.setSrc(src);
		window.setDst(d);

		Image<T2> &dw = tmpWeightNeeded ? tmpWeight : dstWeight; 
		makeCompatible(srcWeight,dw);
		window.setSrcWeight(srcWeight);
		window.setDstWeight(dw);


		//cerr << window << "\n";

		filter();

		if (tmpNeeded)
			CopyOp<T2,T2>().filter(tmp,dst);

		if (tmpWeightNeeded)
			CopyOp<T2,T2>().filter(tmpWeight,dstWeight);
	};




	void filter() const {

		window.initialize();
		window.write();

		if (window.width > window.height){

			// Horizontal window => Horizontal traversal
			while (true){

				//cerr << "slide" << window.location << endl;

				while (window.moveRight())
					window.write();

				if (!window.moveDown())
					return;

				window.write();

				while (window.moveLeft())
					window.write();

				if (!window.moveDown())
					return;

				window.write();

			}

		} else {

			// Vertical window => Vertical traversal
			while (true){

				while (window.moveDown())
					window.write();

				if (!window.moveRight())
					return;

				window.write();

				while (window.moveUp())
					window.write();

				if (!window.moveRight())
					return;

				window.write();

			}

		}
	}

protected:

	mutable SlidingWindow<T,T2> &window;


};

}

}

#endif /*SLIDINGWINDOWOP_H_*/
