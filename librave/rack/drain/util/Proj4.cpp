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
#include "Proj4.h"

namespace drain
{

Proj4::Proj4() : projSrc(NULL), projDst(NULL)
{
	//projSrc = pj_init_plus("");
	//pj_init_plus("+proj=merc +ellps=clrk66 +lat_ts=33");
	//projDst = pj_init_plus("+proj=eqc +lon 0=90w");	
}

Proj4::~Proj4()
{
	//cerr << "~Proj4()"  << endl;
	pj_free( projSrc );
	pj_free( projDst  );
}

void Proj4::setProjection(const string &str)
{ 
	pj_free( projDst );
	projDst = pj_init_plus(str.c_str());

	/*
	if (projDst != NULL){
		projStr = pj_get_def( projDst,0);
		ok = true;
	}
	else 
	{
		projStr += '\n';
		projStr += "Proj4 error: ";
		projStr += pj_strerrno(*pj_get_errno_ref());
		ok = false;
	}
	*/
	
}




}
