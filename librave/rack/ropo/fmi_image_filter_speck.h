/**

    Copyright 2001 - 2010  Markus Peura,
    Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Rack.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

*/


/* THIS LIBRARY CONTAINS IMAGE PROCESSING OPERATIONS FOR GENERAL PURPOSE */

/* remove small bright specks  */
/* Remove specks with area up to A pixels    */
void detect_specks(FmiImage *target,FmiImage *trace,unsigned char min_value,int (* histogram_function)(Histogram));
void Binaryprobe(FmiImage *domain,FmiImage *source,FmiImage *target,int (* histogram_function)(Histogram),unsigned char min_value);
//void remove_specks(FmiImage *img,Byte min_intensity,int max_property,Byte marker,int (* histogram_function)(Histogram));

/* debugging and development */
void test_rotation(FmiImage *target,FmiImage *trace,int i,int j,int rec_depth);
//int hselector;

int ROTX(int dir);
int ROTY(int dir);
int ROT_CODE(int i,int j);



