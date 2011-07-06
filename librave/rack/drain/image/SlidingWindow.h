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
#ifndef SLIDINGWINDOW_H_
#define SLIDINGWINDOW_H_

#include "Window.h"

namespace drain
{

namespace image
{
	
template <class T=unsigned char,class T2=unsigned char>
class SlidingWindow : public Window<T,T2>
{
public:
	 SlidingWindow(unsigned int width=0,unsigned int height=0) {
	   this->setSize(width,height);
	   //value = 0;
	 };
	 	  
	 virtual ~SlidingWindow(){};
	 
	   
	   
	   /** Returns the x coordinate of the outgoing side
         *  @param dx - direction: -1 for left, +1 for right
         *  @return
         */
     inline int getXOld(const int &dx){
       return this->location.x + (dx>0 ? this->iMin-1 : this->iMax+1);
     }

        /** Returns the x coordinate of the incoming side
         *  @param dx - direction: -1 for left, +1 for right
         *  @return
         */
     inline int getXNew(const int &dx){
       return this->location.x + (dx>0 ? this->iMax     : this->iMin);
     }

        /** Returns the y coordinate of the outgoing side
         *  @param dx - direction: -1 for up, +1 for down
         *  @return
         */
      inline int getYOld(const int &dy){
                return this->location.y + (dy>0 ? this->jMin-1 : this->jMax+1);
        }

        /** Returns the y coordinate of the incoming side
         *  @param dx - direction: -1 for up, +1 for down
         *  @return
         */
       inline int getYNew(const int &dy){
             return this->location.y + (dy>0 ? this->jMax     : this->jMin);
       }

	   
	  /** Action to perform after each horizontal move.
        *
        * @param dx
        */
        virtual void updateHorz(int dx) = 0; //{ value = (dx>0)?32:128; };

        /** Action to perform when window is moved vertically.
         *
         * @param dx
         */
        virtual void updateVert(int dy) = 0; //{ value = (dy>0)?64:256; };
	
	   /** Moves one step down. Stop at edge.
         *
         *  @return true, if a legal move was made.
         */
        bool moveDown(){

                //coordinateOverflowHandler.yMin;
                if (++this->location.y <= this->src.getCoordinateHandler().yMax){
                        updateVert(+1);
                        return true;
                }
                else {
                        this->location.y--;
                        return false;
                }
        }

        bool moveUp(){
                if (--this->location.y >= 0){
                        updateVert(-1);
                        return true;
                }
                else {
                        this->location.y++;
                        return false;
                }
        }
 
 
 		bool moveRight(){
                if (++this->location.x <= this->src.getCoordinateHandler().xMax){
                        updateHorz(+1);
                        return true;
                }
                else {
                        this->location.x--;
                        return false;
                }
        }

        bool moveLeft(){
                if (--this->location.x >= 0){
                        updateHorz(-1);
                        return true;
                }
                else {
                        this->location.x++;
                        return false;
                }
        }
	
	  // Write the result in the target image. 
	  /** Sliding windows enjoys a confidence of the application,
	   *  as it has the    
	   */
	  virtual void write() = 0; //{this->dst.at(this->location) = this->value;};
	  
	protected:
	   // For computing legal coordinates inside image, using coordhandler
	   mutable Point2D<> p;
	   //int value;
};

}

}

#endif /*SLIDINGWINDOW_H_*/
