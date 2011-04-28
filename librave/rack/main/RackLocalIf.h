/**


    Copyright 2006 - 2011

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

   Created on: Apr 13, 2011
*/

#ifndef __RACKLOCALIF__
#define __RACKLOCALIF__


/*!  This file is a local interface for Rack used by the main function
 *   called from command line.
 */

/**
 * Call Rack with a polar volume
 * @param[in] volume - The polar volume
 * @param[in] argv - vector with with arguments
 * @return A Polar volume
 */
PolarVolume_t* runRack(PolarVolume_t* volume, std::vector<string>& args);


/**
 * Call Rack with a polar scan
 * @param[in] scan - The polar scan
 * @param[in] argv - vector with with arguments
 * @return A Polar scan
 */
PolarScan_t* runRack(PolarScan_t* scan, std::vector<string>& args);

#endif //__RACKLOCALIF__
