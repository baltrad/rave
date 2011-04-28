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

#ifndef __RACKIF__
#define __RACKIF__

#include <rave_object.h>

/*!  This file is an interface for running Rack from python.
 *
 */

/**
 * Call Rack with a polar volume/scan
 * @param[in] obj - The core object which is a polar volume or scan
 * @param[in] argstr - string with arguments
 * @return A core object which is a polar volume or polar scan
 */
#ifdef __cplusplus
extern "C"
#endif
RaveCoreObject* execRack(RaveCoreObject* obj, const char *argstr);

#endif //__RACKIF__
