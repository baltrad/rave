'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''
## Bogus product generator used to debug registries

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-20


def debugme(files, args):
    import rave

    if args[0] == 'outfile': outfile = args[1]

    this = rave.open(files[0])
    this.save(outfile)



if __name__ == "__main__":
    print __doc__
