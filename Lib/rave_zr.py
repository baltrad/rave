#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# $Id: rave_zr.py,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
#
# Author: Daniel Michelson
#
# Copyright (c): Swedish Meteorological and Hydrological Institute
#                2006-
#                All rights reserved.
#
# $Log: rave_zr.py,v $
# Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
# Project added under CVS
#
#
"""
rave_zr.py

Helper functions for converting between Z, R and everything in between.
"""

# Third-party:
from numpy import power, log10


def dBZ2Z(dbz):
    return power(10.0, (dbz / 10.0))


def dBZ2R(dbz, A, b):
    pow_v = 1.0 / b
    Z = dBZ2Z(dbz)
    Z /= A
    return power(Z, pow_v)


def Z2dBZ(Z):
    return 10 * log10(Z)


def Z2R(dbz, A, b):
    pow_v = 1.0 / b
    reallyZ = power(10.0, (dbz / 10.0))
    reallyZ /= A
    return power(reallyZ, pow_v)


def R2dBZ(R, A, b):
    Z = A * power(R, b)
    return Z2dBZ(Z)


def raw2dbz(raw, gain, offset):
    return gain * raw + offset


def dbz2raw(dbz, gain, offset):
    raw = 0.0
    if dbz <= offset:
        return raw

    raw = (dbz - offset) / gain
    if raw > 254.0:
        raw = 254.0

    return raw


def raw2R(raw, gain, offset, A, b):
    dbz = raw2dbz(raw, gain, offset)
    return dBZ2R(dbz, A, b)


def R2raw(R, gain, offset, A, b):
    dbz = R2dBZ(R, A, b)
    return dbz2raw(dbz, gain, offset)


__all__ = ["dBZ2Z", "dBZ2R", "Z2dBZ", "Z2R", "R2dBZ", "raw2dbz", "dbz2raw", "raw2R", "R2raw"]

if __name__ == "__main__":
    print(__doc__)
