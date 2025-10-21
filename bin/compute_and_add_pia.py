#!/usr/bin/env python
'''
Created on Mon Aug 25 14:12:08 2025

@author: Remco van de beek
@author: Yngve Einarsson SMHI
'''
# TBD: Copyright notice

import os
import _raveio

from rave_quality_plugin import QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY
from compute_pia import ComputePIA



## Computes PIA and performs quality control with it.
# @param object containing command-line arguments
def main(options):
    rio = _raveio.open(options.ifile)
    ComputePIA(rio.object, options.param_name, options.keepPIA,options.reprocess_quality_fields,
        QUALITY_CONTROL_MODE_ANALYZE)
    rio.save(options.ofile)

if __name__=="__main__":
    from optparse import OptionParser

    description = "Computes PIA and uses it to quality control radar observations"

    usage = "usage: %prog -i <input file> -o <output file> [-p <parameter> -k <store PIA >] [h]"

    parser = OptionParser(usage=usage, description=description)

    parser.add_option("-i", "--infile", dest="ifile",
                      help="Name of input file to quality control. Must contain a polar volume or scan with ZDR and RHOHV besides the parameter to QC.")

    parser.add_option("-o", "--outfile", dest="ofile",
                      help="Name of output file to write.")

    parser.add_option("-p", "--parameter", dest="param_name", default="DBZH",
                      help="ODIM quantity/parameter name to QC. Defaults to DBZH.")

    parser.add_option("-k", "--keepPIA", dest="keepPIA", default="store_true",
                      help="Whether (True) or not (False) to keep and store the PIA parameter. Defaults to True.")

    parser.add_option(
        "--reprocess_quality_fields",
        action="store_true",
        dest="reprocess_quality_fields",
        help="Reprocessed the quality fields even if they already exist in the object.",
    )

    parser.add_option(
        "--quality_control_mode",
        dest="quality_control_mode",
        default=QUALITY_CONTROL_MODE_ANALYZE,
        help="Quality_control_mode, default QUALITY_CONTROL_MODE_ANALYZE, QUALITY_CONTROL_MODE_ANALYZE_AND_APPLY ."
    )

    (options, args) = parser.parse_args()

    if not options.ifile or not options.ofile:
        parser.print_help()
        sys.exit()


    main(options)
