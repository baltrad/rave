#!/usr/bin/env python
'''
Copyright (C) 2012- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''

## Odyssey production using the BALTRAD toolbox

## @file
## @author Daniel Michelson, SMHI
## @date 2012-11-05
# Standard python libs:
import sys
import os
import glob
import time
import logging
import multiprocessing

# Module/Project:
import odc_polarQC
import rave_pgf_logger
import _rave, _raveio
import _pycomposite
import compositing, tiled_compositing


## Coordinates all data processing
# @param options object passed directly from the parsed command-line arguments
def generate(options):
    start = time.time()
    if options.ipath == options.opath:
        raise NameError("Input and output paths may not be the same.")
    if not os.path.isdir(options.ipath):
        raise IOError("Input path does not exist.")
    elif not (os.access(options.ipath, os.R_OK) and os.access(options.ipath, os.W_OK)):
        raise IOError("Input path exists but you lack read/write permission.")
    
    if not os.path.isdir(options.opath):
        try:
            os.makedirs(options.opath)
        except:
            raise IOError("Cannot create output directory.")
    elif not os.access(options.opath, os.W_OK):
        raise IOError("Output directory exists but you lack write permission.")
    
    fstrs = glob.glob(options.ipath + '/*')
    if not len(fstrs):
        raise IOError("Empty input directory? Exiting.")
    
    # Initialize logger
    logger = logging.getLogger("ODC")
    rave_pgf_logger.init_logger(logger)
    
    # Compositing includes QC. Therefore do not do QC separately. This composite config is hard wired.
    if options.areaid:
        comp = compositing.compositing()
        comp.igore_malfunc = True
        comp.product = _rave.Rave_ProductType_MAX
        comp.selection_method = _pycomposite.SelectionMethod_HEIGHT
        comp.detectors = options.qc.split(',')  # E.g. ["ropo","beamb","radvol-broad","qi-total"]
        comp.qitotal_field = "pl.imgw.quality.qi_total"  # qi-total must be in comp.detectors !!
        comp.gain = 0.5
        comp.offset = -32.0
        comp.filenames = fstrs
        tc = tiled_compositing.tiled_compositing(comp)
        if options.dump:  # Provisional, until compositing can handle prefab QC
            tc.compositing.opath = options.opath
            tc.compositing.dump = True
        t = tc.generate(None, None, options.areaid)
        rio = _raveio.new()
        rio.object = t
        rio.save(os.path.join(options.opath, options.ofile))
        after = time.time()
        logger.info("odc_area tiled composite: %3.1f sec using %i PVOLs" % ((after - start), len(fstrs)))
    
    else:
        odc_polarQC.opath = options.opath
        odc_polarQC.algorithm_ids = options.qc.split(',')
        odc_polarQC.delete = options.delete
        odc_polarQC.check = options.check
        
        results = odc_polarQC.multi_generate(fstrs, options.procs)
        
        # Log benchmarking results.
        allreads, allvalids, allqcs, allwrites = 0.0, 0.0, 0.0, 0.0
        n = 0  # counter for number of successfully processed files
        exists = 0  # counter for files ignored, already processed
        for result in results:
            if len(result) == 3:
                ifstr, msg, (readt, validt, qct, writet) = result
                allreads += readt
                allvalids += validt
                allqcs += qct
                allwrites += writet
                n += 1
            if result[1] == "EXISTS":
                exists += 1
        
        if not options.procs:
            options.procs = multiprocessing.cpu_count()
        
        totalt = allreads + allvalids + allqcs + allwrites
        if totalt > 0.0:
            readt = allreads / totalt * 100
            validt = allvalids / totalt * 100
            qct = allqcs / totalt * 100
            writet = allwrites / totalt * 100
            runt = time.time() - start
    
            logger.info("Processed %i of %i files in %2.1f (%2.1f) s using %i workers. Breakdown: %2.1f%% read, %2.1f%% validation, %2.1f%% QC, %2.1f%% write. Ignored %i files already processed." % (n, len(fstrs), runt, totalt, options.procs, readt, validt, qct, writet, exists))
        elif exists:
            logger.info("Ignored %i files already processed." % exists)
        else:
            logger.info("No statistics")


if __name__ == "__main__":
    pass
