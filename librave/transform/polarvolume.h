/**
 * Defines the functions available when working with polar volumes
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-14
 */
#ifndef POLARVOLUME_H
#define POLARVOLUME_H

/**
 * Represents one scan in a volume.
 */
typedef struct {
  // Where
  double elangle; /**< elevation of scan */
  long nbins; /**< number of bins */
  double rscale; /**< scale */
  long nrays; /**< number of rays / scan */
  double rstart; /**< start of ray */
  long a1gate; /**< something */

  // What
  char quantity[64]; /**< what does this data represent */
  double gain; /**< gain when scaling */
  double offset; /**< offset when scaling */
  double nodata; /**< nodata */
  double undetect; /**< undetect */
} PolarScan_t;

/**
 * Represents a volume
 */
typedef struct {
  double lon; /**< longitude of the radar that this volume originated from */
  double lat; /**< latitude of the radar that this volume originated from */
  double height; /**< altitude of the radar that this volume originated from */

  int nrAllocatedScans; /**< Number of scans that the volume currently can hold */
  int nrScans; /**< The number of scans that this volume is defined by */
  PolarScan_t** scans; /**< the scans that this volume is defined by */
} PolarVolume_t;

/**
 * Adds a scan to the volume.
 * @param[in] volume - the volume
 * @param[in] scan - the scan
 * ®return 0 on failure, otherwise success
 */
int PolarVolume_Add(PolarVolume_t* volume, PolarScan_t* scan);

#endif
