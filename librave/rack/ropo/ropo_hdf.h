#ifndef _ROPO_HDF_H_
#define _ROPO_HDF_H_

/* Checks if a file is of hdf format.
 */

extern int is_hdf_file(char *filename);

/* Reads radar data from an odim hdf5 file using rave, and fills in a ropo
   FmiImage object. The returned rave object can be used to fill in the missing
   values when converting back from ropo to rave.
 */
extern PolarVolume_t *read_h5_radar_data(char *pvol_file, FmiImage **target);

/* Writes the radar data to file, in odim hdf5 format using rave. Uses a ropo
   object as data source, and a rave template to fill the missing data. A good
   choice for the template would be the originating PolarVolume_t object
   returned by read_h5_radar_data.
 */

extern int write_h5_radar_data(FmiImage *source, char *pvol_file, 
                               PolarVolume_t *template);

#endif /* _ROPO_HDF_H_ */
