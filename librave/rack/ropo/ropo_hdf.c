#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <rave_io.h>
#include <polarvolume.h>
#include <polarscan.h>
#include <fmi_image.h>
#include <fmi_radar_image.h>

#include "ropo_hdf.h"

int
is_hdf_file(char *filename)
{
  FILE *fp;
  char header[5];
  fp = fopen(filename,"r");
  assert(fp!=NULL);
  
  fread(header, sizeof(char), 4, fp);

  fclose(fp);

  header[4] = '\0';  
  return (strcmp(header+1, "HDF") == 0);

}

/* Read a scan from the Rave polarvolume and put it into an FmiImage.
 */

int 
read_scan(RaveCoreObject *rave_pvol, FmiImage *pscan, int index)
{
  RaveCoreObject *rave_scan;
  double value;
  int i,j;
  rave_scan=(RaveCoreObject *)PolarVolume_getScan((PolarVolume_t *)rave_pvol, index);

  pscan->width=PolarScan_getNbins((PolarScan_t *)rave_scan);
  pscan->height=PolarScan_getNrays((PolarScan_t *)rave_scan);
  pscan->bin_depth=PolarScan_getRscale((PolarScan_t *)rave_scan);
  pscan->elevation_angle=PolarScan_getElangle((PolarScan_t *)rave_scan);
  pscan->max_value=255;
  pscan->channels=1;
  initialize_image(pscan);


  for (j=0;j<pscan->height;j++)
    for (i=0;i<pscan->width;i++)
      {
        PolarScan_getValue((PolarScan_t *)rave_scan, i, j, &value);
        put_pixel(pscan, i, j, 0, (Byte)(value+0.5));
      }
  return 1;
}
 
/* Convert PolarVolume_t to FmiImage.
 */

int
rave2ropo(PolarVolume_t *rave_pvol, FmiImage **pvol)
{
  int sweep_count, i;
  FmiImage * current_pvol = NULL;
  sweep_count=PolarVolume_getNumberOfScans(rave_pvol);
  PolarVolume_sortByElevations(rave_pvol, 1);
  
  current_pvol = *pvol;
  for (i=0;i<sweep_count;i++)
    {
      read_scan((RaveCoreObject *)rave_pvol,&current_pvol[1+i],i);
    }

  return 1;
}

/* Open a Odim radar file, read it using Rave, and convert it to Ropo
   format. 
 */

PolarVolume_t * read_h5_radar_data(char *pvol_file, FmiImage **target){
  PolarVolume_t *result = (PolarVolume_t *)RaveIO_getObject(RaveIO_open(pvol_file));
  rave2ropo(result, target);
  return result;
}


PolarScan_t * create_scan(FmiImage *input, int index){
  PolarScan_t* result = NULL;
  PolarScan_t* scan = NULL;
  PolarScanParam_t* parameter = NULL;
  int ray = 0;
  int bin = 0;
  FmiImage *current_input = &input[index + 1];
  scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
  if (scan == NULL) {
    goto error;
  }

  parameter = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
  if (parameter == NULL) {
    goto error;
  }
  PolarScan_setElangle(scan, fmi_radar_sweep_angles[index]);
  PolarScan_setRscale(scan, input->bin_depth);
  PolarScan_setRstart(scan, 0.0);

  if (!PolarScanParam_createData(parameter,
                                 current_input->width,
                                 current_input->height,
                                 RaveDataType_UCHAR)) {
    goto error;
  }

  PolarScan_addParameter(scan, parameter);

  for (ray = 0; ray < current_input->height; ray++) {
    for (bin = 0; bin < current_input->width; bin++) {
      PolarScan_setValue(scan, bin, ray, get_pixel(current_input, bin, ray, 0));
    }
  }
  result = RAVE_OBJECT_COPY(scan);
error:
  RAVE_OBJECT_RELEASE(parameter);
  RAVE_OBJECT_RELEASE(scan);
  return result;
}

int
fill_scan(FmiImage *input, PolarVolume_t *output, int index)
{
  PolarScan_t *scan = PolarVolume_getScan(output, index);
  int ray, bin;
  FmiImage *current_input = &input[index + 1];

  PolarScan_setElangle(scan, fmi_radar_sweep_angles[index]/180.0 * M_PI);
  PolarScan_setRscale(scan, input->bin_depth);
  PolarScan_setRstart(scan, 0.0);
  for (ray = 0; ray < current_input->height; ray++) 
    {
      for (bin = 0; bin < current_input->width; bin++) 
        {
          PolarScan_setValue(scan, bin, ray, get_pixel(current_input, bin, ray, 0));
        }
    }
  return 1;


}

int
ropo2rave(FmiImage *input, PolarVolume_t *output)
{
  int i = 0;

  PolarVolume_sortByElevations(output, 1);

  for(i=0; i < input->sweep_count; i++) {
    fill_scan(input, output, i);
  }
  return 1;
                
}


 
int
write_h5_radar_data(FmiImage *pvol, char *pvol_file, PolarVolume_t *template)
{
  RaveIO_t * raveio = NULL;
  raveio = RAVE_OBJECT_NEW(&RaveIO_TYPE);
  
  assert(raveio != NULL);
  ropo2rave(pvol, template);
  RaveIO_setObject(raveio, (RaveCoreObject*)template);
  RaveIO_save(raveio, pvol_file);
  return 1;
}
 
