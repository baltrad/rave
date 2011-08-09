/**

    Copyright 2001 - 2010  Markus Peura, 
    Finnish Meteorological Institute (First.Last@fmi.fi)


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
    along with Rack.  If not, see <http://www.gnu.org/licenses/>. */

#include "fmi_image_restore.h"
#include "fmi_image_filter.h"
#include "fmi_image_histogram.h"


void mark_image(FmiImage *target,FmiImage *prob,Byte threshold,Byte marker){ 
  register int i;
  check_image_properties(target,prob);

  for (i=0;i<prob->volume;i++)
    if (prob->array[i]>=threshold)
      target->array[i]=marker;
}


/* simple */
void restore_image(FmiImage *source,FmiImage *target,FmiImage *prob,Byte threshold){ 
  register int i;
  canonize_image(source,prob);
  canonize_image(source,target);

  for (i=0;i<prob->volume;i++)
    if (prob->array[i]>=threshold)
      target->array[i]=0;
    else
      target->array[i]=source->array[i];
}

void restore_image_neg(FmiImage *source,FmiImage *target,FmiImage *prob,Byte threshold){ 
  register int i;
  canonize_image(source,prob);
  canonize_image(source,target);

  for (i=0;i<prob->volume;i++)
    if (prob->array[i]<threshold)
      target->array[i]=0;
    else
      target->array[i]=source->array[i];
}

/* other */
void restore_image2(FmiImage *source,FmiImage *target,FmiImage *prob,Byte threshold){ 
  register int i;
  FmiImage median;
  canonize_image(source,prob);
  canonize_image(source,target);
  canonize_image(source,&median);

  /* ERASE ANOMALIES (to black) */
  for (i=0;i<prob->volume;i++)
    if (prob->array[i]>=threshold)
      target->array[i]=0;
    else
      target->array[i]=source->array[i];

  /* CALCULATE ME(DI)AN OF NONZERO PIXELS */
  pipeline_process(target,&median,2,2,histogram_mean_nonzero);

  /* REPLACE ANOMALOUS PIXELS WITH THAT NEIGHBORHOOD ME(DI)AN */
  for (i=0;i<prob->volume;i++)
    if (prob->array[i]>=threshold)
      target->array[i]=median.array[i];
}
