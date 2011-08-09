#define FMI_ROPO_VER "ropo\t\t v1.32\t Jan 2003 (c) Markus.Peura@fmi.fi"
#include <stdio.h>
#include <assert.h>
#include <fmi_util.h>
#include <fmi_image.h>
#include <fmi_image_filter.h>
#include <fmi_image_filter_line.h>
#include <fmi_image_histogram.h>
#include <fmi_image_filter_speck.h>
#include <fmi_image_filter_morpho.h>
#include <fmi_image_restore.h>
#include <fmi_meteosat.h>
#include <fmi_radar_image.h>

#include <polarvolume.h>
#include <ropo_hdf.h>
PolarVolume_t * template;

Byte LAST_FILTER;

void
help(){
  #include "main_help.c"
}

void update_classification(FmiImage *mark,FmiImage *prob,FmiImage *master_prob,Byte marker){
  register int i;
  for (i=0;i<mark->volume;i++)
    if (prob->array[i]>=master_prob->array[i]){
      mark->array[i]=marker;
      master_prob->array[i]=prob->array[i];}
}     

void generate_samples(char **argv,int *i,FmiImage *volume,int sweep_count){
  int bin,j,x,y,alpha,radius;

  setup_context(volume);
  switch (argv[*i][2]){
    /* vertical profile */
  case 'V':
    switch (argv[*i][3]){

    case 'c':
      /* polar coords */
    case 'p':
      if (argv[*i][3]=='p'){
	alpha=FMI_IARG(++(*i));
	radius=FMI_IARG(++(*i));
      } 
      else {
	x=FMI_IARG(++*i);
	y=FMI_IARG(++*i);
	xy_to_polar(x,y,&alpha,&radius);
      }

      printf("#% .3d % .3d\n",alpha,radius);
      for (j=0;j<sweep_count;j++){
	bin=bin_to_bin(radius,0,fmi_radar_sweep_angles[j]);
	if (bin<=500){
	  printf("%d\t",bin_to_altitude(bin,fmi_radar_sweep_angles[j]));
	  if (argv[*i][4]=='b')
	    printf("%d\n",get_pixel(&volume[1+j],bin,alpha,0));
	  else
	    printf("%d\n",byte_to_abs_dbz(get_pixel(&volume[1+j],bin,alpha,0)));}
      }
      return;
    default:
      fmi_error("-SV? - unknown switch");
    }
    break;
  }
}


int read_radar_data(int files,char **filenames,FmiImage *target){
  int i,sweep_count;
  /* RADAR IMAGE INPUT 
     sweeps:
     sweep1.pgm [sweep2.pgm ... sweepN.pgm] */
  if (files>1){
    sweep_count=files;
    for (i=0;i<sweep_count;i++){
      if (FMI_DEBUG(2))
	printf(" input PPI #%d [%.1f°]\n",i+1,fmi_radar_sweep_angles[i]);
      target[1+i].channels=0;
      initialize_image(&target[1+i]);
      read_image(filenames[i],&target[1+i]);
      target[1+i].bin_depth = FMI_RADAR_BIN_DEPTH;
      target[1+i].elevation_angle = fmi_radar_sweep_angles[i];
    }
    concatenate_images_vert(&target[1],sweep_count,target);
    for (i=0;i<sweep_count;i++)
      reset_image(&target[1+i]); /* = free array */
  }
  /* target:
     target.pgm */
  if (files==1 && !is_hdf_file(filenames[0])){
    target[0].channels=0;
    initialize_image(target);
    read_image(filenames[0],target);
    /* TRUE (MULTI SWEEP?) TARGET */
    /*
      Can't be done if variable ray counts are allowed !
      martin.raspaud@NOSPAM.smhi.se, Fri Aug  5 09:04:07 2011.
    */
    sweep_count = 1;
    target[0].sweep_count = sweep_count;
    target[0].heights = (int *)malloc(sizeof(int) * sweep_count);
    target[0].heights[0] = target[0].height;
    target[0].bin_depth = FMI_RADAR_BIN_DEPTH;
    target[1].bin_depth = FMI_RADAR_BIN_DEPTH;

  } else if (files==1) {
    template = read_h5_radar_data(filenames[0], &target);
    sweep_count=PolarVolume_getNumberOfScans(template);
    concatenate_images_vert(&target[1],sweep_count,target);
  }
  
  if(target->bin_depth == 0.0)
    target->bin_depth = FMI_RADAR_BIN_DEPTH;

  split_to_link_array(target,sweep_count,&target[1]);
  return sweep_count;
}

void split_command_line(char *line){
  char *ptr=line;
  if (!ptr)
    return;
  while ((ptr=strchr(ptr,' '))){
    ptr++;
    if (*ptr=='-'){
      ptr++;
      if (((*ptr>'a')&&(*ptr<'z')) || ((*ptr>'A')&&(*ptr<'Z'))){
	ptr--;
	ptr--;
	*ptr='\n';
	ptr++;
	ptr++; 
      }
    }
  }
}

int
main(int argc,char **argv){
  int i,files,sweep_count;
  int intensity,size,weight,length,height,width,radius,azimuth,elevation,altitude;
  int intensity_delta,altitude_delta;
  char *FILTER_CODE_TOTAL="all";  /* const! */
  char *filename,
    *filename_base="ropo",
    *LAST_FILTER_NAME="nofilt",
    *FILTER_CODE=FILTER_CODE_TOTAL,
    *PRODUCT_CODE="noop",
    tempfilename[1024];
  FmiImage cart;
  FmiImage *ppm;
  FmiImage *volume;
  FmiImage *volume_V;
  FmiImage *volume_filtered;
  FmiImage *prob;
  FmiImage *master_prob;
  FmiImage *mark;
  FmiImage meteosat;      
  Celsius t1,t2;
  FmiImage *image_ptr; 
  int argv_start;
  int prob_threshold;
  int s;
  
  ppm = new_image(2);
  image_ptr = new_image(FMI_RADAR_SWEEP_COUNT+1);
  volume = new_image(FMI_RADAR_SWEEP_COUNT+1);
  volume_V = new_image(FMI_RADAR_SWEEP_COUNT+1);
  volume_filtered = new_image(FMI_RADAR_SWEEP_COUNT+1);
  prob = new_image(FMI_RADAR_SWEEP_COUNT+1);
  master_prob = new_image(FMI_RADAR_SWEEP_COUNT+1);
  mark = new_image(FMI_RADAR_SWEEP_COUNT+1);

  extract_command_line(argc,argv);
  split_command_line(fmi_util_command_line);
  fmi_util_comment=FMI_RADAR_IMAGE_VER;
  /* GLOBAL SETTING, BUT USEFUL ONLY FOR B_SCAN IMAGES */

  /*coord_overflow_handler_x:  


    prob_threshold_online=256;


    prob_images=FMI_FLAG("-Oprobs"); */

  if (FMI_HELP(2)){
    printf("%s - radar image filtering program \n",argv[0]);
    printf ("\t%s\n",FMI_ROPO_VER);
    printf ("\t%s\n",FMI_IMAGE_VER);
    printf ("\t%s\n",FMI_RADAR_IMAGE_VER);
    printf("USAGE:\n %s radar_image[s] processing_and_output_options  \n",argv[0]);
    help();
    printf("\n");
    exit(0);}

  fmi_debug_level=1;

  files=file_count(argc,argv);

  if (files>FMI_RADAR_SWEEP_COUNT) 
    fmi_error("More input images than FMI_RADAR_SWEEP_COUNT");

  sweep_count=read_radar_data(files,&argv[1],volume);

  if (sweep_count>FMI_RADAR_SWEEP_COUNT) 
    fmi_error("More sweeps than FMI_RADAR_SWEEP_COUNT");

  /* INTERNAL IMAGES & DATA STRUCTURES:
       treats radar data as both sweeps and volumes
         volume     - input image:
         volume[0]  - input image (concatenated raw volume)
         volume[1]  - input image, 1st sweep (linked to volume[1])
         volume[2]  - input image, 2nd sweep (and so on)
  */
  /* internal auxiliary images (linked like 'volume'):
     volume_filtered - resulting image */

  /* SELVITÄ MIKSI copy ei tee canonizea */
  canonize_image(volume,volume_filtered);
  copy_image(volume,volume_filtered);
  split_to_link_array(volume_filtered,sweep_count,&volume_filtered[1]);

  /*   prob            - 'current' probability */
  canonize_image(volume,prob);
  split_to_link_array(prob,sweep_count,&prob[1]);

  /*   master_prob     - master probability */
  canonize_image(volume,master_prob);
  fill_image(master_prob,16);  
  split_to_link_array(master_prob,sweep_count,&master_prob[1]);

  /*  marked image, volume */
  copy_image(volume,mark);
  split_to_link_array(mark,sweep_count,&mark[1]);


  /* MAIN LOOP */
#define UPDATE_CLASSIFICATION update_classification(&mark[s],&prob[s],&master_prob[s],LAST_FILTER)

  fmi_debug(4,"main loop");
  argv_start=files+1;
  LAST_FILTER=CLEAR;
  s=1;
  i=argv_start;

  while (i<argc){
    argv_start=i;
    
    if (param(argc,argv,i,"-debug")){
      fmi_debug_level=FMI_IARG(++i);
      i++;}

    if (param(argc,argv,i,"-comment")){
      fmi_util_comment=argv[++i];
      i++;}

    /* SWEEP SELECTION (sweep to be processed/output, one (lowest) by default */
    if (param(argc,argv,i,"-volume")){
      s=0;
      i++;
    }
    /*  -sweep s */
    if (param(argc,argv,i,"-sweep")){
      s=atoi(argv[++i]);
      fill_image(prob,0);
      if (s<1) 
	fmi_error("too small -sweep");
      if (s>sweep_count) 
	fmi_error("too large -sweep");
      i++;
    }
    if (FMI_DEBUG(2))
      if (s==0)
	fprintf(stderr,"sweep=%d\n",s);

    if (param(argc,argv,i,"-threshold")){
      intensity=atoi(argv[++i]);
      threshold_image(&volume[s],&volume[s],intensity);
      i++;}

    if (param(argc,argv,i,"-Grad")){
      gradient_rgb(&volume[0]);
      i++;
    }

    /*  -cappi <ALTITUDE>   compute intersection (output to current sweep)
        -cappi 500m */
    if (param(argc,argv,i,"-cappi")){
      height=FMI_IARG(++i);
      volume_to_cappi(volume,height,&volume_filtered[s]);
      i++;
    }

    /*/ROPO DETECTORS (or 'filters') */
    /*/ <OPTIONS>  capital   letters <=> strict values */
    /*/ <options>  lowercase letters <=> fuzzy  values */

    /* LAST_FILTER=CLEAR; */

    /*/ -speck <MIN_DBZ> <max_a>  Threshold by min dBz, detect specks < A  */
    /*/ -speck  -20dBz     5pix      */
    if (param(argc,argv,i,"-speck")){
      LAST_FILTER=SPECK;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      if (intensity<=0)
	intensity=1;
      size=FMI_IARG(++i);
      detect_specks(&volume[s],&prob[s],intensity,histogram_area);
      semisigmoid_image(&prob[s],size);
      invert_image(&prob[s]);
      translate_intensity(&prob[s],255,0);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -specknorm_old <MIN_DBZ> <max_a> <max_n>  Threshold by min dBz, then detect specks, size A_max_range <=> size N*A A  */
    /*/               -20        5       16   */
    if (param(argc,argv,i,"-specknorm_old")){
      LAST_FILTER=SPECK;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      weight=FMI_IARG(++i);
      detect_specks(&volume[s],&prob[s],intensity,histogram_area);
      distance_compensation_mul(&prob[s],weight); 
      semisigmoid_image(&prob[s],size);
      invert_image(&prob[s]);
      translate_intensity(&prob[s],255,0);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -clutter <MIN_DBZ> <max_incomp>	 Remove specks under incompactness A */
    /*/              -5         5  */
    if (param(argc,argv,i,"-clutter")){
      LAST_FILTER=CLUTTER;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      detect_specks(&volume[s],&prob[s],intensity,histogram_compactness);
      semisigmoid_image(&prob[s],size);
      invert_image(&prob[s]);
      translate_intensity(&prob[s],255,0);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -clutter2 <MIN_DBZ> <max_smooth> Remove specks under smoothness */
    /*/               -5           60 */
    if (param(argc,argv,i,"-clutter2")){
      LAST_FILTER=CLUTTER2;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      detect_specks(&volume[s],&prob[s],intensity,histogram_smoothness);
      invert_image(&prob[s]);
      translate_intensity(&prob[s],255,0);
      semisigmoid_image(&prob[s],255-size);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -ground <-DBZ/1000m> <half_altitude> Remove ground echo by steepest-descent */
    /*/            -5dbz            2500m  */
    if (param(argc,argv,i,"-ground")){
      LAST_FILTER=GROUND;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=rel_dbz_to_int(FMI_IARG(++i));
      height=FMI_IARG(++i);
      detect_ground_echo_mingrad(&volume[s],sweep_count,&prob[s],intensity,height);
      i++;
      UPDATE_CLASSIFICATION;
    }
    /*/ -ground2 <-DBZ/1000m> <half_altitude> Remove ground echo by steepest-descent, normalized */
    /*/            -5dbz            2500m  */
    if (param(argc,argv,i,"-ground2")){
      LAST_FILTER=GROUND;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=rel_dbz_to_int(FMI_IARG(++i));
      height=FMI_IARG(++i);
      detect_ground_echo_minnetgrad(&volume[s],sweep_count,&prob[s],intensity,height);
      i++;
      UPDATE_CLASSIFICATION;
    }
    if (param(argc,argv,i,"-sweep_info")) dump_sweep_info();

    /*/ -meteosat img.pgm <T_P50> <T_P75> <MIN DBZ> <min area> Remove hot large specks */
    /*/                     -9'C    -7'C     30dbz     10pix  */
    if (param(argc,argv,i,"-meteosat")){
      LAST_FILTER=METEOSAT;
      LAST_FILTER_NAME=&argv[i][1];
      read_image(FMI_ARG(++i),&meteosat);
      t1=FMI_IARG(++i);
      t2=FMI_IARG(++i);
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      detect_too_warm(&volume[s],&prob[s],&meteosat,t1,t2,intensity,size);
      i++;
      UPDATE_CLASSIFICATION;
    }

   
    /*/ -softcut  <max_dbz>     <r>     <r2>	 Remove insect band */
    /*/           -10dbz    250km   100km */
    if (param(argc,argv,i,"-softcut")){
      LAST_FILTER=CUTOFF;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      radius   =FMI_IARG(++i);
      weight   =FMI_IARG(++i);
      detect_insect_band(&volume[s],&prob[s],intensity,radius,weight);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -biomet  <dbz_max> <dbz_delta> <alt_max> <alt_delta>   Remove insect band */
    /*/             -10dbz      5dBZ       5000m     1km */
    if (param(argc,argv,i,"-biomet")){
      LAST_FILTER=CUTOFF;
      LAST_FILTER_NAME=&argv[i][1];
      intensity      =abs_dbz_to_byte(FMI_IARG(++i));
      intensity_delta=rel_dbz_to_byte(FMI_IARG(++i));
      altitude       =FMI_IARG(++i);
      altitude_delta =FMI_IARG(++i);
      detect_biomet(&volume[s],&prob[s],intensity,intensity_delta,altitude,altitude_delta);
      i++;
      UPDATE_CLASSIFICATION;
    }
    
    
    /*/ -ship <min rel DBZ> <min A> Remove ships */
    /*/           50           20  */
    if (param(argc,argv,i,"-ship")){
      LAST_FILTER=SHIP;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=rel_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      detect_ships(&volume[s],&prob[s],intensity,size);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -emitter <MIN_DBZ> <LENGTH>   Filter unity-width emitter lines */
    /*/            10dbz      4  */
    if (param(argc,argv,i,"-emitter")){
      LAST_FILTER=EMITTER;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      detect_emitters(&volume[s],&prob[s],intensity,size);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -emitter0 <MIN_DBZ> <min_length> <max_width>   Filter emitter lines */
    /*/ -emitter0  -20dBZ       8bins        2°*/
    /* */
    /*  if (param(argc,argv,i,"-emitter0")){ */
    if (param(argc,argv,i,"-xemitter0")){
      LAST_FILTER=EMITTER;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      length=FMI_IARG(++i);
      width=FMI_IARG(++i);      
      detect_emitters2(&volume[s],&prob[s],intensity,length,width);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -emitter2 <MIN_DBZ> <LENGTH> <width> 	 Filter emitter lines */
    /*/              -10dbz 4bins 2° */
    /*/              -10dbz 3bins 3° */
    if (param(argc,argv,i,"-emitter2")){
      LAST_FILTER=EMITTER;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      length=FMI_IARG(++i);
      width=FMI_IARG(++i);
      detect_emitters2(&volume[s],&prob[s],intensity,length,width);
      i++;
      UPDATE_CLASSIFICATION;
    }

  /*/ -emitter2 <MIN_DBZ> <LENGTH> <width> 	 Filter emitter lines */
    /*/               10        4        2  */
    if (param(argc,argv,i,"-emitter2old")){
      LAST_FILTER=EMITTER;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      length=FMI_IARG(++i);
      width=FMI_IARG(++i);
      detect_emitters2(&volume[s],&prob[s],intensity,length,width);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -sun      <MIN_DBZ>  <min_length> <max_thickness>      Remove sun  */
    /*/            -20dBZ        100           3 */
    if (param(argc,argv,i,"-sun")){
      LAST_FILTER=SUN;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      height=FMI_IARG(++i);
      detect_sun(&volume[s],&prob[s],intensity,height,size);
      i++;
      UPDATE_CLASSIFICATION;
    }

    /*/ -sun2   <MIN_DBZ>  <min_length> <max_thickness> <azimuth> <elevation>   Remove sun  */
    /*/           -20dBZ        100          3             45°        2°*/
    if (param(argc,argv,i,"-sun2")){
      LAST_FILTER=SUN;
      LAST_FILTER_NAME=&argv[i][1];
      intensity=abs_dbz_to_byte(FMI_IARG(++i));
      size=FMI_IARG(++i);
      height=FMI_IARG(++i);
      azimuth=FMI_IARG(++i);
      elevation=FMI_IARG(++i);
      detect_sun2(&volume[s],&prob[s],intensity,height,size,azimuth,elevation);
      i++;
      UPDATE_CLASSIFICATION;
    }
  
    /*/ -doppler <filename> <WIDTH> <HEIGHT> <threshold>   Remove doppler anomalies */
    /*/ -doppler file.pgm 1bins 1° 128   Remove doppler anomalies*/
    if (param(argc,argv,i,"-doppler")){
      LAST_FILTER=DOPPLER;
      LAST_FILTER_NAME=&argv[i][1];
      if (files==0)
	read_radar_data(1,&argv[++i],volume_V);
      else
	read_radar_data(files,&argv[++i],volume_V);
      width=FMI_IARG(++i);
      height=FMI_IARG(++i);
      intensity=FMI_IARG(++i);
      detect_doppler_anomaly(&volume_V[s],&prob[s],width,height,intensity);
      i++;
      UPDATE_CLASSIFICATION;
    }


    /*/ sampling */
    /*/  -SVc x y     Volume dbz's, cartesian coords */
    /*/  -SVp r phi   Volume dbz's, polar coords */
    if (paramn(argc,argv,i,"-S")){
      generate_samples(argv,&i,volume,sweep_count);
      i++;
    }


    /*   if (LAST_FILTER!=CLEAR) */


    /*/ONLINE RESTORATION (sweep, affects the source image, unrecoverable)   */
    /*/ -t <PROB>        (use -OO** and -Oo**  to extract images) */
    if (param(argc,argv,i,"-t")){
      prob_threshold=FMI_IARG(++i);
      restore_image(&volume[s],&volume[s],&prob[s],prob_threshold);
      i++;
    }

    /*/ONLINE DETECTION (sweep) (NEGATIVE FILTERING)  */
    /*/ -d <PROB>        (use -Of**  to extract images) */
    if (param(argc,argv,i,"-d")){
      prob_threshold=FMI_IARG(++i);
      restore_image_neg(&volume[s],&volume_filtered[s],&prob[s],prob_threshold);
      i++;
    }

	/*    if (strcmp(argv[i],"-T")==0) */
    if (param(argc,argv,i,"-T"))
      fmi_error("-T obsolete, use -Tv or -Ts instead");

    /*/OFFLINE RESTORATION (results to separate image, repeatable)   */
    /*/ -Ts <PROB>     (current sweep) */
    if (param(argc,argv,i,"-Ts")){
      prob_threshold=FMI_IARG(++i);
      restore_image(&volume[s],&volume_filtered[s],&master_prob[s],prob_threshold);
      i++;
    }

    /*/ -Tv <PROB>     (volume) */
    if (param(argc,argv,i,"-Tv")){
      prob_threshold=FMI_IARG(++i);
      restore_image(volume,volume_filtered,master_prob,prob_threshold);
      i++;
    }

    if (param(argc,argv,i,"-basename")){
      filename_base = FMI_ARG(++i);
      i++;
    }

    /*/OUTPUT (applicable multiple times, in any stage of processing) */
    /*/ -O<image><proj><format> filename[.p?m] */
    if (i<argc)
      if (strncmp(argv[i],"-O",2)==0){
	fmi_debug(3,"output active");
	FILTER_CODE=LAST_FILTER_NAME;
	/*FILTER_CODE="JIMBO"; */
	/*/  <image> */
	switch (argv[i][2]){
      case 'O':
	/*/     O    volume, original/online-filtered */
	PRODUCT_CODE="ORIG";
	FILTER_CODE="all";
	image_ptr=volume;
	break;
      case 'o':
	/*/     o    current sweep, original/online-filtered */
	PRODUCT_CODE="orig";
	FILTER_CODE="all";
	image_ptr=&volume[s];
	break;
      case 'D':
	/*/     D    volume, detected (=negative filtered) */
	PRODUCT_CODE="DETC";
	image_ptr=volume_filtered;
	break;
      case 'd':
	/*/     d    current sweep, detected (=negative filtered) */
	PRODUCT_CODE="detc";
	image_ptr=&volume_filtered[s];
	break;
      case 'F':
	/*/     F    volume, offline-filtered */
	PRODUCT_CODE="FILT";
	FILTER_CODE="all";
	image_ptr=volume_filtered;
	break;
      case 'f':
	/*/     f    current sweep, offline-filtered */
	PRODUCT_CODE="filt";
	FILTER_CODE="all";
	image_ptr=&volume_filtered[s];
	break;
      case 'P':
	/*/     P    probability of anomalies (volume, cumulated by detectors this far) */
	PRODUCT_CODE="PROB";
	FILTER_CODE="all";
	image_ptr=master_prob;
	break;
      case 'p':
	/*/     p    probability of anomalies (sweep,cumulated) */
	PRODUCT_CODE="prob";
	FILTER_CODE="all";
	image_ptr=&master_prob[s];
	break;
      case 'A':
	/*/     A    probability of anomaly (volume, by latest detector) */
	PRODUCT_CODE="PROB";
	image_ptr=prob;
	break;
      case 'a':
	/*/     a    probability of anomaly (sweep, by latest detector) */
	PRODUCT_CODE="prob";
	image_ptr=&prob[s];
	break;
      case 'M':
	/*/     M    image with anomalies marked with EMIT,SHIP etc, vol */
	PRODUCT_CODE="MARK";
	FILTER_CODE="all";
	image_ptr=mark;
	break;
      case 'm':
	/*/     m    image with anomalies marked with EMIT,SHIP etc, vol */
	PRODUCT_CODE="mark";
	FILTER_CODE="all";
	image_ptr=&mark[s];
	break;
      default:
	fmi_error(" -O : unsupported TYPE "); }
      
      /*/  <proj>  image projection */
      switch (argv[i][3]){
      case 'c': 
	/*/     c    cartesian */
	to_cart(image_ptr,&cart,NO_DATA);
	image_ptr=&cart;
	break;
      case 'p': 
	/*/     p    polar */
      case '4': 
	/*/     4    2x2 combo, ex:  -O*4* combo.ppm = (source,*)x(polar,cartesian) */
      case '6':  
	/*/     6    3x2 combo */
	break;
      default:
	fmi_error(" -O : unsupported COORDSYSTEM "); }

      /*/ <format> image coloring and format */
      switch (argv[i][4]){
      case 't': 
	/*/     t    gray, pgm (no LSB codes (threshold=16) black background) */
      case 'G': 
	/*/     G    gray, pgm (black background) */
      case 'g': 
	/*/     g    gray, pgm (white background) */
      case 'W': 
	/*/     V    gray, pgm (printable media, white background) */
      case 'V': 
	/*/     V    gray, pgm (printable media, inverted, white background) */
	break; 
      case 'I':       /* IRIS */
	/*/     I    colored, IRIS-like, black background */
	translate_intensity(image_ptr,NO_DATA,0);
      case 'i':  
	/*/     i    colored, IRIS-like */
	pgm_to_ppm_radar_iris(image_ptr,ppm);
	image_ptr=ppm;
	break;
      case 'r':  
	/*/     r    colored (x<128=green, x>128=red), ppm */
	pgm_to_redgreen(image_ptr,ppm);
	image_ptr=ppm;
	break;
      case 'M': 
	/*/     M    gray, marked (colored background), DATA_MIN cutoff, ppm */
	threshold_image(image_ptr,image_ptr,DATA_MIN);
	translate_intensity(image_ptr,NO_DATA,LAST_FILTER);
      case 'm': 
	/*/     m    gray, marked, ppm */
	pgm_to_ppm_radar(image_ptr,ppm);
	image_ptr=ppm;
	break;
      case 'h':
        /* HDF */
        write_h5_radar_data(image_ptr, filename, template);
        break;
      default:
	fmi_error(" -O : unsupported image FORMAT ");}

      filename = argv[i+1];	  
      if (strlen(filename)==1)
	switch (filename[0]){
	case 'b':
	  filename = tempfilename;
	  strcpy(filename,filename_base);
	  strcat(filename,"_");
	  strcat(filename,PRODUCT_CODE);
	  strcat(filename,"_");
	  strcat(filename,FILTER_CODE);
	  break;
	default:
	  filename = "temp";
	}

      /* IMAGE FORMAT, SINGLE */
      switch (argv[i][4]){
      case 'i':      /* IRIS */
      case 'I':      /* IRIS */
      case 'r': 
      case 'M': 
      case 'm': 
	    /* GRAY PPM */
	write_image(filename,ppm,PPM_RAW);
	break;
      case 'W': 
	/* GRAY PGM */
	translate_intensity(image_ptr,NO_DATA,0);
	pgm_to_pgm_print2(image_ptr,image_ptr); /* spoil data? */
	write_image(filename,image_ptr,PGM_RAW);
	break; 
      case 'V': 
	/* GRAY PGM INVERTED */
	translate_intensity(image_ptr,NO_DATA,0);
	pgm_to_pgm_print(image_ptr,image_ptr); /* spoil data? */
	write_image(filename,image_ptr,PGM_RAW);
	break; 
      case 't':
	threshold_image(image_ptr,image_ptr,DATA_MIN);
      case 'G':
	translate_intensity(image_ptr,NO_DATA,0);
      case 'g': 
	/* GRAY PGM */
	write_image(filename,image_ptr,PGM_RAW);
	    break; 
      case 'h':
        /* HDF */
        assert(template != NULL);
        write_h5_radar_data(image_ptr, filename, template);
        break;
      default:
	fmi_error(" -O : unsupported image FORMAT ");}
      i++; /* filename */
      i++;
    }

    if (i==argv_start){
      fmi_debug(0,"unknown option: ");
      fmi_debug(0,argv[i]);
      fmi_error(" quitting");}

  }


return 1;
}
