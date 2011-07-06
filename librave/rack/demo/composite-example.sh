#!/bin/bash

DEBUG=${DEBUG:-'3'}

# This is a selector for selecting between basically two modes.
# The modes are not direct concepts of rack, but control this script.
#
# Direct mode (default). Read all the raw files and create a composite
# MODE=
#
# Create a tile for a single radar (and save it)
# MODE=tile
#
# Composite from tiles
# MODE=tiled

# Geographical bounding box 
# BBOX=${BBOX:-18,58,28,63}    # Finland
# BBOX=${BBOX:-'8,53,32,70.5'} # Finland and Sweden
BBOX=${BBOX:-'5,45,32,70'}     # Baltradish

# FMI only: retrieve background map
#MAPFILE=mapserver_BBOX=$BBOX}_SIZE=${SIZE}_BGCOLOR=0x506090_MODULATE=70,70_MEDIAN=3.png


# Size of the image
# For Doxygen documentation, this size is rather small.
# Practically, images of over 1000x1000 can be expected.
#
# SIZE=${SIZE:-640,800}
# SIZE=${SIZE:-960,1200}
SIZE=${SIZE:-800,1000}
#SIZE=${SIZE:-200,400}

# Compositing principle
METHOD=${METHOD:-WAVG,3,2}
# METHOD=MAX
# METHOD=MAXQ

# Postprocessing of bins
# INTERPOLATION=d,0   Auto
# INTERPOLATION=d,5,3 Short, slightly horizontal
# INTERPOLATION=d,10,20
INTERPOLATION=${INTERPOLATION:-'d,0'}

# Polar product to be composited. Leave empty if first sweep only (PPI).
# cappi: altitude, beam width
POLARPRODUCT=${POLARPRODUCT:-'cappi,500,0.8'}
POLARPRODUCT=( $POLARPRODUCT ) #trim

polarproduct=${POLARPRODUCT:+'--'${POLARPRODUCT/,/ 500,}}


# Uncomment this if you want anomaly detection and removal
ANDRE=${ANDRE:-'--aSpeckle 32,30 --aEmitter 3,6 --rShip 32,6 --rBiomet 16,4,500,50 --aPaste --aGapFill 3 '}


# Overall weight. In this script, this is for demonstration only.
# The value can be varied for each radar, depending on its overall signal
# quality etc. If uncommented, a random value will be used for illustration.
#
# WEIGHT=1.0

if [[ $# == 0 ]]; then
    echo "Usage:"
    echo "  $0 *.h5"
    echo "  BBOX=$BBOX SIZE=$SIZE METHOD=$METHOD $0 *.h5"
    exit
fi


# PART 1: Initial part 
command="rack --debug $DEBUG  --cBBox $BBOX  --cSize $SIZE  --cMethod $METHOD  --cInterpolation $INTERPOLATION
"

# PART 2: Input the given radars
for i in $*; do
    W=${WEIGHT:-"0.$RANDOM"}
    command="$command $i $ANDRE $polarproduct --cCreateTile dw  --cAddTile $W
"
#  -o ${i%.*}-tile.h5
#  -o ${i%.*}-tile.png
done

# PART 3: Extract image products from the composite
command="$command --cExtract=dwSC -o composite.h5 --view 0 -o composite-data.png  --view 1 -o composite-weight.png --view 2 -o composite-discrepancy.png  --view 3 --rescale 10 -o composite-count.png --view 0 --threshold 60 --palette dbz-16.png -o composite-color.png"


echo $command
echo
eval $command
echo
echo $command



# FMI ONLY

if [[ "$MAPFILE" != '' ]]; then
    wget --no-clobber --proxy=off http://radar.fmi.fi/products/query/mapserver/$MAPFILE
fi

