#!/bin/bash

# Given TIMESTAMP, retrieves data

#TIMESTAMP=${TIMESTAMP:-'201006281211'}
TIMESTAMP=${TIMESTAMP:-'201006120000'}
#TIMESTAMP=${TIMESTAMP:-`date --utc +'%Y%m%d%H%M'`}
YEAR=${YEAR:-${TIMESTAMP:0:4}}
MONTH=${MONTH:-${TIMESTAMP:4:2}}
DAY=${DAY:-${TIMESTAMP:6:2}}
HOUR=${HOUR:-${TIMESTAMP:8:2}}
MINUTE=${MINUTE:-${TIMESTAMP:10:2}}
MINUTE=$(( 10#$MINUTE ))
MINUTE5=`printf '%02d' $(( 10#$MINUTE-10#$MINUTE%5 ))`
TIMESTAMP="$YEAR$MONTH$DAY$HOUR$MINUTE5"
TIMESTAMP_DIR=${TIMESTAMP_DIR:-"$YEAR/$MONTH/$DAY"}

SITES='IKA VAN KOR ANJ'

# Retrieve raw data

function getData() {
    mkdir -p tmp
    for MINUTE in {0,1,2,3,4,5}{0,5}; do
    #for MINUTE in {0,1}{0,5}; do
	for SITE in $SITES; do
	    wget --proxy=off --no-clobber  http://radar.fmi.fi/products/query/$YEAR/$MONTH/$DAY/fi/fmi/radar/raw/$YEAR$MONTH$DAY$HOUR${MINUTE}_fi.fmi.radar.raw_SITE=${SITE}_VOL=00_DATA=Z.png -O tmp/$YEAR$MONTH$DAY$HOUR${MINUTE}-$SITE.png
#	    sleep 2
	done
    done
}

getData; 






# Geographical bounding box (~Finland)
#BBOX=18,58,28,63  # SOUTH
#BBOX=17,58,30,67
#BBOX=19,58,27,62  # tight
BBOX=19,58,29,64  # tight
#BBOX=17,57,31,68  # loose

# Size of the image
SIZE=480,600

# Compositing principle
  METHOD=WAVG,1,5
# METHOD=MAX

# Postprocessing of bins
# d,0 = Auto
# d,5,3 = Short, slightly horizontal
#INTERPOLATION=d,2.0,3.0
INTERPOLATION='d,0'

# Polar product to be composited. Leave empty if first sweep only (PPI).
POLARPRODUCT='--cappi 500,500,0.3'

#FADE=0.8,2,-2
FADE=0.8,0,0

command_prefix="rack --cBBox $BBOX  --cSize $SIZE  --cMethod $METHOD  --cInterpolation $INTERPOLATION"

# -cFade 0.7 --cRead

# radar-KOR-00-Z.png $POLARPRODUCT --SITE KOR --cCreateTile dw --cAddTile 1.0 -o composite-tile-KOR.png \


MAP=map-$BBOX-$SIZE.png

wget --proxy=off  --no-clobber -O $MAP     http://radar.fmi.fi/products/query/mapserver_BBOX=${BBOX}_SIZE=${SIZE}_BGCOLOR=0x506090_MODULATE=70,70.png 





#rm -v tmp/*tile*


mkdir -p ./log
rm -f log/* sites.log

i=0
TIMESTAMP_PREV=''

#while (( i < 20 )); do
while (( i < 10 )); do
    # echo $i
    MINUTE=`printf '%02d' $(( 10#$i ))`
    MINUTE05=`printf '%02d' $(( 10#$i - 10#$i%5 ))`
    #echo $MINUTE05 $MINUTE $RANDOM
    # collect "received" data 
    #SITES_NOW=''
    SITES="`fgrep -v '#' living-composite-sites.txt | fgrep $MINUTE |cut -s -d' ' -f2-`"
    echo $SITES     
    echo $MINUTE $SITES >> sites.log
    #echo ${SITES[*]}
    # sleep 1
    WEIGHT='1.0'
    TIMESTAMP=$YEAR$MONTH$DAY$HOUR$MINUTE
    TIMESTAMP05=$YEAR$MONTH$DAY$HOUR$MINUTE05
    CMD_BODY=''
    LOGFILE=log/$MINUTE05.log
    echo "#$MINUTE" >> $LOGFILE 
    for SITE in $SITES; do
	P=`echo $POLARPRODUCT | tr -d -- '- '`
	TILE=tmp/$TIMESTAMP05-tile-$SITE-$P.png
	echo "# Generating $SITE (if missing)" >> $LOGFILE 
	       # Create tile, if missing
	       if [ ! -f $TILE ]; then
		    input=tmp/$TIMESTAMP05-$SITE.png
		    command="$command_prefix $input --threshold 64 $POLARPRODUCT --SITE $SITE --cCreateTile dw -o $TILE"
		    echo $command >> $LOGFILE
		    eval $command
	       fi
	 CMD_BODY="$CMD_BODY --cLoadTile $TILE --cAddTile $WEIGHT"
    done;
    echo $i 


    # Use previous as backround, if exists.
    TILED="tmp/$TIMESTAMP_PREV-tiled.png"
    if [ -f $TILED ]; then
	TILED="--cFade $FADE  --cLoad tmp/$TIMESTAMP_PREV-tiled.png"
    else
	TILED=''
    fi

    NICK=eradFade-$HOUR$MINUTE

    command="$command_prefix  $TILED  $CMD_BODY --cExtract=dw  -o tmp/$TIMESTAMP-tiled.png --view 0 -o $NICK-d.png --view 1 --average 5  -o $NICK-w.png"
    command="$command --copy 0,a --view a --rescale 2 --view f --palette dbz-16.png -o $NICK-c.png"


    echo "# Generatimg composite $HOUR:$MINUTE" >> $LOGFILE
    echo $command >> $LOGFILE
    echo  >> $LOGFILE
    eval $command

    command="composite -compose over $NICK-c.png $MAP $NICK-m.png;"
    echo $command >> $LOGFILE
    eval $command
    command="convert -append $NICK-m.png $NICK-w.png -resize '50%' $NICK-M.png" 
    echo $command >> $LOGFILE
    eval $command
    echo  >> $LOGFILE

# --view 0 -o composite-data.png  --view 1 -o composite-weight.png  
# --view 0 --palette iris-dbz-16.png -o composite-color.png
# $SITES_NOW
    TIMESTAMP_PREV=$TIMESTAMP
    i=$(( i + 1 ))
#    sleep $SLEEP
done

convert +append erad-000?-M.png erad-$FADE-M.png

cat `ls -1tr log/*` > log.txt
cat log.txt

echo 'cat `ls -1tr log/*`'
echo "xv erad*-m.png erad*-M.png  erad*-d.png erad*-w.png"
echo "rm -v tmp/*tile* erad*-?.png [0-9]?-???.png"
