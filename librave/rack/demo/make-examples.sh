#!/bin/bash

BATCH_FILE=examples.lst

# Picks examples which can be executed "literally" ie. do not contain
# variable markers like <value>

grep ' *FILES\?=' ../main/*.dox > $BATCH_FILE
grep '^[ ]*rack [^<]*' ../main/*.dox | fgrep -v '<'  >> $BATCH_FILE
grep '^[ ]*convert [^<]*' ../main/*.dox | fgrep -v '<'  >> $BATCH_FILE

#source
