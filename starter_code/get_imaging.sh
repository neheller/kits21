#!/bin/bash
if [ ! -d "data" ]
then
  mkdir data
fi
for I in {0..299}
# TODO ignore existing like the python implementation
do
  DSTP=$(printf "data/case_%05d" $I)
  DSTF=$(printf "data/case_%05d/imaging.nii.gz" $I)
  if [ ! -d $DSTP ]
  then
    mkdir $DSTP
  fi
  if [ ! -f $DSTF ]
  then
    wget https://kits19.sfo2.digitaloceanspaces.com/master_$(printf "%05d" $I).nii.gz
    mv master_$(printf "%05d" $I).nii.gz $DSTF
  fi
done