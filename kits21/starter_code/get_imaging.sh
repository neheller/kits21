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
    wget -O "$DSTF" https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_$(printf "%05d" $I).nii.gz
  fi
done