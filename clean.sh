#!/usr/bin/bash

ans=""

rm out*

echo "./fwd_res :"
ls ./fwd_res

echo -n "Clear ./fwd_res ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm ./fwd_res/*
fi

echo "./pred_subfiles :"
ls ./pred_subfiles

echo -n "Clear ./pred_subfiles ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm ./pred_subfiles/*
fi

exit 0
