#!/bin/bash

PATH=$(pwd)

cd $PATH

echo $PATH

cd ../../data

echo $(pwd)

 /usr/bin/gzip -dk covtype.data.gz


