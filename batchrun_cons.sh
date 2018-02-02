#!/bin/bash


cd /home/storage1/ph14036/TestingH/


pwd

lambda=$1
outfile+=lambda_
outfile+=$lambda

fg=$2
outfile+=_fg_
outfile+=$fg

Environment=$3
outfile+=_$Environment

delta=$4
outfile+=_delta_
outfile+=$delta

redshift=$5
outfile+=_redshift_
outfile+=$redshift
outfile+=.dat


outdir=/home/storage1/ph14036/TestingH/lambda$lambda/fg$fg/$Environment/delta$delta/z$redshift/
mkdir -p $outdir


file=params.ini

cp $file $outdir

paramfile=$outdir$file


if [[ "$delta" = "0.5" ]];
then
    sed -i "20s/del1: 1.0/del1: 0.5/" $paramfile 

fi

if [[ "$delta" = "0.1" ]];
then
    sed -i "20s/del1: 1.0/del1: 0.1/" $paramfile

fi

if [[ "$delta" = "0.3" ]];
then
    sed -i "20s/del1: 1.0/del1: 0.3/" $paramfile

fi




if [[ "$fg" = "0.3" ]];
then
    sed -i "27s/alpha:/alpha: 0.3/" $paramfile 

fi

if [[ "$fg" = "0.1" ]];
then
    sed -i "27s/alpha:/alpha: 0.1/" $paramfile 

fi

if [[ "$Environment" = "cluster" ]];
then
    sed -i "18s/m_200:/m_200: 2.0e15/" $paramfile
fi

if [[ "$Environment" = "group" ]];
then
    sed -i "18s/m_200:/m_200: 6.3e13/" $paramfile
fi

if [[ "$Environment" = "galaxy" ]];
then
    sed -i "18s/m_200:/m_200: 1.0e12/" $paramfile
fi


if [[ "$lambda" = "0.04" ]];
then
    sed -i "26s/lmbda:/lmbda:0.04/" $paramfile
fi

if [[ "$lambda" = "0.02" ]];
then
    sed -i "26s/lmbda:/lmbda:0.02/" $paramfile
fi

if [[ "$lambda" = "0.08" ]];
then
    sed -i "26s/lmbda:/lmbda:0.08/" $paramfile
fi




if [[ "$redshift" = "0.0" ]];
then
    sed -i "4s/^z:.*$/z: 0.0/" $paramfile
fi



if [[ "$redshift" = "0.5" ]];
then
    sed -i "4s/^z:.*$/z: 0.5/" $paramfile
fi


if [[ "$redshift" = "1.0" ]];
then
    sed -i "4s/^z:.*$/z: 1.0/" $paramfile
fi

var1=outfile:



for md in 1.0e6 1.0e7 1.0e8 1.0e9 1.0e10 1.0e11 4.0e11 8.0e11 1.0e12 2.0e12 4.0e12 6.0e12 8.0e12 1.0e13
    do
	    for v in 1 1.5 2 2.5 3
	    do
	        sed  -i "29s/^md_tot: .*$/md_tot: $md/" $paramfile
            	sed  -i "30s/^vel_factor:.*$/vel_factor:$v/" $paramfile
                tmp="${paramfile/params.ini/md_${md}_v_${v}_params.ini}"
                tmp1=${outdir}md_${md}_v_${v}_.dat
                sed -i "5s|^$var1.*$|outfile: $tmp1|" "$paramfile"
                cp $paramfile $tmp
	    	echo $tmp
	        qsub run.sh $tmp

            done
	

    done






