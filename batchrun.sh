#!/bin/bash


cd /jbodstorage/data_ankit/Trajec_ana/


pwd


theta=$1
outfile+=theta_
outfile+=$theta

vel=$2
outfile+=vel_
outfile+=$vel

lambda=$3
outfile+=lambda_
outfile+=$lambda

fg=$4
outfile+=_fg_
outfile+=$fg

Environment=$5
outfile+=_$Environment

delta=$6
outfile+=_delta_
outfile+=$delta

redshift=$7
outfile+=_redshift_
outfile+=$redshift
outfile+=.dat


outdir=/jbodstorage/data_ankit/Trajec_ana/theta$theta/vel$vel/lambda$lambda/fg$fg/$Environment/delta$delta/z$redshift/
mkdir -p $outdir


file=params.ini

cp $file $outdir

paramfile=$outdir$file




if [[ "$delta" = "0.5" ]];
then
    sed -i "22s/del1: 1.0/del1: 0.5/" $paramfile 

fi

if [[ "$delta" = "0.1" ]];
then
    sed -i "22s/del1: 1.0/del1: 0.1/" $paramfile

fi

if [[ "$delta" = "0.3" ]];
then
    sed -i "22s/del1: 1.0/del1: 0.3/" $paramfile

fi




if [[ "$fg" = "0.3" ]];
then
    sed -i "29s/alpha:/alpha: 0.3/" $paramfile 

fi

if [[ "$fg" = "0.1" ]];
then
    sed -i "29s/alpha:/alpha: 0.1/" $paramfile 

fi

if [[ "$Environment" = "cluster" ]];
then
    sed -i "20s/m_200:/m_200: 2.0e15/" $paramfile
fi

if [[ "$Environment" = "group" ]];
then
    sed -i "20s/m_200:/m_200: 6.3e13/" $paramfile
fi

if [[ "$Environment" = "galaxy" ]];
then
    sed -i "20s/m_200:/m_200: 1.0e12/" $paramfile
fi


if [[ "$lambda" = "0.04" ]];
then
    sed -i "28s/lmbda:/lmbda:0.04/" $paramfile
fi

if [[ "$lambda" = "0.02" ]];
then
    sed -i "28s/lmbda:/lmbda:0.02/" $paramfile
fi

if [[ "$lambda" = "0.08" ]];
then
    sed -i "28s/lmbda:/lmbda:0.08/" $paramfile
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



for md in 1.0e6 1.0e7 1.0e8 1.0e9 1.0e10 1.0e11 4.0e11 8.0e11 1.0e12 
    do
	        sed  -i "31s/^md_tot: .*$/md_tot: $md/" $paramfile
		
		sed -i "5s/^theta:.*$/theta: $theta/" $paramfile

		sed -i "6s/^vel_factor:.*$/vel_factor: $vel/" $paramfile

                tmp="${paramfile/params.ini/md_${md}_params.ini}"
                tmp1=${outdir}md_${md}_.dat
                sed -i "7s|^$var1.*$|outfile: $tmp1|" "$paramfile"
                cp $paramfile $tmp
	    	echo $tmp
	        ipython testing_dyn.py $tmp &

	

    done






