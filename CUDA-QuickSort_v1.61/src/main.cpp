/*
 * main.cpp
 *
  * Copyright © 2012-2015 Emanuele Manca
 *
 **********************************************************************************************
 **********************************************************************************************
 *
 	This file is part of CUDA-Quicksort.

    CUDA-Quicksort is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDA-Quicksort is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDA-Quicksort.

    If not, see http://www.gnu.org/licenses/gpl-3.0.txt and http://www.gnu.org/copyleft/gpl.html


  **********************************************************************************************
  **********************************************************************************************
 *
 * Contact: Ing. Emanuele Manca
 *
 * Department of Electrical and Electronic Engineering,
 * University of Cagliari,
 * P.zza D’Armi, 09123, Cagliari, Italy
 *
 * email: emanuele.manca@diee.unica.it
 *
 *
 * This software contains source code provided by NVIDIA Corporation
 * license: http://developer.download.nvidia.com/licenses/general_license.txt
 *
 * the following Functions are based or derived from the NVIDIA CUDA SDK:
 *
 * 		1. bitonicSort()
 * 		2. mergesort()
 * 		3. thrust::sort()
 *
 *
 *
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort
 *
 */
# include <cstdlib>

#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>

#include <gpuqsort.h>
#include "randomDistr.h"
#include "CUDA-Quicksort.h"


using namespace std;


bool validateSortedValue(Type* dataTest, Type* data, unsigned int dataSize)
{

	unsigned int i = 0;

	while(i<dataSize)
	{
		if( data[i] != dataTest[i] )
		{//cout<<i<<' '<<data[i]<<endl;
			cout<<"Error: dataTest != data"<<endl;
			//cerr<<"dataTest["<<i<<"] ==\t"<< dataTest[i]<<endl;
			//cerr<<"data    ["<<i<<"} ==\t"<< data[i]    <<endl;
			//getchar();
			return false;
		}
		i++;
	}

	return true;
}

void GPUtest(unsigned int size,string& select,int device)
{
	const unsigned int N = size ==0 ? 2<<24 : size;

	Type* inData    =  new Type[N];
	Type* outData   =  new Type[N];
	Type* datatest  =  new Type[N];

	double timerQuick;
	string distr;

	if(select == "all")
		distr = "uniform";
	else
		distr = select;


	distribution(inData,N,distr);

	memcpy(datatest,inData,N*sizeof(unsigned int));

	std::ofstream myfile;
	myfile.open(distr +"resultsOldCode.csv", std::ios::app);
	//myfile << distr << "\n";
	unsigned int dataSize = size ==0 ? 2<<19 : size;
	while( dataSize<=N )
	{
		cout<<"\ndataSize: "<<dataSize<<"\tdistribution: "<<distr<<endl;
		//CUDA-QuickSort works only for thread=128|256 if the shared memory size is 1024 (see SHARED_LIMIT on CUDA-Quicksort.h).
		//The limit is thread*4<=SHARED_LIMIT
		for(int threads=256; threads<=256 ; threads*=2)
				{
					CUDA_Quicksort(inData,outData,dataSize,threads,device,&timerQuick);

					cout<<"time: "<<timerQuick<<" ms";
					cout<<"\tthreads: "<<threads<<endl;
					myfile << timerQuick << ",";

					sort(datatest,datatest+dataSize);
					validateSortedValue(datatest,outData,dataSize);
				}
		dataSize *= 2;

	}
	myfile  << "\n";

	delete inData;
	delete outData;
	delete datatest;
}

int main(int argc,const char* argv[])
{
	string ds[3] = {"uniform", "gaussian", "sorted"};
	unsigned int sizes[2] = { (2<<11) , (2 << 27) };
	
	GPUtest(sizes[0], ds[2], 0);
	

	return 0;
}



