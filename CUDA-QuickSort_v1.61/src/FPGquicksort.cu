/*
 * CUDA-Quicksort.cu
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
 * this software uses the library of NVIDIA CUDA SDK and the Cederman and Tsigas' GPU Quick Sort
 *
 */



#include <thrust/scan.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "scan.h"
#include <scan_common.h>
#include "CUDA-Quicksort.h"

 // extern __shared__ char sMemory[];


__device__ inline  double atomicMax(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int assumed;
	unsigned long long int old = *address_as_ull;

	assumed = old;
	old = atomicCAS(address_as_ull,
		assumed,
		__double_as_longlong(max(val, __longlong_as_double(assumed))));

	while (assumed != old)
	{
		assumed = old;
		old = atomicCAS(address_as_ull,
			assumed,
			__double_as_longlong(max(val, __longlong_as_double(assumed))));
	}
	return __longlong_as_double(old);
}


__device__ inline double atomicMin(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	assumed = old;
	old = atomicCAS(address_as_ull,
		assumed,
		__double_as_longlong(min(val, __longlong_as_double(assumed))));
	while (assumed != old)
	{
		assumed = old;
		old = atomicCAS(address_as_ull,
			assumed,
			__double_as_longlong(min(val, __longlong_as_double(assumed))));
	}
	return __longlong_as_double(old);
}





template <typename Type>
__device__ inline void Comparator(

	Type& valA,
	Type& valB,
	uint dir
) {
	Type t;
	if ((valA > valB) == dir) {
		t = valA; valA = valB; valB = t;
	}
}




static __device__ __forceinline__ unsigned int __qsflo(unsigned int word)
{
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
	return ret;
}

template <typename Type>
__global__ void globalBitonicSort(Type* indata, Type* outdata, Block<Type>* bucket, bool inputSelect, int shmem, int blocks)
{
	extern __shared__ uint shared[];


	Type* data;
	// 0 1 - 0 
	// 2 3 - 1 
	// 2 3 - 2 
	Block<Type> cord = bucket[blockIdx.x / blocks];
	uint tid = threadIdx.x + ((blockIdx.x % blocks) * blockDim.x);

	uint size = cord.end - cord.begin;
	bool select = !(cord.select);

	if (cord.end - cord.begin > shmem || cord.end - cord.begin == 0)
		return;

	unsigned int bitonicSize = 1 << (__qsflo(size - 1U) + 1);


	if (select)
		data = indata;
	else
		data = outdata;

	//__syncthreads();

	for (int i = tid;i < size;i += blockDim.x)
		shared[i] = data[i + cord.begin];


	for (int i = tid + size;i < bitonicSize;i += blockDim.x)
		shared[i] = 0xffffffff;

	__syncthreads();


	for (uint size = 2; size < bitonicSize; size <<= 1) {
		//Bitonic merge
		uint ddd = 1 ^ ((tid & (size / 2)) != 0);
		for (uint stride = size / 2; stride > 0; stride >>= 1) {
			__syncthreads();
			uint pos = 2 * tid - (tid & (stride - 1));
			//if(pos <bitonicSize){
			Comparator(
				shared[pos + 0],
				shared[pos + stride],
				ddd
			);
			// }
		}
	}


	//ddd == dir for the last bitonic merge step

	for (uint stride = bitonicSize / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		uint pos = 2 * tid - (tid & (stride - 1));
		// if(pos <bitonicSize){
		Comparator(
			shared[pos + 0],
			shared[pos + stride],
			1
		);
		// }
	}

	__syncthreads();

	// Write back the sorted data to its correct position
	for (int i = tid;i < size;i += blockDim.x)
		indata[i + cord.begin] = shared[i];

}




template <typename Type>
__global__ void quick(Type* indata, Type* buffer, Partition<Type>* partition, Block<Type>* bucket, int shmem, int size)
{
	extern __shared__ char s[];
	Type* sh_out = (Type*)&s[128];

	// __shared__ uint start1,end1;
	// __shared__ uint left,right;

	uint* start1, * end1;
	uint* left, * right;

	start1 = (uint*)&s; end1 = (uint*)&s[sizeof(uint)];
	left = (uint*)&s[2 * sizeof(uint)]; right = (uint*)&s[3 * sizeof(uint)];

	// printf("Here also");
	// *left = 32;
	// printf("Here also also");

	int tix = threadIdx.x;

	uint start = partition[blockIdx.x].from;
	uint end = partition[blockIdx.x].end;
	Type pivot = partition[blockIdx.x].pivot;
	uint nseq = partition[blockIdx.x].ibucket;

	uint lo = 0;
	uint hi = 0;

	Type lmin = 0xffffffff;
	Type rmax = 0;

	Type d;
	Type dd;


	// start read on 1° tile and store the coordinates of the items that must
	// be moved on the left or on the right of the pivot

	/*lo = (((d < pivot) * (lo + 1) + (d >= pivot) * lo) * (ii < end)) + (lo * (ii >= end));
	hi = (((d <= pivot) * (hi)+(d > pivot) * (hi + 1)) * (ii < end)) + (hi * (ii >= end));*/
	/*lo = (((dd < pivot) * (lo + 1) + (dd >= pivot) * lo) * (ii < end)) + (lo * (ii >= end));
	hi = (((dd <= pivot) * (hi)+(dd > pivot) * (hi + 1)) * (ii < end)) + (hi * (ii >= end));*/

	uint ii = tix + start;
	bool endUslov = (ii < end);
	d = indata[ii * endUslov];
	lo += (d < pivot) && endUslov;
	hi += (d > pivot) && endUslov;
	lmin = d * endUslov + lmin * !endUslov;
	rmax = d * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	// 4 + 4
	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) && endUslov;
	hi += (dd > pivot) && endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;


	// 8 + 8
	/*ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;

	ii += blockDim.x;
	endUslov = (ii < end);
	dd = indata[ii * endUslov];
	lo += (dd < pivot) * endUslov;
	hi += (dd > pivot) * endUslov;
	lmin = min(lmin, dd) * endUslov + lmin * !endUslov;
	rmax = max(rmax, dd) * endUslov + rmax * !endUslov;*/


	/*i++;
	d = indata[(tix + start + i * blockDim.x) * ((tix + start + i * blockDim.x) < end)];
	lo = ((d < pivot) * (lo + 1) + (d >= pivot) * lo) * ((tix + start + i * blockDim.x) < end) + lo * ((tix + start + i * blockDim.x) >= end);
	hi = ((d <= pivot) * (hi)+(d > pivot) * (hi + 1)) * ((tix + start + i * blockDim.x) < end) + lo * ((tix + start + i * blockDim.x) >= end);
	lmin = min(lmin, d);
	rmax = max(rmax, d);*/

	/*if (tix + start<end)
	{
		d = indata[tix+start];

		//count items smaller or bigger than the pivot
		// if d<pivot then ll++ else ll
		lo += (d < pivot);
		hi += (d > pivot);
		// lo=(d<pivot)*(lo+1)+(d>=pivot)*lo;
		// if d>pivot then lr++ else lr
		// hi=(d<=pivot)*(hi)+(d>pivot)*(hi+1);
		// lo = ((d < pivot) ? (lo + 1) : (lo));
		// hi = ((d > pivot) ? (hi + 1) : (hi));

		lmin = d;
		rmax= d;
	}*/

	/*i += blockDim.x;
	Type dd = indata[ii * (ii < end)];
	lo = (((dd < pivot) * (lo + 1) + (dd >= pivot) * lo) * (ii < end)) + (lo * (ii >= end));
	hi = (((dd <= pivot) * (hi)+(dd > pivot) * (hi + 1)) * (ii < end)) + (hi * (ii >= end));
	lmin = min(lmin, dd) * (ii < end) + lmin * (ii >= end);
	rmax = max(rmax, dd) * (ii < end) + rmax * (ii >= end);*/

	//read and store the coordinates on next tiles for each block
	/*for (uint i = tix + start + blockDim.x;i<end;i += blockDim.x)
	{
		Type d= indata[i];

		//count items smaller or bigger than the pivot
		// lo = ( d <  pivot ) *(lo+1) + ( d >= pivot )*lo;
		// hi = ( d <= pivot ) *(hi)   +  (d >  pivot )*(hi+1);
		lo += (d < pivot);
		hi += (d > pivot);
		// lo = ((d < pivot) ? (lo + 1) : (lo));
		// hi = ((d > pivot) ? (hi + 1) : (hi));

		//compute max and min of tile items
		lmin = min(lmin,d);
		rmax = max(rmax,d);

	}*/

	//compute max and min of every partition

	compareInclusive(rmax, lmin, (Type*)sh_out, blockDim.x);
	// printf("Hrer");
	__syncthreads();

	if (tix == blockDim.x - 1)
	{
		//compute absolute max and min for the bucket
		atomicMax(&bucket[nseq].maxPiv, rmax);
		atomicMin(&bucket[nseq].minPiv, lmin);
	}
	__syncthreads();


	/*
	 * calculate the coordinates of its assigned item to each thread,
	 * which are necessary to known in which subsequences the item must be copied
	 *
	 */

	scan1Inclusive2(lo, hi, (uint*)sh_out, blockDim.x);
	lo = lo - 1;
	hi = shmem - hi;

	if (tix == blockDim.x - 1)
	{
		*left = lo + 1;
		*right = shmem - hi;

		*start1 = atomicAdd(&bucket[nseq].nextbegin, *left);
		*end1 = atomicSub(&bucket[nseq].nextend, *right);
	}

	__syncthreads();

	//if (threadIdx.x == 0 && blockIdx.x == 0) {
	//	printf("%d, %d, %d < %d\n", lo, hi, d, pivot);
	//}

	/*if (threadIdx.x == 0 && blockIdx.x == 0) {
		// printf("%d, %d, %d, %d, %d\n", d, pivot, (iii < end), (d < pivot), ((iii < end) && (d < pivot)));
		printf("%d\n", sh_out[hi]);
		printf("%d\n", sh_out[lo]);
		printf("%d, %d, %d < %d\n", lo, hi, d, pivot);
	}*/

	//thread blocks write on the shared memory the items smaller and bigger than the first tile's pivot
	//uint iii = tix + start;
	//sh_out[lo] = d * (iii < end) * (d < pivot)/* + (sh_out[lo] * (iii >= end)) + (sh_out[lo] * ((iii < end) & (d >= pivot)))*/;
	//lo -= ((iii < end) & (d < pivot));
	//sh_out[hi] = d * (iii < end) * (d > pivot)/* + (sh_out[hi] * (iii >= end)) + (sh_out[hi] * ((iii < end) & (d <= pivot)))*/;
	//hi += ((iii < end) & (d > pivot));*/

	// for (int i = 0; i < shmem; i += blockDim.x) {
	//	sh_out[i] = 0;
	//}

	/*uint iii = tix + start;
	sh_out[lo] = d * (iii < end) * (d < pivot) + (sh_out[lo] * (iii >= end)) + (sh_out[lo] * ((iii < end) & (d >= pivot)));
	lo -= (iii < end) * (d < pivot);
	sh_out[hi] = d * (iii < end) * (d > pivot) + (sh_out[hi] * (iii >= end)) + (sh_out[hi] * ((iii < end) & (d <= pivot)));
	hi += (iii < end) * (d > pivot);*/
	// uint iii = tix + start;
	// uint loIndex = lo * (iii < end) * (d < pivot) + 

	/*uint iii = tix + start;
	bool lo1 = ((iii < end) && (d < pivot));
	// uint loIndex = lo * lo1 + (shmem + tix) * !lo1;
	bool hi1 = ((iii < end) && (d > pivot));
	// uint hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	uint index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	// sh_out[loIndex] = d * lo1;
	sh_out[index] = d;
	lo -= lo1;
	// sh_out[hiIndex] = d * hi1;
	hi += hi1;*/

	uint iii = tix + start;
	bool lo1 = ((iii < end) && (d < pivot));
	bool hi1 = ((iii < end) && (d > pivot));
	uint index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = d;
	lo -= lo1;
	hi += hi1;

	/*if (tix + start<end)
	{
		//items smaller than pivot
		if(d<pivot)
			{sh_out[lo]=d; lo--;}

		//items bigger than pivot
		if(d>pivot)
			{sh_out[hi]=d; hi++;}
	}*/

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	hi1 = ((iii < end) && (dd > pivot));
	index = lo * lo1 + hi * hi1 + ((shmem + tix) * !lo1 * !hi1);
	sh_out[index] = dd;
	lo -= lo1;
	hi += hi1;

	/*iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	// 4 + 4
	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;*/


	// 8 + 8
	/*iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;

	iii += blockDim.x;
	dd = indata[iii];
	lo1 = ((iii < end) && (dd < pivot));
	loIndex = lo * lo1 + (shmem + tix) * !lo1;
	hi1 = ((iii < end) && (dd > pivot));
	hiIndex = hi * hi1 + (shmem + tix) * !hi1;
	sh_out[loIndex] = dd * lo1;
	lo -= lo1;
	sh_out[hiIndex] = dd * hi1;
	hi += hi1;*/

	//thread blocks write on the shared memory the items smaller and bigger than next tiles' pivot
	/*for (uint i = start + tix + blockDim.x;i<end;i += blockDim.x)
	{

		Type d=indata[i];
		//items smaller than the pivot
		if(d<pivot)
			{sh_out[lo--]=d;}

		//items bigger than the pivot
		if(d>pivot)
			{sh_out[hi++]=d;}


	}*/

	__syncthreads();
	iii = tix;
	bool buffer1 = (iii < (*left)); bool buffer2 = iii >= shmem - (*right);
	int bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	// 4 + 4
	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	// 8 + 8
	/*iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];

	iii += blockDim.x;
	buffer1 = (iii < (*left)); buffer2 = iii >= shmem - (*right);
	bufferIndex = ((*start1) + iii) * buffer1 + ((*end1) + iii - shmem) * buffer2 + ((size + threadIdx.x) * !buffer1 * !buffer2);
	buffer[bufferIndex] = sh_out[iii];*/



	//items smaller and bigger than the pivot already sorted in the shared memory are coalesced written on the global memory
	//partial results of each thread block stored on the shared memory are merged together in two subsequences within the global memory
	//coalesced writing of next tiles on the global memory
	/*for (uint i = tix;i<shmem;i += blockDim.x)
	{
		if (i<*left)
			buffer[*start1+i]=sh_out[i];

		if(i>=shmem-*right)
			buffer[*end1+i-shmem]=sh_out[i];
	}*/

}



//this function assigns the attributes to each partition of each bucket
//a thread block is assigned to a specific partition
template <typename Type>
__global__ void partitionAssign(struct Block<Type>* bucket, uint* npartitions, struct Partition<Type>* partition, int shmem, int partitionSize, int nbucket)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	uint beg = bucket[bx].nextbegin;
	uint end = bucket[bx].nextend;
	Type pivot = bucket[bx].pivot;
	uint from;
	uint to;

	bool uslov = bx > 0;
	from = npartitions[(bx - 1) * uslov] * uslov;
	to = npartitions[bx];
	/*if (bx>0)
	{
		from=npartitions[bx-1];
		to=npartitions[bx];
	}
	else
	{
		from=0;
		to=npartitions[bx];
	}*/


	uint i = tx + from;


	/*uslov = i < to;
	uint begin = (beg + shmem * tx);
	int partitionIndex = (i * uslov) + ((partitionSize) * !uslov);
	partition[partitionIndex].from = begin * uslov;
	partition[partitionIndex].end = (begin + shmem) * uslov;
	partition[partitionIndex].pivot= pivot * uslov;
	partition[partitionIndex].ibucket= bx * uslov + (nbucket + 1) * !uslov;*/

	if (i < to)
	{
		uint begin = beg + shmem * tx;
		partition[i].from = begin;
		partition[i].end = begin + shmem;
		partition[i].pivot = pivot;
		partition[i].ibucket = bx;

	}


	for (uint i = tx + from + blockDim.x;i < to;i += blockDim.x)
	{
		uint begin = beg + shmem * (i - from);
		partition[i].from = begin;
		partition[i].end = begin + shmem;
		partition[i].pivot = pivot;
		partition[i].ibucket = bx;
	}
	__syncthreads();
	if (tx == 0 && to - from > 0) partition[to - 1].end = end;


}

//this function enters the pivot value in the central bucket's items
template <typename Type>
__global__ void insertPivot(Type* data, struct Block<Type>* bucket, int nbucket)
{

	Type pivot = bucket[blockIdx.x].pivot;
	uint start = bucket[blockIdx.x].nextbegin;
	uint end = bucket[blockIdx.x].nextend;
	bool is_altered = bucket[blockIdx.x].done;

	if (is_altered && blockIdx.x < nbucket)
		for (uint j = start + threadIdx.x; j < end; j += blockDim.x)
			data[j] = pivot;


}


//this function assigns the new attributes of each bucket
template <typename Type>
__global__ void bucketAssign(Block<Type>* bucket, uint* npartitions, int nbucket, int select, int shmem, int minSize)
{

	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nbucket) {
		bool is_altered = bucket[i].done;
		if (is_altered)
		{
			//read on i node
			uint orgbeg = bucket[i].begin;
			uint from = bucket[i].nextbegin;
			uint orgend = bucket[i].end;
			uint end = bucket[i].nextend;
			Type pivot = bucket[i].pivot;
			Type minPiv = bucket[i].minPiv;
			Type maxPiv = bucket[i].maxPiv;

			//compare each bucket's max and min to the pivot
			Type lmaxpiv = min(pivot, maxPiv);
			Type rminpiv = max(pivot, minPiv);

			//write on i+nbucket node
			bucket[i + nbucket].begin = orgbeg;
			bucket[i + nbucket].nextbegin = orgbeg;
			bucket[i + nbucket].nextend = from;
			bucket[i + nbucket].end = from;
			bucket[i + nbucket].pivot = (minPiv + lmaxpiv) / 2;

			//if(select)
			//	bucket[i+nbucket].done   = (from-orgbeg)>1024;// && (minPiv!=maxPiv);
			//else
			bucket[i + nbucket].done = (from - orgbeg) > (minSize / 2) && (minPiv != maxPiv);
			bucket[i + nbucket].select = select;
			bucket[i + nbucket].minPiv = 0xffffffffffffffff;
			bucket[i + nbucket].maxPiv = 0;
			//bucket[i+nbucket].finish=false;

			//calculate the number of partitions (npartitions) necessary to the i+nbucket bucket
			/*if (!bucket[i + nbucket].done)
				 npartitions[i+nbucket] = 0;
			else npartitions[i+nbucket] = (from-orgbeg+shmem-1)/shmem;*/
			npartitions[i + nbucket] = bucket[i + nbucket].done * (from - orgbeg + shmem - 1) / shmem;

			//write on i node
			bucket[i].begin = end;
			bucket[i].nextbegin = end;
			bucket[i].nextend = orgend;
			bucket[i].pivot = (rminpiv + maxPiv) / 2 + 1;

			//if(select)
				//bucket[i].done   = (orgend-end)>1024;// && (minPiv!=maxPiv);
			//	else
			bucket[i].done = (orgend - end) > (minSize / 2) && (minPiv != maxPiv);
			bucket[i].select = select;
			bucket[i].minPiv = 0xffffffffffffffff;
			bucket[i].maxPiv = 0;
			//bucket[i].finish=false;

			//calculate the number of partitions (npartitions) necessary to the i-bucket bucket
			npartitions[i] = bucket[i].done * (orgend - end + shmem - 1) / shmem;
			/*if (!bucket[i].done)
				npartitions[i]=0;
			else
				npartitions[i]=(orgend-end+shmem-1)/shmem;*/

		}
	}


}



template <typename Type>
__global__ void init(Type* data, Block<Type>* bucket, uint* npartitions, int size, int nblocks)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < nblocks)
	{
		bucket[i].nextbegin = 0;
		bucket[i].begin = 0;

		bucket[i].nextend = 0 + size * (i == 0);
		bucket[i].end = 0 + size * (i == 0);
		npartitions[i] = 0;
		bucket[i].done = false + i == 0;
		bucket[i].select = false;
		bucket[i].maxPiv = 0x0;
		bucket[i].minPiv = 0xffffffffffffffff;
		bucket[i].pivot = 0 + (i == 0) * ((min(min(data[0], data[size / 2]), data[size - 1]) + max(max(data[0], data[size / 2]), data[size - 1])) / 2);
	}

}



template <typename Type>
void sort(Type* inputData, Type* outputData, uint size, uint threadCount, int device, double* wallClock)
{
	int shmem = threadCount * 8;
	printf("%d\n", shmem);
	cudaSetDevice(device);

	cudaGetLastError();
	//cudaDeviceReset();

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	StopWatchInterface* htimer = NULL;
	StopWatchInterface* btimer = NULL;
	Type* ddata;
	Type* dbuffer;

	Block<Type>* dbucket;
	struct Partition<Type>* partition;
	uint* npartitions1, * npartitions2;

	uint* cudaBlocks = (uint*)malloc(4);

	uint blocks = (size + shmem - 1) / shmem;
	int partition_max = 262144;
	uint partitions = 2 * blocks * sizeof(Partition<Type>);

	unsigned long long int total = partition_max * sizeof(Block<Type>) + blocks * sizeof(Partition<Type>) + 2 * partition_max * sizeof(uint) + 3 * (size) * sizeof(Type);

	// printf("%d\n", deviceProp.sharedMemPerBlock);
	printf("\nINFO: Device Memory consumed is %.3f GB out of %.3f GB of available memory\n", ((double)total / GIGA), (double)deviceProp.totalGlobalMem / GIGA);

	//Allocating and initializing CUDA arrays
	sdkCreateTimer(&htimer);
	sdkCreateTimer(&btimer);
	checkCudaErrors(cudaMalloc((void**)&dbucket, partition_max * sizeof(Block<Type>)));
	checkCudaErrors(cudaMalloc((void**)&partition, partitions + 4 * sizeof(Partition<Type>))); //nblock


	checkCudaErrors(cudaMalloc((void**)&npartitions1, partition_max * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&npartitions2, partition_max * sizeof(uint)));

	checkCudaErrors(cudaMalloc((void**)&dbuffer, (size) * sizeof(Type) + threadCount * sizeof(Type)));
	checkCudaErrors(cudaMalloc((void**)&ddata, (size) * sizeof(Type) + threadCount * sizeof(Type)));

	checkCudaErrors(cudaMemcpy(ddata, inputData, size * sizeof(Type), cudaMemcpyHostToDevice));

	initScan();

	//setting GPU Cache
	cudaFuncSetCacheConfig(init<Type>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(insertPivot<Type>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(bucketAssign<Type>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(partitionAssign<Type>, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(quick<Type>, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(globalBitonicSort<Type>, cudaFuncCachePreferShared);


	// 128 threads - 2048 elements, bucket limit - 4096, threads for bitonic 2048 
	uint maxBucket = 4 * shmem > 2048 ? 2048 : 4 * shmem;
	uint maxThreadBucket = maxBucket / 2 > 1024 ? 1024 : maxBucket / 2;
	uint bitonicBlockMult = (maxBucket / 2048) > 0 ? (maxBucket / 2048) : 1;

	checkCudaErrors(cudaDeviceSynchronize());
	sdkResetTimer(&htimer);
	sdkResetTimer(&btimer);
	sdkStartTimer(&htimer);

	//initializing bucket array: initial attributes for each bucket
	init<Type> << <(2 * blocks + 255) / 256, 256 >> > (ddata, dbucket, npartitions1, size, partition_max);

	// uint maxBucket = ((2 * shmem > 2048) ? (2048) : (shmem));
	uint nbucket = 1;
	uint numIterations = 0;
	bool inputSelect = true;

	*cudaBlocks = blocks;
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("init() execution FAILED\n");
	checkCudaErrors(cudaMemcpy(&npartitions2[0], cudaBlocks, sizeof(uint), cudaMemcpyHostToDevice));


	// beginning of the first phase
	// this phase goes on until the size of the buckets is comparable to the SHARED_LIMIT size
	while (1)
	{

		/*
		 *       	---------------------    Pre-processing: Partitioning    ---------------------
		 *
		 * buckets are further divided in partitions based on their size
		 * the number of partitions needed for each subsequence is determined by the number of elements which can be
		 * processed by each thread block.
		 *
		 * the number of partitions (npartitions) for each block will depend on the shared memory size (SHARED_LIMIT)
		 *
		 */

		if (numIterations > 0)
		{	//1024 is the shared memory limit of scanInclusiveShort()
			if (nbucket <= 1024)
				scanInclusiveShort(npartitions2, npartitions1, 1, nbucket);
			else
				scanInclusiveLarge(npartitions2, npartitions1, 1, nbucket);

			checkCudaErrors(cudaMemcpy(cudaBlocks, &npartitions2[nbucket - 1], sizeof(uint), cudaMemcpyDeviceToHost));
		}

		if (*cudaBlocks == 0)
			break;


		/*
		 *  ---------------------     step 1    ---------------------
		 *
		 * 	A thread block is assigned to each different partition
		 * 	each partition is assigned coordinates, pivot and ....
		 */


		partitionAssign<Type> << <nbucket, 1024 >> > (dbucket, npartitions2, partition, shmem, partitions, nbucket);
		cudaDeviceSynchronize();
		getLastCudaError("partitionAssign() execution FAILED\n");

		/*
			 ---------------------    step 2a    ---------------------

			 in this function each thread block creates two subsequences
			 to divide the items in the partition whose value is lower than
			 the pivot value, from the items whose value is higher than the pivot value
		 */

		if (inputSelect)
			quick<Type> << <*cudaBlocks, threadCount, 2 * sizeof(Type) * shmem + 256 >> > (ddata, dbuffer, partition, dbucket, shmem, size);
		else
			quick<Type> << <*cudaBlocks, threadCount, 2 * sizeof(Type) * shmem + 256 >> > (dbuffer, ddata, partition, dbucket, shmem, size);
		cudaDeviceSynchronize();
		getLastCudaError("quick() execution FAILED\n");

		//step 2b: this function enters the pivot value in the central bucket's items
		insertPivot<Type> << <nbucket, 512 >> > (ddata, dbucket, nbucket);


		//step 3: parameters are assigned, linked to the two new buckets created in step 2
		bucketAssign<Type> << <(nbucket + 255) / 256, 256 >> > (dbucket, npartitions1, nbucket, inputSelect, shmem, maxBucket);
		cudaDeviceSynchronize();
		getLastCudaError("insertPivot() or bucketAssign() execution FAILED\n");

		nbucket *= 2;

		inputSelect = !inputSelect;
		numIterations++;
		printf("%d--", numIterations);

		if (nbucket > (deviceProp.maxGridSize[0]))
			break;
		// if(numIterations==9) break;
	}

	/*
	 * start second phase:
	 * now the size of the buckets is such that they can be entirely processed by a thread block
	 *
	 */

	sdkStopTimer(&htimer);
	*wallClock = sdkGetTimerValue(&htimer);
	printf("\nIteracija: %d\n", numIterations);
	printf("Quicksort exec time: %f\n", sdkGetTimerValue(&htimer));

	sdkStartTimer(&btimer);
	if (nbucket > deviceProp.maxGridSize[0])
		fprintf(stderr, "ERROR: CUDA-Quicksort can't terminate sorting as the block threads needed to finish it are more than the Maximum x-dimension of FERMI GPU thread blocks. Please use Kepler GPUs as the Maximum x-dimension of their thread blocks is much higher\n");
	else
		globalBitonicSort<Type> << <bitonicBlockMult * nbucket, maxThreadBucket, sizeof(uint)* maxBucket >> > (ddata, dbuffer, dbucket, inputSelect, maxBucket, bitonicBlockMult);

	cudaDeviceSynchronize();
	sdkStopTimer(&btimer);
	printf("Bitonic sort exec time: %f\n\n", sdkGetTimerValue(&btimer));

	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("globalBitonicSort() execution FAILED\n");




	// Copy the final result to the CPU in the outputData array
	checkCudaErrors(cudaMemcpy(outputData, ddata, size * sizeof(Type), cudaMemcpyDeviceToHost));

	// release resources
	checkCudaErrors(cudaFree(ddata));
	checkCudaErrors(cudaFree(dbuffer));
	checkCudaErrors(cudaFree(dbucket));
	checkCudaErrors(cudaFree(npartitions2));
	checkCudaErrors(cudaFree(npartitions1));
	free(cudaBlocks);

	closeScan();
	return;
}



extern "C"
void CUDA_Quicksort(uint * inputData, uint * outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, Device);

	if (deviceProp.major < 2)
	{
		fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
			Device, deviceProp.major, deviceProp.minor);

		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		fprintf(stderr, "       the Host system has the following GPU devices:\n");

		for (int device = 0; device < deviceCount; device++) {

			fprintf(stderr, "\t  the GPU device %d is a %s, with Compute Capability %d.%d\n",
				device, deviceProp.name, deviceProp.major, deviceProp.minor);
		}

		return;
	}

	sort<uint>(inputData, outputData, dataSize, threadCount, Device, wallClock);
}

extern "C"
void CUDA_Quicksort_64(double* inputData, double* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, Device);

	if (deviceProp.major < 2)
	{
		fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
			Device, deviceProp.major, deviceProp.minor);

		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		fprintf(stderr, "       the Host system has the following GPU devices:\n");

		for (int device = 0; device < deviceCount; device++) {

			fprintf(stderr, "\t  the GPU device %d is a %s, with Compute Capability %d.%d\n",
				device, deviceProp.name, deviceProp.major, deviceProp.minor);
		}

		return;
	}

	sort<double>(inputData, outputData, dataSize, threadCount, Device, wallClock);

}
