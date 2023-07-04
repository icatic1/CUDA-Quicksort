#include <cstdlib>
#include <algorithm>
#include <cmath>

using namespace std;

template <typename Type> inline
void zero (Type* data,int N)
{
	//srand (time(NULL));
	srand(1);
	Type value = rand();
	for (int i = 0; i < N; ++i)
		data[i] = value ;
}

template <typename Type> inline
void uniform(Type* data,int N)
{
	srand(time(NULL));
	for (int i = 0; i < N; ++i)
		if (rand() > 16384) {
			data[i] = (Type)((rand() << 15) | (rand()));
		}
		else {
			data[i] = (Type)(rand());
		}
			
}

template <typename Type> inline
void sorted(Type* data,int N)
{
	//srand (time(NULL));
	srand(1);
	for (int i = 0; i < N; ++i)
		data[i] = rand()%16384 ;

	std::sort(data,data+N);
}

template <typename Type> inline
void inverseSorted(Type* data, int N)
{
	srand(time(NULL));
	for (int i = 0; i < N; ++i)
		data[i] = rand() % 16384;

	std::sort(data, data + N, std::greater<>());
}

template <typename Type> inline
void gaussian(Type* data,int N)
{
	srand (time(NULL));
	int value = 0;
	for (int i = 0; i < N; ++i) {
		value = 0;
		if (rand() > 16384) {
			value += (Type)((rand() << 15) | (rand()));
		}
		else {
			value += (Type)(rand());
		}
		
		/*
		for (int j = 0; j < 4; ++j) {
			value += rand()%16384;
		}*/

		data[i] = value /4;

	}
}

template <typename Type>
void distribution(Type* data, int dataSize, std::string& typeDist)
{
	if(typeDist == "uniform")
	{
		uniform(data,dataSize);
	}
	else if(typeDist == "sorted")
	{
		sorted(data,dataSize);
	}
	else if (typeDist == "inverseSorted") {
		inverseSorted(data, dataSize);
	}
	else if (typeDist == "zero")
	{
		zero(data, dataSize);
	}
	else if(typeDist == "gaussian")
	{
		gaussian(data,dataSize);
	}
	else {
		uniform(data, dataSize);
	}
}

