#pragma once
#include <vector>
#include <immintrin.h>

using std::vector;

//Fastest known algorithm to add two STL vectors, to the extent of my knowledge
//Requires a vector with length divisible by 8 (for chunking).
//As such, make sure to pad vectors whose v.size()%8 != 0 such that it is the case.

void AvxSum(vector<float> &a, vector<float> &b){
	int n = a.size();
	for (size_t i = 0; i < n; i += 8)
	{
		__m256 va = _mm256_loadu_ps(&a[i]);
		__m256 vb = _mm256_loadu_ps(&b[i]);
		va = _mm256_add_ps(va, vb);
		_mm256_storeu_ps(&a[i], va);
	}
}
