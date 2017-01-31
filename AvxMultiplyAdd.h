#pragma once
#include <vector>
#include <immintrin.h>

//multiply a with b, add result to result
//Requires a vector with length divisible by 8 (for chunking).
//As such, make sure to pad vectors whose v.size()%8 != 0 such that it is the case.
//Should be redone with FMA (in a single instruction instead of 2)

void AVXMultiplySum(vector<float> a, vector<float> b, vector<float> result)
{
	int n = a.size();
	for (size_t i = 0; i < n; i += 8)
	{
		__m256 va = _mm256_loadu_ps(&a[i]);
		__m256 vb = _mm256_loadu_ps(&b[i]);
		__m256 vres = _mm256_loadu_ps(&result[i]);
		vres = _mm256_add_ps(vres, _mm256_add_ps(va, vb));
		_mm256_storeu_ps(&result[i], vres);
	}
}
