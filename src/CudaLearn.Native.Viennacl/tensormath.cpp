#include "common.hpp"

typedef double ScalarType;

typedef enum BlasTranspose
{
	None = 0,
	Transpose = 1
};

extern "C" {

	DLLEXPORT int vclGetVersion()
	{
		return 162;
	};

	DLLEXPORT void vclGemm(const BlasTranspose transA, const BlasTranspose transB,
						const int m, const int n, const int k, 
						const ScalarType alpha, const ScalarType a[], const int aOffset, 
						const ScalarType b[], const int bOffset,
						const float beta, ScalarType c[], const int cOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;
		auto cPtr = &(c[0]) + cOffset;

		int lda = transA == BlasTranspose::None ? m : k;
		int ldb = transB == BlasTranspose::None ? k : n;

		int aSize1 = (transA == BlasTranspose::Transpose) ? k : m;
		int aSize2 = (transA == BlasTranspose::Transpose) ? m : k;
		
		int bSize1 = (transB == BlasTranspose::Transpose) ? n : k;
		int bSize2 = (transB == BlasTranspose::Transpose) ? k : n;

		viennacl::matrix<ScalarType> aMatrix((ScalarType*)aPtr, viennacl::MAIN_MEMORY, aSize1, aSize2);
		viennacl::matrix<ScalarType> bMatrix((ScalarType*)bPtr, viennacl::MAIN_MEMORY, bSize1, bSize2);
		viennacl::matrix<ScalarType> cMatrix(cPtr, viennacl::MAIN_MEMORY, m, n);


	};

	DLLEXPORT void vclGemv(const BlasTranspose transA, const int m, const int n, 
						const ScalarType alpha, const ScalarType a[], const int aOffset, const int aLength, 
						const ScalarType x[], const int xOffset, const int xLength, 
						const ScalarType beta, ScalarType y[], const int yOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		// wrap host buffer within ViennaCL
		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, xLength);
		viennacl::matrix<ScalarType> aMatrix((ScalarType*)aPtr, viennacl::MAIN_MEMORY, m, n);
		viennacl::matrix<ScalarType> yMatrix(yPtr, viennacl::MAIN_MEMORY, m, n);

		// auto aux = alpha * viennacl::linalg::prod(aMatrix, xVec) + beta * yMatrix;
	};
	
	DLLEXPORT void vclAxpy(const int n, 
							const ScalarType alpha, const ScalarType x[], const int xOffset,
							ScalarType y[], const int yOffset)
	{
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		// wrap host buffer within ViennaCL vectors:
		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> yVec(yPtr, viennacl::MAIN_MEMORY, n);

		viennacl::linalg::avbv(yVec,
			xVec, alpha, n, false, false,
			yVec, 1.0, n, false, false);

	};

	DLLEXPORT void vclAxpby(const int n, const ScalarType alpha, const ScalarType x[], const int xOffset,
							 ScalarType beta, ScalarType y[], const int yOffset)
	{
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		// wrap host buffer within ViennaCL vectors:
		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> yVec(yPtr, viennacl::MAIN_MEMORY, n);

		viennacl::linalg::avbv(yVec,
			xVec, alpha, n, false, false,
			yVec, beta, n, false, false);
	};

	DLLEXPORT void vclSet(const int n, const ScalarType alpha, ScalarType y[], const int yOffset)
	{
		auto yPtr = &(y[0]) + yOffset;

		viennacl::vector<ScalarType> yVec(yPtr, viennacl::MAIN_MEMORY, n);
		viennacl::linalg::vector_assign(yVec, alpha);
	};


	DLLEXPORT void vclCopy(const int n, const ScalarType x[], const int xOffset, ScalarType y[], const int yOffset)
	{
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> yVec(yPtr, viennacl::MAIN_MEMORY, n);

		yVec = xVec;
	};

	DLLEXPORT void vclAddScalar(const int n, const ScalarType alpha, ScalarType y[], const int yOffset)
	{
		auto yPtr = &(y[0]) + yOffset;

		viennacl::vector<ScalarType> yVec(yPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> scalarVec ( viennacl::scalar_vector<ScalarType>(n, alpha) );

		yVec += scalarVec;
	};

	DLLEXPORT void vclAdd(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec += aVec;
	};
	DLLEXPORT void vclSubstract(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = aVec - bVec;
	};

	DLLEXPORT void vclMultiply(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = viennacl::linalg::element_prod(aVec, bVec);
	};
	DLLEXPORT void vclDivide(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = viennacl::linalg::element_div(aVec, bVec);
	};
	DLLEXPORT void vclPowx(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = viennacl::linalg::element_pow(aVec, bVec);
	};

	DLLEXPORT void vclSquare(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = viennacl::linalg::element_prod(aVec, aVec);
	};
	DLLEXPORT void vclExp(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = viennacl::linalg::element_exp(aVec);
	};

	DLLEXPORT void vclAbs(const int n, const ScalarType a[], const int aOffset, ScalarType b[], const int bOffset)
	{
		auto aPtr = &(a[0]) + aOffset;
		auto bPtr = &(b[0]) + bOffset;

		viennacl::vector<ScalarType> aVec((ScalarType*)aPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> bVec(bPtr, viennacl::MAIN_MEMORY, n);

		bVec = viennacl::linalg::element_abs(aVec);
	};

	DLLEXPORT ScalarType vclDot(const int n, const ScalarType x[], const int xOffset, const ScalarType y[], const int yOffset)
	{
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> yVec((ScalarType*)yPtr, viennacl::MAIN_MEMORY, n);

		return viennacl::linalg::inner_prod(xVec, yVec);
	};

	DLLEXPORT ScalarType vclDotEx(const ScalarType x[], const int xOffset, const int xLength, const int incx,
								  const ScalarType y[], const int yOffset, const int yLength, const int incy)
	{
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, xLength);
		viennacl::vector<ScalarType> yVec((ScalarType*)yPtr, viennacl::MAIN_MEMORY, yLength);


		return 0;
	};

	DLLEXPORT ScalarType vclAsum(const int n, const ScalarType x[], const int xOffset)
	{
		auto xPtr = &(x[0]) + xOffset;

		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, n);
		return viennacl::linalg::norm_1(xVec);
	};

	DLLEXPORT void vclScaleInline(const int n, const ScalarType alpha, ScalarType x[], const int xOffset)
	{
		auto xPtr = &(x[0]) + xOffset;

		viennacl::vector<ScalarType> xVec(xPtr, viennacl::MAIN_MEMORY, n);
		xVec *= alpha;		
	};

	DLLEXPORT void vclScale(const int n, const ScalarType alpha, const ScalarType x[], const int xOffset, ScalarType y[], const int yOffset)
	{
		auto xPtr = &(x[0]) + xOffset;
		auto yPtr = &(y[0]) + yOffset;

		viennacl::vector<ScalarType> xVec((ScalarType*)xPtr, viennacl::MAIN_MEMORY, n);
		viennacl::vector<ScalarType> yVec((ScalarType*)yPtr, viennacl::MAIN_MEMORY, n);

		yVec = xVec * alpha;
	};
}