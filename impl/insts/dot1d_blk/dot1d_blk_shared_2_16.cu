#include "dot1d_blk.cuh"

/*	Difference :
	Au lieux de directement deriver avec que des atomicAdd le
__shared__ noyau, on fait la méthode que j'avais avant
ou on fait une autre opération pour calc dx et dp.

	Mathématiquement ca correspond a deriver y=X@P+B
en dX=p@dY.T
	dx = (p @ ((y-_y)*dtanh(x@p)).T).T
	dp = x.T @ ((y-_y)*dtanh(x@p))
*/

#define BLOQUE 16
#define BLOQUE_MAX 16

static __global__ void kerd_blk_stricte_16__shared2(
	uint X_blk, uint Y_blk, uint P_blk,
	uint depart_y, uint depart_x, uint depart_p,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	// <KERD(T, BLOQUE), KERD(Y,BLOQUE)>
	// <         BLOQUE,         BLOQUE>

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _y = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	FOR(0, d, X_blk/BLOQUE) {
		__partage__x[thy][thx] = x[depart_x+(depart+_t)*( X_vars ) + DEPART_x +d*BLOQUE + thx];
		__partage__p[thy][thx] = p[depart_p+_y*(X_blk+1) + d*BLOQUE + thy];
		__syncthreads();

	//#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

#define __partage__b __partage__x[0]

	if (thy == 0) __partage__b[thx] = p[depart_p+_y*(X_blk+1) + (X_blk+1-1)];
	__syncthreads();

	s = (s + __partage__b[thx]);
	float a = dot1d_blk_ACTIV(dot1d_blk_ACTIVATION, s);
	   y[(0+_t)*Y + depart_y+_y] = a;
	locd[(0+_t)*Y + depart_y+_y] = dot1d_blk_dACTIV(dot1d_blk_ACTIVATION, s,a);
};

void nvidia_dot1d_blk_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	//
	ASSERT(X % (DOT1D_BLK_BLOQUES*BLOQUE_MAX) == 0);
	ASSERT(Y % (DOT1D_BLK_BLOQUES*BLOQUE)     == 0);
	//
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	if (T%BLOQUE!=0) ERR("ATTENTION T%%16 != 0 (T=%i)", T);
	if (X_blk%BLOQUE==0 && Y_blk%BLOQUE==0 && T%BLOQUE==0) {
		FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
			uint depart_y = dot1d_blk * Y_blk;	//depat d'un bloque en x
			uint depart_x = dot1d_blk * X_blk;	// 					en y
			uint depart_p = dot1d_blk * P_blk;
			kerd_blk_stricte_16__shared2<<<dim3(KERD(Y_blk, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
				X_blk, Y_blk, P_blk,
				depart_y, depart_x, depart_p,
				X_vars, Y_vars,
				X, Y,
				depart, T,
				DEPART_x,
				x, y,
				p,
				locd);
		}
		ATTENDRE_CUDA();
	} else {
		nvidia_dot1d_blk_naive(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd);
	}
}

static __global__ void kerd_blk_stricte_16__shared2____dx(
	uint X_blk, uint Y_blk, uint P_blk,
	uint depart_y, uint depart_x, uint depart_p,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T

	__shared__ float __partage__x[BLOQUE][BLOQUE];
	__shared__ float __partage__p[BLOQUE][BLOQUE];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	//uint _y = thx + blockIdx.x * blockDim.x;
	uint _x = thx + blockIdx.x * blockDim.x;
	uint _t = thy + blockIdx.y * blockDim.y;

	float s = 0;

	FOR(0, d, Y_blk/BLOQUE) {
		__partage__x[thy][thx] = locd[depart_y+(depart+_t)*Y+d*BLOQUE+thx] * dy[depart_y+(depart+_t)*Y+d*BLOQUE+thx];//x[(depart+_t)*( X ) + d*BLOQUE + thx];
		__partage__p[thy][thx] = p[depart_p+(d*BLOQUE+thy)*(X_blk+1) + _x];//p[_y*(X+1) + d*BLOQUE + thy];
		__syncthreads();

	//#pragma unroll
		FOR(0, i, BLOQUE) s += __partage__x[thy][i] * __partage__p[i][thx];
		__syncthreads();
	};

	dx[(depart+_t)*X_vars+DEPART_x +depart_x+_x] = s;
	//printf("s=%f\n", s);
};


static __global__ void kerd_blk_stricte_32__shared2____dp(
	uint X_blk, uint Y_blk, uint P_blk,
	uint depart_y, uint depart_x, uint depart_p,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	//dp = x.T @ ((y-_y)*dtanh(x@p))

	__shared__ float __partage__x[BLOQUE_MAX][BLOQUE_MAX];
	__shared__ float __partage__p[BLOQUE_MAX][BLOQUE_MAX];

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	uint _x = thx + blockIdx.x * blockDim.x;
	uint _y = thy + blockIdx.y * blockDim.y;

	float s = 0;
	float biais = 0;

	uint d = blockIdx.z;
	//FOR(0, d, T/BLOQUE) {
		__partage__x[thy][thx] = locd[depart_y+(depart+d*BLOQUE_MAX+thx)*Y+_y] * dy[depart_y+(depart+d*BLOQUE_MAX+thx)*Y+_y];//x[(depart+_t)*( X ) + d*BLOQUE + thx];
		__partage__p[thy][thx] = x[depart_x+(depart+(d*BLOQUE_MAX+thy))*X_vars+DEPART_x +_x];//p[_y*(X+1) + d*BLOQUE + thy];
		__syncthreads();

	#pragma unroll
		FOR(0, i, BLOQUE_MAX) {
			s += __partage__x[thy][i] * __partage__p[i][thx];
			if (_x == 0) biais += __partage__x[thy][i];
		}
		__syncthreads();
	//};

#define __partage__b __partage__x[0]

	//if (thy == 0) __partage__b[thx] = p[_y*(X+1) + (X+1-1)];
	if (_x == 0) atomicAdd(&dp[depart_p+_y*(X_blk+1) + (X_blk+1-1)], biais);
	__syncthreads();

	atomicAdd(&dp[depart_p+_y*(X_blk+1)+_x], s);
};

void d_nvidia_dot1d_blk_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	ASSERT(X % (DOT1D_BLK_BLOQUES*BLOQUE_MAX) == 0);
	ASSERT(Y % (DOT1D_BLK_BLOQUES*BLOQUE)     == 0);
	//
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	if (T%(MAX2(BLOQUE_MAX,BLOQUE))!=0) ERR("ATTENTION T%%%i != 0 (T=%i)", T, (MAX2(BLOQUE_MAX,BLOQUE)));
	if (X_blk%BLOQUE==0 && Y_blk%BLOQUE==0 && T%BLOQUE==0) {
		FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
			uint depart_y = dot1d_blk * Y_blk;	//depat d'un bloque en x
			uint depart_x = dot1d_blk * X_blk;	// 					en y
			uint depart_p = dot1d_blk * P_blk;
			//
			kerd_blk_stricte_16__shared2____dx<<<dim3(KERD(X_blk, BLOQUE), KERD(T, BLOQUE)), dim3(BLOQUE, BLOQUE)>>>(
				X_blk, Y_blk, P_blk,
				depart_y, depart_x, depart_p,
				X_vars, Y_vars,
				X, Y,
				depart, T,
				DEPART_x,
				x, y,
				p,
				locd,
				dy,
				dx,
				dp);
			kerd_blk_stricte_32__shared2____dp<<<dim3(KERD(X_blk, BLOQUE_MAX), KERD(Y_blk, BLOQUE_MAX), DIV(T,BLOQUE_MAX)), dim3(BLOQUE_MAX, BLOQUE_MAX,1)>>>(
				X_blk, Y_blk, P_blk,
				depart_y, depart_x, depart_p,
				X_vars, Y_vars,
				X, Y,
				depart, T,
				DEPART_x,
				x, y,
				p,
				locd,
				dy,
				dx,
				dp);
		}
		ATTENDRE_CUDA();
	} else {
		d_nvidia_dot1d_blk_naive(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			x, y,
			p,
			locd,
			dy,
			dx,
			dp);
	}
}