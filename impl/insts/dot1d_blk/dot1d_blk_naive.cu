#include "dot1d_blk.cuh"

#define BLOQUE_T 32
#define BLOQUE_Y 32

static __global__ void kerd_dot1d_blk_naive(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	if (_t < T && _y < Y) {
		uint dot1d_blk = (_y-_y%Y_blk)/Y_blk;
		//
		uint depart_y = dot1d_blk * Y_blk;
		uint depart_x = dot1d_blk * X_blk;
		uint depart_p = dot1d_blk * P_blk;
		//
		float s = p[depart_p+_y*(X_blk+1) + (X_blk+1-1)];
		FOR(0, i, X_blk) s += x[depart_x+(depart+_t)*X_vars + DEPART_x + i] * p[depart_p+_y*(X_blk+1) + i];
		float a = dot1d_blk_ACTIV(dot1d_blk_ACTIVATION, s);
		y[(depart+_t)*Y + depart_y+_y] = a;
		locd[(depart+_t)*Y + depart_y+_y] = dot1d_blk_dACTIV(dot1d_blk_ACTIVATION, s,a);
	}
};

void nvidia_dot1d_blk_naive(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	kerd_dot1d_blk_naive<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
		X_vars, Y_vars,
		X, Y,
		depart, T,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
}

//	============================= Derivation ==============================

static __global__ void kerd_deriv_dot1d_blk_naive(
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
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;
	uint _y = threadIdx.y + blockIdx.y * blockDim.y;
	//
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;

	if (_t < T && _y < Y) {
		uint dot1d_blk = (_y-_y%Y_blk)/Y_blk;
		//
		uint depart_y = dot1d_blk * Y_blk;
		uint depart_x = dot1d_blk * X_blk;
		uint depart_p = dot1d_blk * P_blk;
		//
		float _locd = locd[(depart+_t)*Y + depart_y+_y] * dy[(depart+_t)*Y + depart_y+_y];
		atomicAdd(&dp[depart_p+_y*(X_blk+1) + (X_blk+1-1)], _locd);
		FOR(0, i, X_blk) {
			atomicAdd(&dx[(depart+_t)*X_vars + DEPART_x +depart_x+ i], _locd * p[depart_p+_y*(X_blk+1) + i]);
			atomicAdd(&dp[depart_p+_y*(X_blk+1) + i], _locd * x[(depart+_t)*X_vars + DEPART_x +depart_x+ i]);
		}
	}
};

void d_nvidia_dot1d_blk_naive(
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
	kerd_deriv_dot1d_blk_naive<<<dim3(KERD(T, BLOQUE_T), KERD(Y, BLOQUE_Y)), dim3(BLOQUE_T, BLOQUE_Y)>>>(
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
	ATTENDRE_CUDA();
};