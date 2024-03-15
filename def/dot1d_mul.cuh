#pragma once
#include "insts.cuh"

#define        mul_TANH(s)   (tanh(s)           )
#define       mul_dTANH(s,a) (1.-a*a            )
#define  mul_LOGISTIQUE(s)   (1. / (1.+expf(-s)))
#define mul_dLOGISTIQUE(s,a) (a*(1.-a)          )

#include "mdl.cuh"

#define PLUS 1.0

//	y = tanh(ax+b)*logistic(cx+d)+tanh(ex+f)

void cree_dot1d_mul(Mdl_t * mdl, uint inst);
void plume_dot1d_mul(Mdl_t * mdl, uint c);

//	============================================

void intel_dot1d_mul(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_mul_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d_mul(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);

//	============================================

void d_intel_dot1d_mul(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_mul_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void df_dot1d_mul(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);