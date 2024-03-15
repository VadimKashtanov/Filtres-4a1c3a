#pragma once
#include "insts.cuh"

#define     dot1d_blk_TANH 0

#define dot1d_blk_ACTIVATION dot1d_blk_TANH

#define dot1d_blk_tanh_f(s)          (tanh(s))
#define dot1d_blk_tanh_df(s,a)       (1 - a*a)

#define dot1d_blk_ACTIV(mode, s) (\
	(mode == dot1d_blk_TANH ? dot1d_blk_tanh_f(s) \
	: 0 ))
#define dot1d_blk_dACTIV(mode, s,a) (\
	(mode == dot1d_blk_TANH ? dot1d_blk_tanh_df(s,a) \
	: 0 ))

#include "mdl.cuh"

#define DOT1D_BLK_BLOQUES 8//16//8

void cree_dot1d_blk(Mdl_t * mdl, uint inst);
void plume_dot1d_blk(Mdl_t * mdl, uint c);

//	============================================

void intel_dot1d_blk(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_blk_naive(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_blk_shared(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_blk_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d_blk(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);

//	============================================

void d_intel_dot1d_blk(
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
	float * dp);

void d_nvidia_dot1d_blk_shared(
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
	float * dp);

void df_dot1d_blk(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);