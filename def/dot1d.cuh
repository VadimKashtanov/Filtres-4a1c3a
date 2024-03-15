#pragma once

#include "insts.cuh"

#define TANH 0
#define LOGISTIC 1
#define BINAIRE 2

#define ACTIVATION /*BINAIRE*/TANH

#define tanh_f(s)          (tanh(s))
#define tanh_df(s,a)       (1 - a*a)

#define logistique_f(s)    (1.f/(1.f + expf(-s)))
#define logistique_df(s,a) (a * (a - 1.f))

#define   COEF 0.80//0.30
#define N_COEF (1-COEF)

#define  binaire_f(s)     (( ((s >= -0.5 ? 0.5 : 0.0) + (s >= +0.5 ? 0.5 : 0.0))*N_COEF + COEF*tanh(s)))
#define binaire_df(s,a)   ((0.0*N_COEF + COEF*(1 - a*a)))

#define ACTIV(mode, s) (\
	(mode == TANH ? tanh_f(s) \
	: (mode == LOGISTIC ? logistique_f(s) \
	: (mode == BINAIRE ? binaire_f(s) \
	: 0 ))))
#define dACTIV(mode, s,a) (\
	(mode == TANH ? tanh_df(s,a) \
	: (mode == LOGISTIC ? logistique_df(s,a) \
	: (mode == BINAIRE ? binaire_df(s,a) \
	: 0 ))))

#include "mdl.cuh"

void cree_dot1d(Mdl_t * mdl, uint inst);
void plume_dot1d(Mdl_t * mdl, uint c);

//	============================================

void intel_dot1d(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_naive(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared_2_16(
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE,);

//	============================================

void d_intel_dot1d(
	uint _t_MODE, uint GRAINE,
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

void d_nvidia_dot1d_naive(
	uint _t_MODE, uint GRAINE,
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

void d_nvidia_dot1d_shared(
	uint _t_MODE, uint GRAINE,
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

void d_nvidia_dot1d_shared_2_16(
	uint _t_MODE, uint GRAINE,
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

void df_dot1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE,);