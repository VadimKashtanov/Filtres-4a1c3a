#pragma once

#include "marchee.cuh"

#define SCORE_Y_COEF_BRUIT 0.00

#define P_S      2.00
#define P_somme  1.00
#define P_coef   0.50 //0.25, 0.50, 1.0

#define sng(x)	((x>=0) ? 1.0 : -1.0)

#define PUISS(diff,P) (powf(diff,P)/P)
#define dPUISS(diff,P) powf(diff,P-1)

#define S(y,w)  ((sng(y)==sng(w))?PUISS(y-w,P_S):PUISS(y-w,P_S*1))//(powf(y-w, P_S)/P_S)
#define dS(y,w) ((sng(y)==sng(w))?dPUISS(y-w,P_S):dPUISS(y-w,P_S*1))//(powf(y-w, P_S-1))
#define K(p1,p0,alea) (powf(fabs((100+10*alea/(SCORE_Y_COEF_BRUIT>0?SCORE_Y_COEF_BRUIT:1.0))*(p1/p0 - 1)),P_coef))

#define __SCORE(y,p1,p0,alea)  (K(p1,p0,alea) * S(y, sng(p1/p0 - 1)) )
#define __dSCORE(y,p1,p0,alea) (K(p1,p0,alea) * dS(y, sng(p1/p0 - 1)) )

//	----

static float SCORE(float y, float p1, float p0, float alea) {
	return __SCORE(y,p1,p0,alea);
};

static float APRES_SCORE(float somme) {
	return powf(somme, P_somme) / P_somme;
};

static float dAPRES_SCORE(float somme) {
	return powf(somme, P_somme - 1);
};

static float dSCORE(float y, float p1, float p0, float alea) {
	return __dSCORE(y,p1,p0,alea);
};

//	----

static __device__ float cuda_SCORE(float y, float p1, float p0, float alea) {
	return __SCORE(y,p1,p0,alea);
};

static __device__ float cuda_dSCORE(float y, float p1, float p0, float alea) {
	return __dSCORE(y,p1,p0,alea);
};

//	S(x) --- Score ---

float  intel_somme_score(float * y, uint depart, uint T);
float nvidia_somme_score(float * y, uint depart, uint T);

float  intel_score_finale(float somme, uint T);
float nvidia_score_finale(float somme, uint T);

//	dx

float d_intel_score_finale(float somme, uint T);
float d_nvidia_score_finale(float somme, uint T);

void  d_intel_somme_score(float d_somme, float * y, float * dy, uint depart, uint T);
void d_nvidia_somme_score(float d_somme, float * y, float * dy, uint depart, uint T);

//	%% Prediction

float* intel_prediction(float * y, uint depart, uint T);
float* nvidia_prediction(float * y, uint depart, uint T);