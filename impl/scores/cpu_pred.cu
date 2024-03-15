#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float* intel_prediction(float * y, uint depart, uint T) {
	float * pourcent = (float*)calloc(P, sizeof(float));
	//
	FOR(0, i, T) {
		FOR(0, p, P) {
			if (signe(y[(0+i)*P+p]) == signe(prixs[depart+i+p+1]/prixs[depart+i/*+p*/]-1)) {
				pourcent[p] += 1.0;
			}
		}
	}
	//
	FOR(0, p, P)
		pourcent[p] /= (float)T;
	//
	return pourcent;
};