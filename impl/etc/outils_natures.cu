#include "marchee.cuh"

void _outil_ema(float * y, float * x, uint K) {
	float _K = 1 / ((float)K);
	y[0] = x[0];
	FOR(1, t, PRIXS) {
		y[t] = y[t-1]*(1 - _K) + x[t] * _K;
	}
};

void _outil_macd(float * y, float * x, float coef) {
	ASSERT(coef > 0.0);
	float ema12[PRIXS], ema26[PRIXS], ema9[PRIXS], __macd[PRIXS];
	_outil_ema(ema12, x, 12*coef);
	_outil_ema(ema26, x, 26*coef);
	FOR(0, i, PRIXS) __macd[i] = ema12[i] - ema26[i];
	_outil_ema(ema9, __macd, 12*coef);
	FOR(0, i, PRIXS) y[i] = __macd[i] - ema9[i];
};

void _outil_chiffre(float * y, float * x, float chiffre) {
	FOR(0, t, PRIXS) {
		//y[t] = 2*(chiffre-MIN2(fabs(x[t]-chiffre*roundf((x[t]+0)/chiffre)), fabs(x[t]-chiffre*roundf((x[t]+chiffre)/chiffre))))/chiffre-1;
		float haut =  ceil(x[t]/chiffre)*chiffre;
		float bas  = floor(x[t]/chiffre)*chiffre;
		if (haut == bas) haut += chiffre;
		y[t] = 2*((x[t]-bas)/(haut-bas)-0.5);
	}
};

void _outil_awesome(float * y, float * x, float coef) {
	ASSERT(coef > 0.0);
	float ema5[PRIXS], ema30[PRIXS];
	_outil_ema( ema5, x,  5*coef);
	_outil_ema(ema30, x, 34*coef);
	FOR(0, i, PRIXS) y[i] = ema5[i] - ema30[i];
};

void _outil_pourcent_r(float * y, float * x, uint interv, uint ema_post_r) {
	ASSERT(interv > 0);
	uint n = 14*interv;
	FOR(0, i, n) y[i] = 0.0;
	FOR(n, i, PRIXS) {
		float max=x[i], min=x[i];
		FOR(0, j, n) {
			if (x[i-j] > max) max = x[i-j];
			if (x[i-j] < min) min = x[i-j];
		}
		y[i] = (max - x[i])/(max - min)  * (-1);	//%r a pour nature d'etre negatif
	};
	//
	//	Petit Ema post %R, pour lisser (ou pas) la courbe chaotique
	float _K = 1 / ((float)ema_post_r);
	FOR(1, t, PRIXS) {
		y[t] = y[t-1]*(1 - _K) + y[t] * _K;
	}
};

void _outil_rsi(float * y, float * x, uint interv) {
	ASSERT(interv > 0);
	uint n = 14*interv;
	FOR(0, i, n+1) y[i] = 0.0;
	//
	float changements[PRIXS];
	FOR(1, i, PRIXS) changements[i] = x[i] - x[i-1];
	//
	//#pragma omp parallel
	//#pragma omp for
	FOR(n+1, t, PRIXS) {
		float  gain_moy = 0;
		float perte_moy = 0;
		FOR(0, i, n) {
			if (changements[t-i] >= 0) gain_moy  += +(changements[t-i]);
			if (changements[t-i] <  0) perte_moy += -(changements[t-i]);
		}
		if (perte_moy != 0) {
			float rs = (gain_moy/n) / (perte_moy/n);
			y[t] = 1.0 - 1.0/(1+rs);
		} else {
			y[t] = 1.0;
		}
	}
};