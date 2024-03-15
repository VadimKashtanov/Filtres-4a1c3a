#include "dot1d_blk.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_dot1d_blk(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	//
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	mdl->inst_POIDS        [c] = P_blk*DOT1D_BLK_BLOQUES;
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	printf("Poids = %i\n", P_blk*DOT1D_BLK_BLOQUES);
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);

	/*			Distribution Uniforme et Normale	 	*/

	//	somme(r) = 1 && |max(r)|==|min(r)|==COEF
	/*FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
		//uint depart_y = dot1d_blk * Y_blk;
		//uint depart_x = dot1d_blk * X_blk;
		uint depart_p = dot1d_blk * P_blk;
		//
		FOR(0, y, Y_blk) {
			FOR(0, x, X_blk+1) {
				mdl->p[c][depart_p+y*(X_blk+1)+x] = (2*rnd()-1) * sqrtf(15.0 / X_blk);
			}
		}
	}*/
	FOR(0, i, mdl->inst_POIDS[c]) {
		mdl->p[c][i] = (2*rnd()-1) * sqrtf(/*10.0*/6.0 / (X_blk+Y_blk));
	}
};

void plume_dot1d_blk(Mdl_t * mdl, uint c)
{
	printf("POIDS dot1d_blk: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	//
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
		printf("[bloque=%i] ", dot1d_blk);
		//uint depart_y = dot1d_blk * Y_blk;
		//uint depart_x = dot1d_blk * X_blk;
		uint depart_p = dot1d_blk * P_blk;
		//
		FOR(0, y, Y_blk) {
			printf("y=%i : ", y);
			FOR(0, x, X_blk) {
				printf("%+f,", mdl->p[c][depart_p+y*(X_blk+1)+x]);
			}
			printf(" biais=%+f\n", mdl->p[c][depart_p+y*(X_blk+1)+X_blk+1-1]);
		}
	}
};

void intel_dot1d_blk(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
//#pragma omp parallel
//#pragma omp for
	FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
		uint depart_y = dot1d_blk * Y_blk;
		uint depart_x = dot1d_blk * X_blk;
		uint depart_p = dot1d_blk * P_blk;
		//
		/*#pragma omp parallel
		#pragma omp for*/
		FOR(0, t, T) {
			uint depart_plus_t = t_MODE_GENERALE(depart, DEPART, FIN, t);
			FOR(0, _y, Y_blk) {
				float s = p[depart_p+_y*(X_blk+1)+(X_blk+1-1)];
				FOR(0, k, X_blk) {
					float __x = x[/*(depart+t)*/depart_plus_t*X_vars+DEPART_x+depart_x+k];
					float __p = p[depart_p+_y*(X_blk+1)+k];
					s += __x * __p;
				}
				float a = dot1d_blk_ACTIV(dot1d_blk_ACTIVATION, s);
				y[/*(depart+t)*/depart_plus_t*Y+depart_y+_y]    = a;
				locd[/*(depart+t)*/depart_plus_t*Y+depart_y+_y] = dot1d_blk_dACTIV(dot1d_blk_ACTIVATION, s, a);
			}
		}
	}
}

void d_intel_dot1d_blk(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,	//<-- eventuallement pour un 
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	/*uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
		uint depart_y = dot1d_blk * Y_blk;
		uint depart_x = dot1d_blk * X_blk;
		uint depart_p = dot1d_blk * P_blk;
		//
		FOR(0, t, T) {
			FOR(0, _y, Y_blk) {
				float ds = locd[(depart+t)*Y+depart_y+_y] * dy[(depart+t)*Y+depart_y+_y];
				//
				dp[depart_p+_y*(X_blk+1)+(X_blk+1-1)] += ds;
				FOR(0, k, X_blk) {
					float __x = x[(depart+t)*X_vars+DEPART_x+depart_x+k];
					float __p = p[depart_p+_y*(X_blk+1)+k];
					//s += __x * __p;
					dx[(depart+t)*X_vars+DEPART_x+depart_x+k] += ds * __p; 
					dp[depart_p+_y*(X_blk+1)+k] += ds * __x;
				}
			}
		}
	}*/
	uint X_blk = X / DOT1D_BLK_BLOQUES;
	uint Y_blk = Y / DOT1D_BLK_BLOQUES;
	uint P_blk = (X_blk+1)*Y_blk;
	//
	FOR(0, dot1d_blk, DOT1D_BLK_BLOQUES) {
		uint depart_y = dot1d_blk * Y_blk;	//depat d'un bloque en x
		uint depart_x = dot1d_blk * X_blk;	// 					en y
		uint depart_p = dot1d_blk * P_blk;	//					en p
		//dx = (p @ ((y-_y)*dtanh(x@p)).T).T
	//#pragma omp parallel
	//#pragma omp for
		FOR(0, t, T) {
			uint depart_plus_t = t_MODE_GENERALE(depart, DEPART, FIN, t);
			FOR(0, _x, X_blk) {
				//float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
				float s = 0;
				FOR(0, k, Y_blk) {
					float __x = p[depart_p+k*(X_blk+1)+_x];//x[(depart+t)*X+k];
					float __p = locd[/*(depart+t)*/depart_plus_t*Y+depart_y+k] * dy[/*(depart+t)*/depart_plus_t*Y+depart_y+k];//p[_y*(X+1)+k];
					s += __x * __p;
				}
				dx[/*(depart+t)*/depart_plus_t*X_vars+DEPART_x+depart_x+_x]   = s;
			}
		}

		//dp = x.T @ ((y-_y)*dtanh(x@p))
	//#pragma omp parallel
	//#pragma omp for
		FOR(0, _y, Y_blk) {
			float dbiais = 0;
			FOR(0, _x, X_blk) {
				float s = 0;
				FOR(0, t, T) {
					uint depart_plus_t = t_MODE_GENERALE(depart, DEPART, FIN, t);
					float __x = locd[/*(depart+t)*/depart_plus_t*Y+depart_y+_y] * dy[/*(depart+t)*/depart_plus_t*Y+depart_y+_y];//x[(depart+t)*X+k];
					float __p = x[/*(depart+t)*/depart_plus_t*X_vars+DEPART_x+depart_x+_x];//p[_y*(X+1)+k];
					s += __x * __p;
					if (_x == 0) {	//	Biais
						dbiais += __x;
					}
				}
				dp[depart_p+_y*(X_blk+1)+_x] = s;
			}
			dp[depart_p+_y*(X_blk+1)+(X_blk+1-1)] = dbiais;
		}
	}
}

//	=========================================================

void f_dot1d_blk(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	ERR("Pas ajouté le depart_plus_t")
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		intel_dot1d_blk(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst]);
	} else if (mode == 1) {
		nvidia_dot1d_blk_naive(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 2) {
		nvidia_dot1d_blk_shared_2_16(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 3) {
		nvidia_dot1d_blk_shared_2_16(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void df_dot1d_blk(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	ERR("Pas ajouté le depart_plus_t")
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		d_intel_dot1d_blk(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst],
			mdl->dy[inst],
			mdl->dy[inst-1],
			mdl->dp[inst]);
	} else if (mode == 1) {
		d_nvidia_dot1d_blk_naive(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == 2) {
		d_nvidia_dot1d_blk_shared_2_16(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == 3) {
		d_nvidia_dot1d_blk_shared_2_16(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda df(x)", mode);
	}
}