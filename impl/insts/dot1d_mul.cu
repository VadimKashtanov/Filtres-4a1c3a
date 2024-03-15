#include "dot1d_mul.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define POLYNOMES 3

//y = x*logistic(ax+b)

void cree_dot1d_mul(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	mdl->inst_POIDS        [c] = (mdl->Y[c-1]+1)*mdl->Y[c] * POLYNOMES;
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);
	//
	FOR(0, pol, POLYNOMES) {
		FOR(0, y, Y) {
			FOR(0, x, X+1) {
				mdl->p[c][pol*((X+1)*Y) + y*(X+1)+x] = (2*rnd()-1) * sqrtf(/*10.0*/ 10.0 / (X+Y));
			}
		}
	}
};

void plume_dot1d_mul(Mdl_t * mdl, uint c)
{
	printf("POIDS DOT1D_MUL (tanh(ax+b)*logistique(cx+d)+tanh(ex+f)): \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	printf("tanh(ax+b)\n");
	FOR(0, y, Y) {
		printf("y=%i : ", y);
		FOR(0, x, X) {
			printf("%+f,", mdl->p[c][0*(X+1)*Y + y*(X+1)+x]);
		}
		printf(" biais=%+f\n", mdl->p[c][0*(X+1)*Y + y*(X+1)+X+1-1]);
	}
	printf("logistique(cx+d)\n");
	FOR(0, y, Y) {
		printf("y=%i : ", y);
		FOR(0, x, X) {
			printf("%+f,", mdl->p[c][1*(X+1)*Y + y*(X+1)+x]);
		}
		printf(" biais=%+f\n", mdl->p[c][1*(X+1)*Y + y*(X+1)+X+1-1]);
	}
	printf("tanh(ex+f)\n");
	FOR(0, y, Y) {
		printf("y=%i : ", y);
		FOR(0, x, X) {
			printf("%+f,", mdl->p[c][2*(X+1)*Y + y*(X+1)+x]);
		}
		printf(" biais=%+f\n", mdl->p[c][2*(X+1)*Y + y*(X+1)+X+1-1]);
	}
};

void intel_dot1d_mul(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	float s;
	FOR(0, t, T) {
		FOR(0, _y, Y) {
			//	tanh(ax+b)
			s = p[0*(X+1)*Y + _y*(X+1)+(X+1-1)];
			FOR(0, k, X) {
				float __x = x[(depart+t)*X_vars+DEPART_x+k];
				float __p = p[0*(X+1)*Y + _y*(X+1)+k];
				s += __x * __p;
			}
			float P0 = mul_TANH(s);
			locd[(depart+t)*(Y*3)+_y+0] = P0;
		
			//	logistique(cx+d)
			s = p[1*(X+1)*Y + _y*(X+1)+(X+1-1)];
			FOR(0, k, X) {
				float __x = x[(depart+t)*X_vars+DEPART_x+k];
				float __p = p[1*(X+1)*Y + _y*(X+1)+k];
				s += __x * __p;
			}
			float P1 = mul_LOGISTIQUE(s);
			locd[(depart+t)*(Y*3)+_y+1] = P1;
		
			//	tanh(ex+f)
			s = p[2*(X+1)*Y + _y*(X+1)+(X+1-1)];
			FOR(0, k, X) {
				float __x = x[(depart+t)*X_vars+DEPART_x+k];
				float __p = p[2*(X+1)*Y + _y*(X+1)+k];
				s += __x * __p;
			}
			float P2 = mul_TANH(s);
			locd[(depart+t)*(Y*3)+_y+2] = P2;

			//	P0*P1+P2
			y[(depart+t)*Y+_y] = P0 * P1 + P2;
		}
	}
}

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
	float * dp)
{
	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T
	FOR(0, t, T) {
		FOR(0, _x, X) {
			float s = 0;
			FOR(0, k, Y) {
				float __x = p[k*(X+1)+_x];
				float __p = locd[(depart+t)*Y+k] * dy[(depart+t)*Y+k];
				s += __x * __p;
			}
			dx[(depart+t)*X_vars+DEPART_x+_x]   = s;
		}
	}

	//dp = x.T @ ((y-_y)*dtanh(x@p))
	FOR(0, _y, Y) {
		float dbiais = 0;
		FOR(0, _x, X) {
			float s = 0;
			FOR(0, t, T) {
				float __x = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
				float __p = x[(depart+t)*X_vars+DEPART_x+_x];
				s += __x * __p;
				if (_x == 0) {	//	Biais
					dbiais += __x;
				}
			}
			dp[_y*(X+1)+_x] = s;
		}
		dp[_y*(X+1)+(X+1-1)] = dbiais;
	}
};

//	=========================================================

void f_dot1d_mul(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	ERR("Pas ajouté le depart_plus_t")
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		intel_dot1d_mul(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst]);
	} else if (mode == 3) {
		nvidia_dot1d_mul_shared_2_16(
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

void df_dot1d_mul(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	ERR("Pas ajouté le depart_plus_t")
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = 0;//t0;
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		d_intel_dot1d_mul(
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
	} else if (mode == 3) {
		d_nvidia_dot1d_mul_shared_2_16(
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