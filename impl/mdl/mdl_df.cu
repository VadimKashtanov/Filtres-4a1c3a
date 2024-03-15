#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//	======================= Cible Decale Future =======================

static float filtre(float * x, float * dif_x, float * f) {
	float s = 0, d = 0;
	float f_nouveau = f[0];
	s += sqrtf(1 + fabs(x[0] - f_nouveau));
	float f_avant   = f_nouveau;
	FOR(1, i, N) {
		f_nouveau = f[i];
		s += sqrtf(1 + fabs(  x[i]   -   f_nouveau  ));
		d += powf((1 + fabs(dif_x[i] - (f_nouveau-f_avant))), 2);
		f_avant   = f_nouveau;
	};

	s = s/8-1;
	d = d/7-1;

	float y = expf(-s*s -d*d);

	return 2*y-1;
};

void intel_score_cible_filtres_future(
	uint DECALE_FUTURE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * dy)
{
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				float flt = filtre(
						x + b*PRIXS*N_FLTR + (depart+t+DECALE_FUTURE)*N_FLTR,
					dif_x + b*PRIXS*N_FLTR + (depart+t+DECALE_FUTURE)*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N
				);

				//	On ecrit par dessus le score
				dy[(0+t)*bloques*f_par_bloque + b*f_par_bloque + _f] = \
					(y[(0+t)*bloques*f_par_bloque + b*f_par_bloque + _f] - flt)/10000;
			}
		}
	}
};

#define BLOQUE_T  16

#define _repete_T 16

static __global__ void kerd_score_cible_filtres_futureshared(
	uint DECALE_FUTURE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * dy)
{
	uint depart_bloque_t = blockIdx.y * BLOQUE_T * _repete_T;
	uint depart_thread_t = depart_bloque_t + threadIdx.y * _repete_T;

	uint _b = blockIdx.x;
	uint _f = blockIdx.z;	//(ligne dans bloque)

	uint LIGNE  = _b;
	uint BLOQUE = _b; 

	uint thx = threadIdx.x;
	uint thy__t = threadIdx.y;

	//if (_t < T)
	__shared__ float __f__[N];
	//
	if (thy__t==0) __f__[thx]  = f[BLOQUE*F_PAR_BLOQUES*N + _f*N + thx];
	__syncthreads();
	//
	float fi, fi1;
	fi = __f__[thx];
	if (thx != 0)
		fi1 = __f__[thx-1];
	//
	__shared__ float __ret[BLOQUE_T][2];	//s, d
	__shared__ float __y  [BLOQUE_T];
	//
	float xi, dif_xi;
	//
	uint _t;
	FOR(0, plus_t, _repete_T) {
		_t = depart_thread_t + plus_t;
		//
		if (thx < 2) {
			__ret[thy__t][thx] = 0;
		}
		__syncthreads();
		//
		xi = x[LIGNE*PRIXS*N_FLTR + (depart+_t+DECALE_FUTURE)*N_FLTR + thx];
		//
		if (thx != 0) {
			dif_xi = dif_x[LIGNE*PRIXS*N_FLTR + (depart+_t+DECALE_FUTURE)*N_FLTR + thx];
			atomicAdd(&__ret[thy__t][1], powf((1 + fabs(dif_xi - (fi-fi1))), 2));
		}
		atomicAdd(&__ret[thy__t][0], sqrtf(1 + fabs(xi - fi)));
		__syncthreads();
		//
		if (thx < 2) {
			__ret[thy__t][thx] = __ret[thy__t][thx]/(float)(8-thx) - 1.0;
		}
		__syncthreads();
		//
		if (thx < 1) {
			__y[thy__t] = expf(-__ret[thy__t][0]*__ret[thy__t][0] -__ret[thy__t][1]*__ret[thy__t][1]);
		}
		__syncthreads();
		//
		/*if (thx < 2) {
			locd[(0+_t)*BLOQUES*(F_PAR_BLOQUES*2) + BLOQUE*(F_PAR_BLOQUES*2) + _f*2 + thx] = -2*2*__ret[thy__t][thx]*__y[thy__t];
		}
		__syncthreads();*/
		//
		if (thx < 1) {
			float flt = 2*__y[thy__t] - 1;
			float d_score = y[(0+_t)*BLOQUES*F_PAR_BLOQUES + BLOQUE*F_PAR_BLOQUES + _f] - flt;
			dy[(0+_t)*BLOQUES*F_PAR_BLOQUES + BLOQUE*F_PAR_BLOQUES + _f] = d_score / 10000;
		}
	};
};

void d_nvidia_score_cible_filtres_futureshared(
	uint DECALE_FUTURE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * dy)
{
	ASSERT(BLOQUE_T*_repete_T <= T);
	kerd_score_cible_filtres_futureshared<<<dim3(bloques, KERD((DIV(T,_repete_T)), BLOQUE_T), F_PAR_BLOQUES), dim3(N, BLOQUE_T,1)>>>(
		DECALE_FUTURE,
		X_vars, Y_vars,
		depart, T,
		bloques,
		x, dif_x,
		f,
		y,
		dy);
	ATTENDRE_CUDA();
};

//	===================================================================

void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	RETRO_FOR(0, c, C) {
		if        (mdl->cible[c] == CIBLE_NORMALE        ) {
			//	Rien a faire
		} else if (mdl->cible[c] == CIBLE_FILTRES_FUTURES) {
			uint depart = t0;
			uint X_vars=0, Y_vars=mdl->inst_VARS[c];
			uint T = (t1-t0);
			ASSERT(T == mdl->T);
			if (mode == 0) {
				intel_score_cible_filtres_future(
					mdl->decale_future[c],
					X_vars, Y_vars,
					depart, T,
					BLOQUES, F_PAR_BLOQUES,
					mdl->normalisee, mdl->dif_normalisee,
					mdl->constantes[c],//mdl->p[c],
					mdl->y[c],
					mdl->dy[c]);
			} else {
				d_nvidia_score_cible_filtres_futureshared(
					mdl->decale_future[c],
					X_vars, Y_vars,
					depart, T,
					BLOQUES, F_PAR_BLOQUES,
					mdl->normalisee__d, mdl->dif_normalisee__d,
					mdl->constantes__d[c],//mdl->p__d [c],
					mdl->y__d [c],
					mdl->dy__d[c]);
			};
		} else {
			ERR("Pas de cible %i", mdl->cible[c]);
		}
		//
		inst_df[mdl->insts[c]](mdl, c, mode, t0, t1);
	};
};