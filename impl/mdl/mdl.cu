#include "mdl.cuh"

#include "filtres_prixs.cuh"
#include "filtres_prixs_tolerant.cuh"
#include "dot1d.cuh"
#include "dot1d_blk.cuh"
#include "dot1d_mul.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

mdl_inst_f cree_inst[INSTS] = {
	cree_filtres_prixs,
	cree_filtres_prixs_tolerant,
	cree_dot1d,
	cree_dot1d_blk,
	cree_dot1d_mul
};

mdl_f_f inst_f [INSTS] = {
	f_filtres_prixs,
	f_filtres_prixs_tolerant,
	f_dot1d,
	f_dot1d_blk,
	f_dot1d_mul
};

mdl_f_f inst_df[INSTS] = {
	df_filtres_prixs,
	df_filtres_prixs_tolerant,
	df_dot1d,
	df_dot1d_blk,
	df_dot1d_mul
};

char * nom_inst[INSTS] = {
	"filtres_prixs         ",
	"filtres_prixs tolerant",
	"dot1d                 ",
	"dot1d bloques         ",
	"dot1d mul             "
};

mdl_inst_f plume_inst[INSTS] = {
	plume_filtres_prixs,
	plume_filtres_prixs_tolerant,
	plume_dot1d,
	plume_dot1d_blk,
	plume_dot1d_mul
};

uint * UNIFORME_C(uint x) {
	uint * ret = alloc<uint>(C);
	FOR(0, i, C) ret[i] = x;
	return ret;
};

static void calculer_normalisee__et__dif_normalisee(Mdl_t * mdl) { 
	FOR(0, b, BLOQUES) {
		FOR(DEPART, t, FIN) {
			//	_max & _min pour ce filtre-8
			float _max, _min;
			
			if (mdl->type_de_norme[b] == NORME_CLASSIQUE) {
				_max = mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle];
				_min = mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle];
				FOR(1, i, N_FLTR) {
					float xi = mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle];
					if (_max < xi)
						_max = xi;
					if (_min > xi)
						_min = xi;
				}


			} else if (mdl->type_de_norme[b] == NORME_RELATIVE) {
				_max = fabs(mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle]);
				FOR(1, i, N_FLTR) {
					float xi = fabs(mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle]);
					if (_max < xi)
						_max = xi;
				}
				_max = +_max;
				_min = -_max;

			} else if (mdl->type_de_norme[b] == NORME_THEORIQUE) {
				_min = /*1.0;//*/mdl->min_theorique[b];
				_max = /*0.0;//*/mdl->max_theorique[b];
				FOR(0, i, N_FLTR) {
					float xi = mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle];
					if (!(_min <= xi && xi <= _max))
						printf("%f  %f %f\n", xi, _min, _max);
					ASSERT(_min <= xi && xi <= _max);
				}

			} else {
				ERR("Norme ni classique, ni theorique, mdl->type_de_norme[b] == %i", mdl->type_de_norme[b]);
			};

			ASSERT(_min != _max);

			//
			FOR(0, i, N_FLTR) {
				mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] = ( mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle] - _min)/( _max - _min );
				//if (mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i]==0) printf("%i %i %i\n", b,t,i);
			}
		};

		//#pragma omp parallel
		//#pragma omp for
		FOR(DEPART, t, FIN) {
			FOR(1, i, N_FLTR)
				mdl->dif_normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] = mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] - mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i-1];
			mdl->dif_normalisee[b*PRIXS*N_FLTR+t*N_FLTR+N_FLTR+0] = 0.f;
		}
	}

	mdl->normalisee__d     = cpu_vers_gpu<float>(mdl->normalisee,     BLOQUES * PRIXS * N_FLTR);
	mdl->dif_normalisee__d = cpu_vers_gpu<float>(mdl->dif_normalisee, BLOQUES * PRIXS * N_FLTR);
};

static uint * tout_zeroiser = UNIFORME_C(1);

Mdl_t * cree_mdl(
	uint T,
	uint Y[C], uint insts[C],
	uint cible[C], uint decale_future[C],
	ema_int_t * bloque[BLOQUES]
) {
	ASSERT(Y[C-1] == P);
	ASSERT(Y[ 0 ] == BLOQUES * F_PAR_BLOQUES);
	ASSERT(insts[C-1] == DOT1D);				//	Afin d'assurer un Y=inst_VARS
	
	Mdl_t * mdl = alloc<Mdl_t>(1);

	mdl->T = T;

	//
	FOR(0, i, BLOQUES) {
		mdl->bloque[i]  = bloque[i];
		mdl->intervalles[i] = bloque[i]->intervalle;
		//
		mdl->type_de_norme[i] = bloque[i]->type_de_norme;
		mdl->min_theorique[i] = bloque[i]->min_theorique;
		mdl->max_theorique[i] = bloque[i]->max_theorique;
	};

	mdl->intervalles__d = cpu_vers_gpu<uint>(mdl->intervalles, BLOQUES);

	//
	calculer_normalisee__et__dif_normalisee(mdl);
	//raise(SIGINT);

	//	Architecture
	memcpy(mdl->insts,                 insts, sizeof(uint) * C);
	memcpy(mdl->Y,                         Y, sizeof(uint) * C);
	memcpy(mdl->cible,                 cible, sizeof(uint) * C);
	memcpy(mdl->decale_future, decale_future, sizeof(uint) * C);

	//	Allocation
	mdl->total_POIDS = 0;
	FOR(0, c, C) {
		//if (c>0) ASSERT(insts[c] != 0);
		ASSERT(Y[c] <= MAX_Y);
		//
		ASSERT(cible[c] == CIBLE_NORMALE || cible[c] == CIBLE_FILTRES_FUTURES);
		if (cible[c] == CIBLE_FILTRES_FUTURES) {
			ASSERT(Y[c] == F_PAR_BLOQUES * BLOQUES);
		}
		ASSERT(decale_future[c] <= MAX_DECALE_FUTURE);
		//
		cree_inst[insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		mdl->y [c] = alloc<float>(mdl->inst_VARS [c] * T);
		mdl->l [c] = alloc<float>(mdl->inst_LOCDS[c] * T);
		mdl->dy[c] = alloc<float>(mdl->inst_VARS [c] * T);
		mdl->dp[c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		mdl->y__d [c] = cudalloc<float>(mdl->inst_VARS [c] * T);
		mdl->l__d [c] = cudalloc<float>(mdl->inst_LOCDS[c] * T);
		mdl->dy__d[c] = cudalloc<float>(mdl->inst_VARS [c] * T);
		mdl->dp__d[c] = cudalloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];

		//	-----
		if (cible[c] == CIBLE_FILTRES_FUTURES) {
			mdl->constantes   [c] = gpu_vers_cpu<float>(mdl->p__d[c], mdl->inst_POIDS[c]);
			mdl->constantes__d[c] = cpu_vers_gpu<float>(mdl->p   [c], mdl->inst_POIDS[c]);
		}
	}
	ASSERT(mdl->inst_DEPART_SORTIE[C-1] == 0);
	//
	mdl_normer_les_filtres(mdl);
	//
	//uint * tout_zeroiser = UNIFORME_C(1);
	//
	mdl_zero_deriv_gpu(mdl, tout_zeroiser);
	mdl_zero_deriv_cpu(mdl, tout_zeroiser);
	//
	return mdl;
};

void mdl_re_cree_poids(Mdl_t * mdl) {
	//	Allocation
	mdl->total_POIDS = 0;
	FOR(0, c, C) {
		if (c>0) ASSERT(mdl->insts[c] != 0);
		ASSERT(mdl->Y[c] <= MAX_Y);
		//
		free(mdl->p[c]);
		CONTROLE_CUDA(cudaFree(mdl->p__d[c]));
		//
		cree_inst[mdl->insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];
	}
};

void mdl_changer_couche_Y(Mdl_t * mdl, uint c, uint nouveau_Y) {
	mdl->total_POIDS -= mdl->inst_POIDS[c];
	{
		if (c>0) ASSERT(mdl->insts[c] != 0);
		mdl->Y[c] = nouveau_Y;
		ASSERT(mdl->Y[c] <= MAX_Y);
		//
		free(mdl->p[c]);
		CONTROLE_CUDA(cudaFree(mdl->p__d[c]));
		//
		cree_inst[mdl->insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];
	}
};

void mdl_normer_les_filtres(Mdl_t * mdl) {
	FOR(0, b, BLOQUES) {
		FOR(0, f, F_PAR_BLOQUES) {
			float min, max;
			uint type_norme = mdl->type_de_norme[b];
			if (type_norme == NORME_CLASSIQUE) {
				max=mdl->p[0][b*F_PAR_BLOQUES*N + f*N+0];
				min=mdl->p[0][b*F_PAR_BLOQUES*N + f*N+0];
				FOR(1, i, N) {
					if (max < mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]) max = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
					if (min > mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]) min = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
				}
			} else if (type_norme == NORME_THEORIQUE || type_norme == NORME_RELATIVE) {
				min = 0.0;	//se sont des filtres
				max = 1.0;	//pas les natures

				//	Borne au cas ou
				FOR(0, i, N) {
					float f_val = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
					mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i] = MAX2(MIN2(f_val, max), min);
				}
			} else {
				ERR("mdl->type_de_norme[b]=%i", mdl->type_de_norme[b]);
			}
			//
			FOR(0, i, N) mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i] = (mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]-min)/(max-min);
		};
	}
	CONTROLE_CUDA(cudaMemcpy(mdl->p__d[0], mdl->p[0], sizeof(float)*BLOQUES*F_PAR_BLOQUES*N, cudaMemcpyHostToDevice))
};

PAS_OPTIMISER()
void mdl_verif(Mdl_t * mdl) {
	FOR(1, c, C) {
		float * r = gpu_vers_cpu<float>(mdl->p__d[c], mdl->inst_POIDS[c]);
		FOR(0, i, mdl->inst_POIDS[c]) ASSERT(fabs(r[i]-mdl->p[c][i]) < 0.01);
		free(r);
	}
};

PAS_OPTIMISER()
void mdl_poids_cpu_vers_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],  mdl->p[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void mdl_poids_gpu_vers_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p[c],  mdl->p__d[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
	}
};

PAS_OPTIMISER()
void mdl_gpu_vers_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p[c],  mdl->p__d[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->y[c],  mdl->y__d[c],  sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->l[c],  mdl->l__d[c],  sizeof(float)*mdl->inst_LOCDS[c]*mdl->T, cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy[c], mdl->dy__d[c], sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp[c], mdl->dp__d[c], sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
	}
}

PAS_OPTIMISER()
void mdl_cpu_vers_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],  mdl->p[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->y__d[c],  mdl->y[c],  sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->l__d[c],  mdl->l[c],  sizeof(float)*mdl->inst_LOCDS[c]*mdl->T, cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy__d[c], mdl->dy[c], sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp__d[c], mdl->dp[c], sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void liberer_mdl(Mdl_t * mdl) {
	FOR(0, c, C) {
		free(mdl->p [c]);
		free(mdl->y [c]);
		free(mdl->l [c]);
		free(mdl->dy[c]);
		free(mdl->dp[c]);
		//
		CONTROLE_CUDA(cudaFree(mdl->p__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->y__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->l__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->dy__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dp__d[c]));
	}
};

PAS_OPTIMISER()
void mdl_zero_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		memset(mdl->y [c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T);
	}
};

PAS_OPTIMISER()
void mdl_zero_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemset(mdl->y__d [c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T));
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_cpu(Mdl_t * mdl, uint zeroiser[C]) {
	FOR(0, c, C) {
		if (zeroiser[c]) {
			memset(mdl->dy[c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T);
			memset(mdl->dp[c], 0, sizeof(float) * mdl->inst_POIDS[c]);
		}
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_gpu(Mdl_t * mdl, uint zeroiser[C]) {
	FOR(0, c, C) {
		if (zeroiser[c]) {
			CONTROLE_CUDA(cudaMemset(mdl->dy__d[c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T));
			CONTROLE_CUDA(cudaMemset(mdl->dp__d[c], 0, sizeof(float) * mdl->inst_POIDS[c]));
		}
	}
};