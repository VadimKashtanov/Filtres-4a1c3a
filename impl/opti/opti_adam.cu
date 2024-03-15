#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define adam_beta1 0.9
#define adam_beta2 0.99

static __global__ void kerd_opti_adam(
	float * p, float * dp, float * v, float * s,
	float alpha, uint POIDS, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		float _grad = dp[thx] / div;
		float _v = adam_beta1*v[thx] + (1-adam_beta1)*_grad;
		float _s = adam_beta2*s[thx] + (1-adam_beta2)*_grad*_grad;
		//
		float corr_v = _v / (1 - adam_beta1);
		float corr_s = _s / (1 - adam_beta2);
		//
		p[thx] -= alpha * _grad * corr_v / (sqrtf(corr_s) + 1e-8);
		v[thx] = _v;
		s[thx] = _s;
	}
};

static __global__ void kerd_opti_adam_masque(
	float * p, float * dp, float * v, float * s,
	float alpha, uint POIDS, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		if (masque[thx] == NON_MASQUEE) {
			float _grad = dp[thx] / div;
			float _v = adam_beta1*v[thx] + (1-adam_beta1)*_grad;
			float _s = adam_beta2*s[thx] + (1-adam_beta2)*_grad*_grad;
			//
			float corr_v = _v / (1 - adam_beta1);
			float corr_s = _s / (1 - adam_beta2);
			//
			p[thx] -= alpha * _grad * corr_v / (sqrtf(corr_s) + 1e-8);
			v[thx] = _v;
			s[thx] = _s;
		}
	}
};

Adam_t * cree_adam(
	Mdl_t * mdl)
{
	Adam_t * ret = alloc<Adam_t>(1);
	FOR(0, c, C) ret->v[c] = cudazero<float>(mdl->inst_POIDS[c]);
	FOR(0, c, C) ret->s[c] = cudazero<float>(mdl->inst_POIDS[c]);
	return ret;
};

void liberer_adam(Adam_t * adam) {
	FOR(0, c, C) cudafree<float>(adam->v[c]);
	FOR(0, c, C) cudafree<float>(adam->s[c]);
	free(adam);
};

void opti_adam(
	uint zero_accumulation_tous_les[C], uint optimiser_la_couche[C], Mdl_t * mdl, Adam_t * adam,
	float * alpha, float div, uint ** masque)
{
#define coef_div zero_accumulation_tous_les
	//	Filtres
	if (optimiser_la_couche[0] == 1) {
		uint FILTRES = mdl->inst_POIDS[0];	//pas de *N, car c'est le filtre qu'on ignore, pas les points
		kerd_opti_adam<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
			mdl->p__d[0], mdl->dp__d[0], adam->v[0], adam->s[0], alpha[0], FILTRES, div * coef_div[0]);
	}
	//	Poids
	FOR(1, c, C) {
		if (optimiser_la_couche[c] == 0) continue;
		
		uint POIDS = mdl->inst_POIDS[c];
		
		if (masque == 0) {
			kerd_opti_adam<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], adam->v[c], adam->s[c],
				alpha[c], POIDS, div * coef_div[c]
			);
		} else {
			kerd_opti_adam_masque<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], adam->v[c], adam->s[c],
				alpha[c], POIDS, div * coef_div[c], masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
	mdl_poids_gpu_vers_cpu(mdl);
};