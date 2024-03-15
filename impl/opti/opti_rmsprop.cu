#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define rms_alpha 0.90

static __global__ void kerd_opti_rmsprop(
	float * p, float * dp, float * g,
	float alpha, uint POIDS, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		float _grad = dp[thx] / div;
		float _g = rms_alpha*g[thx] + (1-rms_alpha)*_grad*_grad;
		p[thx] -= alpha * _grad / (sqrtf(_g) + 1e-8);
		g[thx] = _g;
	}
};

static __global__ void kerd_opti_rmsprop_masque(
	float * p, float * dp, float * g,
	float alpha, uint POIDS, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		if (masque[thx] == NON_MASQUEE) {
			float _grad = dp[thx] / div;
			float _g = rms_alpha*g[thx] + (1-rms_alpha)*_grad*_grad;
			p[thx] -= alpha * _grad / (sqrtf(_g) + 1e-8);
			g[thx] = _g;
		}
	}
};

Rmsprop_t * cree_rmsprop(
	Mdl_t * mdl)
{
	Rmsprop_t * ret = alloc<Rmsprop_t>(1);
	FOR(0, c, C) ret->g[c] = cudazero<float>(mdl->inst_POIDS[c]);
	return ret;
};

void liberer_rmsprop(Rmsprop_t * rmsprop) {
	FOR(0, c, C) cudafree<float>(rmsprop->g[c]);
	free(rmsprop);
};

void opti_rmsprop(
	uint zero_accumulation_tous_les[C], uint optimiser_la_couche[C], Mdl_t * mdl, Rmsprop_t * rmsprop,
	float * alpha, float div, uint ** masque)
{
#define coef_div zero_accumulation_tous_les
	//	Filtres
	if (optimiser_la_couche[0] == 1) {
		uint FILTRES = mdl->inst_POIDS[0];	//pas de *N, car c'est le filtre qu'on ignore, pas les points
		kerd_opti_rmsprop<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
			mdl->p__d[0], mdl->dp__d[0], rmsprop->g[0], alpha[0], FILTRES, div * coef_div[0]);
	}
	//	Poids
	FOR(1, c, C) {
		if (optimiser_la_couche[c] == 0) continue;

		uint POIDS = mdl->inst_POIDS[c];
		
		if (masque == 0) {
			kerd_opti_rmsprop<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], rmsprop->g[c],
				alpha[c], POIDS, div * coef_div[c]
			);
		} else {
			kerd_opti_rmsprop_masque<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], rmsprop->g[c],
				alpha[c], POIDS, div * coef_div[c], masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
	mdl_poids_gpu_vers_cpu(mdl);
};