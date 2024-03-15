#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static __global__ void kerd_opti_simple(
	float * p, float * dp, float alpha, uint POIDS, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		p[thx] -= alpha * dp[thx] / div;
	}
};

static __global__ void kerd_opti_simple_masque(
	float * p, float * dp, float alpha, uint POIDS, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		if (masque[thx] == NON_MASQUEE)
			p[thx] -= alpha * dp[thx] / div;
	}
};

void opti_simple(uint zero_accumulation_tous_les[C], uint optimiser_la_couche[C], Mdl_t * mdl, float * alpha, float div, uint ** masque) {
#define coef_div zero_accumulation_tous_les
	//	Filtres
	if (optimiser_la_couche[0] == 1) {
		uint FILTRES = mdl->Y[0];	//pas de *N, car c'est le filtre qu'on ignore, pas les points
		if (masque == 0) {
			kerd_opti_simple<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
				mdl->p__d[0], mdl->dp__d[0], alpha[0], FILTRES, div * coef_div[0]);
		} else {
			kerd_opti_simple_masque<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
				mdl->p__d[0], mdl->dp__d[0], alpha[0], FILTRES, div * coef_div[0], masque[0]
			);
		}
	}
	//	Poids
	FOR(1, c, C) {
		if (optimiser_la_couche[c] == 0) continue;

		uint POIDS = mdl->inst_POIDS[c];
		if (masque == 0) {
			kerd_opti_simple<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], alpha[c], POIDS, div * coef_div[c]
			);
		} else {
			kerd_opti_simple_masque<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], alpha[c], POIDS, div * coef_div[c], masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
	mdl_poids_gpu_vers_cpu(mdl);
};