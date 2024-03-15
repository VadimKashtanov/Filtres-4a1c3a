#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define PSEUDO_ALEA_flt(x) (float)((1234*x + 4563) % 1000)/1000.0

static __global__ void kerd_cree_masque(
	uint graine,
	uint * masque, uint POIDS, float p)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < POIDS) {
		masque[thx] = (PSEUDO_ALEA_flt(graine+thx) >= p ? NON_MASQUEE : MASQUEE);
		//	0 - pas masque
		//	1 - masque

		//	masque = ne sera pas optimise
	}
};

static uint ** __cree_masque(
	Mdl_t * mdl, float * p)
{
	uint ** masque = alloc<uint*>(C);
	masque[0] = cudalloc<uint>(mdl->Y[0]);
	kerd_cree_masque<<<dim3(KERD(mdl->Y[0],1024)), dim3(1024)>>>(
		rand() % 10000,
		masque[0], mdl->Y[0], p[0]
	);
	ATTENDRE_CUDA();
	//
	FOR (1, c, C) {
		uint POIDS = mdl->inst_POIDS[c];
		masque[c] = cudalloc<uint>(POIDS);
		kerd_cree_masque<<<dim3(KERD(POIDS,1024)), dim3(1024)>>>(
			rand() % 10000,
			masque[c], POIDS, p[c]
		);
		ATTENDRE_CUDA();
	};
	return masque;
};

static void __liberer_masque(uint ** masque) {
	FOR(0, c, C) cudafree<uint>(masque[c]);
	free(masque);
}

// ---------------------------------------------------------

static __global__ void cree_poids_masquees__kerd(
	float * a_masquer, float * origine, uint * masque, uint POIDS)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS)
	{
		if (masque[thx] == MASQUEE) {
			a_masquer[thx] = 0;
		} else {
			a_masquer[thx] = origine[thx];
		}
	};
}

static float ** cree_poids_masques(
	Mdl_t * mdl, uint ** masque)
{
	float ** poids_masques = alloc<float*>(C);
	FOR(1, c, C) {
		uint POIDS = mdl->inst_POIDS[c];
		poids_masques[c] = cudalloc<float>(POIDS);
		cree_poids_masquees__kerd<<<dim3(KERD(POIDS,1024)),dim3(1024)>>>(
			poids_masques[c], mdl->p__d[c], masque[c], POIDS
		);
		ATTENDRE_CUDA();
	}
	return poids_masques;
};

static __global__ void coller_les_poids_masques_kerd(
	float * poids_masques, uint * masque,
	float * anciens_non_masques, uint POIDS)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS)
	{
		if (masque[thx] == MASQUEE) {
			//	poids_masques[thx] normalement = 0
			poids_masques[thx] = anciens_non_masques[thx];
		}
	};
};

static void coller_les_poids_masques(
	Mdl_t * mdl, uint ** masque, float ** non_masques)
{
	FOR(1, c, C) {
		uint POIDS = mdl->inst_POIDS[c];
		coller_les_poids_masques_kerd<<<dim3(KERD(POIDS,1024)),dim3(1024)>>>(
			mdl->p__d[c], masque[c], non_masques[c], POIDS
		);
		ATTENDRE_CUDA();
	}
};

static void liberer_poids_non_masques(float ** poids_non_masques) {
	FOR(1, c, C) CONTROLE_CUDA(cudaFree(poids_non_masques[c]));
	free(poids_non_masques);
};

//	==============================================================================
//	==============================================================================
//	==============================================================================

Masque_t * cree_masque(Mdl_t * mdl, float * p) {
	Masque_t * ret = alloc<Masque_t>(1);
	//
	ret->masque = __cree_masque(mdl, p);
	ret->poids_non_masques = alloc<float*>(C);
	FOR(1, c, C) ret->poids_non_masques[c] = mdl->p__d[c];
	ret->poids_masques = cree_poids_masques(mdl, ret->masque);
	
	//	Temporairement remplacer les poids par les nouveaux masques
	FOR(1, c, C) {
		mdl->p__d[c] = ret->poids_masques[c];
	}
	free(ret->poids_masques);	//car mdl a [C]

	return ret;
};

void sortire_masque(Mdl_t * mdl, Masque_t * masque) {
	//	Coller les poids masques depuis la version non masque (car ils sont =0 dans la partie masquee)
	coller_les_poids_masques(mdl, masque->masque, masque->poids_non_masques);

	//	Finalement remplacer l'ancien non masque
	//	par le ne nouveau masque (mis a jour avec toutes les vrais valeurs)

	// rien a faire c'est deja fait

	liberer_poids_non_masques(masque->poids_non_masques);
	__liberer_masque(masque->masque);
	free(masque);
};