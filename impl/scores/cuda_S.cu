#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define pseudo_alea_d_une_grain(i) ((float)((121+(i%1234))*31 % 1001 ) / 1001.0)

//	===============================================================

static __global__ void kerd_nvidia_score_somme(
	float * y, uint depart, uint T,
	float * score, float * _PRIXS)
{
	uint t = threadIdx.x + blockIdx.x + blockDim.x;
	if (t < T) {
		float s = 0;
		FOR(0, p, P) {
			float _y = y[(0 + t)*P + p];
			float alea = 2*pseudo_alea_d_une_grain(t + ((uint)_y % 10001))-1;
			_y += alea * SCORE_Y_COEF_BRUIT;
			s += (P-p)*cuda_SCORE(
				_y, _PRIXS[depart+t+p+1], _PRIXS[depart+t], alea * SCORE_Y_COEF_BRUIT
			);
		}
		atomicAdd(score, s);
	}
};

float nvidia_somme_score(float * y, uint depart, uint T)
{
	float * somme_score__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(somme_score__d, 0, sizeof(float)*1));
	kerd_nvidia_score_somme<<<dim3(KERD(T,1)),dim3(1)>>>(
		y, depart, T,
		somme_score__d, prixs__d
	);
	ATTENDRE_CUDA();
	float somme_score;
	CONTROLE_CUDA(cudaMemcpy(&somme_score, somme_score__d, sizeof(float)*1, cudaMemcpyDeviceToHost));
	CONTROLE_CUDA(cudaFree(somme_score__d));
	return somme_score;
};

float  nvidia_score_finale(float somme, uint T) {
	return APRES_SCORE(somme / (float)(P * T));
};

//	===============================================================

float d_nvidia_score_finale(float somme, uint T) {
	return dAPRES_SCORE(somme / (float)(P * T)) / (float)(P * T);
};

//	===============================================================

static __global__ void kerd_nvidia_score_dpowf(
	float _dy, float * y, float * dy,
	uint depart, uint T,
	float * _PRIXS)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;

	if (_t < T) {
		FOR(0, p, P) {
			float _y = y[(0+_t)*P+p];
			float alea = 2*pseudo_alea_d_une_grain(_t + ((uint)_y % 10001))-1;
			dy[(0+_t)*P+p] = _dy * (P-p)*cuda_dSCORE(
				y[(0+_t)*P+p]+alea*SCORE_Y_COEF_BRUIT, _PRIXS[depart+_t+p+1], _PRIXS[depart+_t/*+p*/], alea*SCORE_Y_COEF_BRUIT
			);
		}
	}
};

void d_nvidia_somme_score(float d_score, float * y, float * dy, uint depart, uint T) {
	kerd_nvidia_score_dpowf<<<dim3(KERD(T,1024)), dim3(1024)>>>(
		d_score,
		y, dy,
		depart, T,
		prixs__d
	);
	ATTENDRE_CUDA();
};