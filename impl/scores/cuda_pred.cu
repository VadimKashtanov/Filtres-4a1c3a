#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static __global__ void kerd_nvidia_prediction_somme(
	float * y, uint depart, uint T,
	float * pred, float * _PRIXS,
	uint canal_p)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < T) {
		float p1 = _PRIXS[depart+thx+canal_p+1];
		float p0 = _PRIXS[depart+thx/*+canal_p*/];
		atomicAdd(
			pred,
			1.0*(uint)(cuda_signe((y[(0+thx)*P+canal_p])) == cuda_signe((p1/p0-1)))
		);
	};
};

static float __nvidia_prediction(float * y, uint depart, uint T, uint canal_p) {
	float * pred__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(pred__d, 0, 1*sizeof(float)));
	kerd_nvidia_prediction_somme<<<dim3(KERD(T,1024)),dim3(1024)>>>(
		y, depart, T,
		pred__d, prixs__d,
		canal_p
	);
	ATTENDRE_CUDA();
	float _pred;
	CONTROLE_CUDA(cudaMemcpy(&_pred, pred__d, sizeof(float)*1, cudaMemcpyDeviceToHost));
	cudafree<float>(pred__d);
	return _pred / (float)T;
};

float * nvidia_prediction(float * y, uint depart, uint T) {
	float * pred = (float*)malloc(sizeof(float) * P);
	FOR(0, p, P) pred[p] = __nvidia_prediction(y, depart, T, p);
	return pred;
};