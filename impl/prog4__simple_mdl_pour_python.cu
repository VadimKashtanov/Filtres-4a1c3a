#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static float filtre(uint inst0, uint depart, float * x, float * f, uint intervalle, uint type_norme, float _min, float _max) {
	float normer_x[N];
	//
	FOR(0, i, N) normer_x[i] = x[depart - (i)*intervalle];
	//
	if (type_norme == NORME_CLASSIQUE) {
		_min=normer_x[0];
		_max=normer_x[0];
		//
		FOR(1, i, N) {
			float a = normer_x[i];
			if (a > _max) _max = a;
			if (a < _min) _min = a;
		}
	} else if (type_norme == NORME_THEORIQUE) {
		// rien
	} else if (type_norme == NORME_RELATIVE) {
		_max=fabs(normer_x[0]);
		//
		FOR(1, i, N) {
			float a = fabs(normer_x[i]);
			if (a > _max) _max = a;
		}
		_max = +_max;
		_min = -_max;
	} else {
		ERR("type_norme == %i", type_norme);
	}
	//
	FOR(0, i, N) normer_x[i] = (normer_x[i]-_min)/(_max-_min);
	if (inst0 == 0) {
		//
		float s = 0, d = 0;
		float f_nouveau = f[0];
		float x_nouveau = normer_x[0];
		//
		float Ps = (0.5+0/N*0.5);
		s += powf(1 + fabs(x_nouveau - f_nouveau), Ps);
		//
		float f_avant = f_nouveau;
		float x_avant = x_nouveau;
		FOR(1, i, N) {
			f_nouveau = f[i];
			x_nouveau = normer_x[i];
			//
			float Ps = (0.5+i/N*0.5);
			float Pd = (1.0+i/N*1.0);
			//
			s += powf(1 + fabs(  x_nouveau   -   f_nouveau  ), Ps);
			d += powf(1 + fabs((x_nouveau-x_avant) - (f_nouveau-f_avant)), Pd);
			f_avant   = f_nouveau;
			x_avant   = x_nouveau;
		};

		s = s/8-1;
		d = d/7-1;

		return 2*expf(-s*s-d*d)-1;
	} else if (inst0 == 1) {
		//
		float s = 0;
		float f_nouveau = f[0];
		float x_nouveau = normer_x[0];
		//
		float Ps = 1.0;//(0.5+0/N*0.5);
		s += powf(1 + fabs(x_nouveau - f_nouveau), Ps);
		//
		float f_avant = f_nouveau;
		float x_avant = x_nouveau;
		FOR(1, i, N) {
			f_nouveau = f[i];
			x_nouveau = normer_x[i];
			//
			float Ps = 1.0;//(0.5+i/N*0.5);
			//
			s += powf(1 + fabs(  x_nouveau   -   f_nouveau  ), Ps);
			f_avant   = f_nouveau;
			x_avant   = x_nouveau;
		};

		s = s/8-1;

		return 2*expf(-s*s)-1;
	} else {
		ERR("Pas possible")
	}
	return 0;
};

static __global__ void simple_DOT1D(
	float * y_nouveau, float * y_avant, float * poids,
	uint X, uint Y, uint PRIXS_bitget, uint T)
{
	uint y = threadIdx.x + blockIdx.x + blockDim.x;
	uint t = DEPART + threadIdx.y + blockIdx.y + blockDim.y;
	if (y < Y && t < PRIXS_bitget) {
		float s = poids[(X+1)*y + X-1+1];
		FOR(0, j, X) s += poids[(X+1)*y + j] * y_avant[t*MAX_Y + j];
		y_nouveau[t*MAX_Y + y] = tanh(s);
	}
};

int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	//
	FILE * fp = fopen(argv[1], "rb");
	//
	uint Y[C];
	FREAD(Y, sizeof(uint), C, fp);
	uint insts[C];
	FREAD(insts, sizeof(uint), C, fp);
	uint cibles[C];
	FREAD(cibles, sizeof(uint), C, fp);
	uint depart_future[C];
	FREAD(depart_future, sizeof(uint), C, fp);
	//
	//
	//
	//
	uint PRIXS_bitget;
	FREAD(&PRIXS_bitget, sizeof(uint), 1, fp);
	uint intervalles[BLOQUES];
	FREAD(intervalles, sizeof(uint), BLOQUES, fp);
	//
	//
	//
	uint type_norme[BLOQUES];
	float _min[BLOQUES], _max[BLOQUES];
	FREAD(type_norme, sizeof(uint), BLOQUES, fp);
	FREAD(_min,       sizeof(float), BLOQUES, fp);
	FREAD(_max,       sizeof(float), BLOQUES, fp);
	//
	//
	//
	float * lignes = alloc<float>(PRIXS_bitget*BLOQUES);
	FREAD(lignes, sizeof(float), PRIXS_bitget*BLOQUES, fp);
	//
	float * poids[C];
	float * poids_cuda[C];
	FOR(0, c, C) {
		uint POIDS;
		FREAD(&POIDS, sizeof(uint), 1, fp);
		poids[c] = alloc<float>(POIDS);
		FREAD(poids[c], sizeof(float), POIDS, fp);
		//
		poids_cuda[c] = cpu_vers_gpu<float>(poids[c], POIDS);
	}
	//
	fclose(fp);

	//	------------- Calcule ----------------
	float * y_avant   = alloc<float>( PRIXS_bitget*MAX_Y );
	float * y_nouveau = alloc<float>( PRIXS_bitget*MAX_Y );
	//
	float * y_avant_cuda   = cudalloc<float>(PRIXS_bitget*MAX_Y);
	float * y_nouveau_cuda = cudalloc<float>(PRIXS_bitget*MAX_Y);
	//
	//#pragma omp parallel
	//#pragma omp for
	FOR(0, f, BLOQUES*F_PAR_BLOQUES) {
		uint b = (f - (f % F_PAR_BLOQUES)) / F_PAR_BLOQUES;
		FOR(DEPART, t, PRIXS_bitget) {
			y_nouveau[t*MAX_Y + f] = filtre(
				insts[0],
				//
				b*PRIXS_bitget + t,
				lignes,
				poids[0] + f*N,
				intervalles[b],
				type_norme[b],
				_min[b], _max[b]
			);
		}
	};
	FOR(0, i, PRIXS_bitget*MAX_Y) y_avant[i] = y_nouveau[i];
	//
	CONTROLE_CUDA(cudaMemcpy(y_avant_cuda, y_avant, PRIXS_bitget*MAX_Y*sizeof(float), cudaMemcpyHostToDevice));
	//
	FOR(1, c, C) {
		if (insts[c] == DOT1D) {
#include "dot1d.cuh"
			uint X = Y[c-1];
			//#pragma omp parallel
			//#pragma omp for
			FOR(0, i, Y[c]) {
				FOR(DEPART, t, PRIXS_bitget) {
					float s = poids[c][(X+1)*i + X-1+1];
					FOR(0, j, X) s += poids[c][(X+1)*i + j] * y_avant[t*MAX_Y + j];
					y_nouveau[t*MAX_Y + i] = ACTIV(ACTIVATION, s);//tanh(s);
				};
			};
			/*uint T = (PRIXS_bitget-DEPART);
			simple_DOT1D<<<dim3(KERD(Y[c], 16), KERD(T,16)), dim3(16,16)>>>(
				y_nouveau_cuda, y_avant_cuda, poids_cuda[c],
				X, Y[c], PRIXS_bitget, T);
			ATTENDRE_CUDA();*/
			//
		} else if (insts[c] == DOT1D_BLK) {
#include "dot1d_blk.cuh"
			ERR("a implementer");
			//
			uint  X = Y[c-1];
			uint _Y = Y[ c ];
			//
			uint X_blk =  X / DOT1D_BLK_BLOQUES;
			uint Y_blk = _Y / DOT1D_BLK_BLOQUES;
			uint P_blk =  ( X_blk + 1 ) * Y_blk;
			//
			FOR(DEPART, t, PRIXS_bitget) {
				FOR(0, blk, DOT1D_BLK_BLOQUES) {
					//
					uint depart_y = blk * Y_blk;
					uint depart_x = blk * X_blk;
					uint depart_p = blk * P_blk;
					//
					FOR(0, y, Y_blk) {
						float s = poids[c][depart_p + (X_blk+1)*y + (X_blk+1)+1];
						FOR(0, j, X_blk)
							s += poids[c][depart_p + (X_blk+1)*y] * y_avant[t*MAX_Y + depart_x + j];
						y_nouveau[t*MAX_Y + depart_y + y] = tanh(s);
					};
				};
			}
		} else {
			ERR("Inst = %i", insts[c]);
		}

		/*#pragma omp parallel
		#pragma omp for*/
		//#pragma omp parallel
		//#pragma omp for
		FOR(0, i, PRIXS_bitget*MAX_Y) y_avant[i] = y_nouveau[i];
		//
		CONTROLE_CUDA(cudaMemcpy(y_avant_cuda, y_nouveau_cuda, PRIXS_bitget*MAX_Y*sizeof(float), cudaMemcpyDeviceToDevice));
		//CONTROLE_CUDA(cudaMemcpy(y_avant_cuda, y_avant, PRIXS_bitget*MAX_Y*sizeof(float), cudaMemcpyHostToDevice));
	};

	//CONTROLE_CUDA(cudaMemcpy(y_nouveau, y_nouveau_cuda, PRIXS_bitget*MAX_Y*sizeof(float), cudaMemcpyDeviceToHost));

	//	---------- Ecrire Resultat ----------
	fp = fopen(argv[1], "wb");
	//
	float res[PRIXS_bitget];
	FOR(DEPART, t, PRIXS_bitget) res[t] = y_nouveau[t*MAX_Y + 0];
	FWRITE(res+DEPART, sizeof(float), (PRIXS_bitget-DEPART), fp);
	//
	fclose(fp);
}