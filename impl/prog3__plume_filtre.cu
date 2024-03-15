#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

/*
Utilisation :
	./prog0__plume_filtre mdl.bin bloque f_dans_bloque
*/

#define GRAND_T 16*16*1

int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");  charger_tout();
	//
	if (argc == 4) {
		Mdl_t * mdl = ouvrire_mdl(GRAND_T, argv[1]);
		//
		char cmd[1000];
		//
		uint bloque = atoi(argv[2]);
		uint      f = atoi(argv[3]);
		//
		uint depart = bloque*F_PAR_BLOQUES*N + f*N;
		snprintf(cmd, 1000, "python3 -c \"import matplotlib.pyplot as plt;plt.plot([%f,%f,%f,%f,%f,%f,%f,%f]);plt.show()\"",
			mdl->p[0][depart + 0],
			mdl->p[0][depart + 1],
			mdl->p[0][depart + 2],
			mdl->p[0][depart + 3],
			mdl->p[0][depart + 4],
			mdl->p[0][depart + 5],
			mdl->p[0][depart + 6],
			mdl->p[0][depart + 7]
		);
		//
		printf("Type de norme = %s\n", nom_type_de_norme[mdl->bloque[bloque]->type_de_norme]);
		//
		SYSTEM(cmd);
		liberer_mdl(mdl);
		//
	} else {
		ERR("./prog0__plume_filtre mdl.bin bloque f_dans_bloque")
	}
	liberer_tout();
};