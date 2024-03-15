#pragma once

#include "mdl.cuh"

#define SGD 0
#define RMSPROP 1
#define ADAM 2

//	======= SGD =============

void opti_simple(
	uint zero_accumulation_tous_les[C],
	uint optimiser_la_couche[C],
	Mdl_t * mdl,
	float * alpha, float div,
	uint ** masque);

//	======= RMSPROP ========

typedef struct {
	float * g[C];
} Rmsprop_t;

Rmsprop_t * cree_rmsprop(
	Mdl_t * mdl);

void opti_rmsprop(
	uint zero_accumulation_tous_les[C],
	uint optimiser_la_couche[C],
	Mdl_t * mdl, Rmsprop_t * rmsprop,
	float * alpha, float div,
	uint ** masque);

void liberer_rmsprop(Rmsprop_t * rmsprop);

//	======= RMSPROP ========

typedef struct {
	float * v[C];
	float * s[C];
} Adam_t;

Adam_t * cree_adam(
	Mdl_t * mdl);

void opti_adam(
	uint zero_accumulation_tous_les[C],
	uint optimiser_la_couche[C],
	Mdl_t * mdl, Adam_t * adam,
	float * alpha, float div,
	uint ** masque);

void liberer_adam(Adam_t * adam);

//	======================================
//	======= Optimisation Generale ========
//	======================================

typedef union {
	uint sgd;
	Rmsprop_t * rmsprop;
	Adam_t    *    adam;
} Opti_classe_t;

void __interne_optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	uint ** masque,
	uint PERTURBATIONS,
	uint zeroiser_tous_les[C]);

//	--- Version avec Masque ---

/*	Méthode :
1) Mdl a des poids
2) on copie et *0 les poids a masqué
3) temporairement on pointe mdl->p vers la copie avec les *0
4) on optimise
5) puis on remet les valeurs nulles par leurs anciennes valeurs
dans la version copiée (qui est dans mdl) depuis masque->poids_non_masqués
6) Finalement mdl a des poids optimisées (que ceux qui étaient pas masqué)
et donc on peut supprimer ces anciens poids qui sont maintenant
dans masque->poids_non_masqués

Conclusion : c'est indirect discret et pas tres propre
mais l'implementation est propre donc ca va temporairement
*/

typedef struct {
	uint ** masque;
	float ** poids_masques;
	float ** poids_non_masques;
} Masque_t;

#define NON_MASQUEE 0
#define MASQUEE 1

Masque_t * cree_masque(Mdl_t * mdl, float * p);

void sortire_masque(Mdl_t * mdl, Masque_t * masque);

//	--- Super optimisation ---

void optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque,
	uint PERTURBATIONS,
	uint zeroiser_tous_les[C]);

void optimisation_mini_packet(
	Mdl_t * mdl,
	uint t0, uint t1, uint _T_mini_paquet,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque,
	uint PERTURBATIONS,
	uint zeroiser_tous_les[C]);