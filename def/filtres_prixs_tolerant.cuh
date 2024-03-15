#pragma once
#include "insts.cuh"

#include "mdl.cuh"

void cree_filtres_prixs_tolerant(Mdl_t * mdl, uint inst);
void plume_filtres_prixs_tolerant(Mdl_t * mdl, uint c);

//	=====================================

void intel_filtres_prixs_tolerant___naive(				//	mode == 0
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void nvidia_filtres_prixs_tolerant___naive(			//	mode == 1
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void nvidia_filtres_prixs_tolerant___shared(			//	mode == 2
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd);

void f_filtres_prixs_tolerant(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);

//	----------------------------

void d_intel_filtres_prixs_tolerant___naive(				//	mode == 0
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void d_nvidia_filtres_prixs_tolerant___naive(		//	mode == 1
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void d_nvidia_filtres_prixs_tolerant___shared(		//	mode == 2
	uint _t_MODE, uint GRAINE,
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df);

void df_filtres_prixs_tolerant(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1, uint _t_MODE, uint GRAINE);