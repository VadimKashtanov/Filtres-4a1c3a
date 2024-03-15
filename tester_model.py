#! /usr/bin/python3

from mdl import *

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: (x if abs(x) >= 0.0 else 0)

prixs = I__sources[0]

if __name__ == "__main__":
	mdl = Mdl("mdl.bin")

	print("Calcule ...")
	pred = mdl()
	print(pred)
	print("Fin Calcule")

	print("5 dernieres predictions : ", pred[-5:])

	_prixs = I__sources[0][DEPART:]

	print(len(pred), len(_prixs))

	plt.plot(e_norme(_prixs));
	plt.plot(pred, 'o');
	#
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)): plt.plot([i for _ in pred], e_norme(list(range(len(pred)))), '--')

	plt.show()

	##	================ Gain ===============
	LEVIER = 125
	METHODES  = ["LIBRE", "SIGNE", "REDUCTION PERTES"]
	POURCENTS = [0.03, 0.10, .15, 0.25, 0.35, 0.50]
	fig, axs = plt.subplots(len(POURCENTS), len(METHODES))
	for _i,POURCENT in enumerate(POURCENTS):
		for _j,METHODE in enumerate(METHODES):
			for version in (+1,-1):
				#
				u = 40
				usd = [u]
				#
				for i in range(DEPART, I_PRIXS-1):

					__pred = version*pred[i-DEPART]

					if METHODE == "LIBRE":
						u += u * LEVIER * POURCENT * (__pred) * (prixs[i+1]/prixs[i]-1)
					elif METHODE == "SIGNE":
						u += u * LEVIER * POURCENT * signe(__pred) * (prixs[i+1]/prixs[i]-1)
					elif METHODE == "REDUCTION PERTES":
						mise = u*POURCENT*abs(__pred)
						gain = mise*LEVIER*signe(__pred)*(prixs[i+1]/prixs[i]-1)
						assert low[ i ] <= prixs[ i ] <= hight[ i ]
						assert low[i+1] <= prixs[i+1] <= hight[i+1]
						if signe(__pred) == +1:
							if mise * LEVIER * (+1) * (low  [i+1]/prixs[i]-1) <= -mise:
								gain = -mise # Dès que la mise est perdu, la prediction est stopée
						else:
							if mise * LEVIER * (-1) * (hight[i+1]/prixs[i]-1) <= -mise:
								gain = -mise # Dès que la mise est perdu, la prediction est stopée
								
						u += gain
					else:
						raise Exception(f"Pas de METHODE : {METHODE}")

					if (u <= 0): u = 0
					usd += [u]

				if version == +1:
					axs[_i][_j].plot(usd, 'b', label=f'mdt={METHODE} %={POURCENT} x1 u={usd[-1]}')
				else:
					axs[_i][_j].twinx().plot(usd, 'r', label=f'mdt={METHODE} %={POURCENT} x-1 u={usd[-1]}')
			axs[_i][_j].legend()
	plt.show()