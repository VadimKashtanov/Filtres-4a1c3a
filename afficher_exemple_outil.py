from sys import argv

PRIXS = int(argv[1])

import matplotlib.pyplot as plt
import struct as st

def lire_liste(fichier):
	with open(fichier, "rb") as co:
		bins = co.read()
		return st.unpack('f'*PRIXS, bins)

fig, ax = plt.subplots(2,1)

ax[0].plot(lire_liste("tmp_source.float"))
ax[0].plot(lire_liste("tmp_ema.float"))
#
ax[1].plot(lire_liste("tmp_brute.float"))
ax[1].plot(lire_liste("tmp_filtre.float"))
#
plt.show()