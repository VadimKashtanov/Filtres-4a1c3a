prixs, haut, bas, volumes, median = "SRC_PRIXS", "SRC_HIGH", "SRC_LOW", "SRC_VOLUMES", "SRC_MEDIAN"

directes = "DIRECT", """
{'K': 1, 'interv': 1, 'params': ()}
{'K': 1, 'interv': 2, 'params': ()}
{'K': 1, 'interv': 2, 'params': ()}
{'K': 1, 'interv': 4, 'params': ()}
{'K': 2, 'interv': 1.0, 'params': ()}
{'K': 2, 'interv': 2, 'params': ()}
{'K': 2, 'interv': 4, 'params': ()}
{'K': 2, 'interv': 8, 'params': ()}
{'K': 4, 'interv': 2.0, 'params': ()}
{'K': 4, 'interv': 4, 'params': ()}
{'K': 4, 'interv': 8, 'params': ()}
{'K': 4, 'interv': 16, 'params': ()}
{'K': 8, 'interv': 4.0, 'params': ()}
{'K': 8, 'interv': 8, 'params': ()}
{'K': 8, 'interv': 16, 'params': ()}
{'K': 8, 'interv': 32, 'params': ()}
{'K': 16, 'interv': 8.0, 'params': ()}
{'K': 16, 'interv': 8.0, 'params': ()}
{'K': 16, 'interv': 16, 'params': ()}
{'K': 16, 'interv': 32, 'params': ()}
{'K': 16, 'interv': 64, 'params': ()}
{'K': 32, 'interv': 16.0, 'params': ()}
{'K': 32, 'interv': 32, 'params': ()}
{'K': 32, 'interv': 64, 'params': ()}
{'K': 32, 'interv': 128, 'params': ()}
{'K': 64, 'interv': 32.0, 'params': ()}
{'K': 64, 'interv': 64, 'params': ()}
{'K': 64, 'interv': 128, 'params': ()}
{'K': 64, 'interv': 256, 'params': ()}
{'K': 128, 'interv': 64.0, 'params': ()}
{'K': 128, 'interv': 128, 'params': ()}
{'K': 128, 'interv': 256, 'params': ()}
{'K': 256, 'interv': 128.0, 'params': ()}
{'K': 256, 'interv': 256, 'params': ()}
""", (prixs, haut, bas, volumes), "cree_DIRECTE"
#""", (prixs, haut, bas, volumes,median), "cree_DIRECTE"


macds = "MACD", """
{'K': 1, 'interv': 8, 'params': (1,)}
{'K': 1, 'interv': 8, 'params': (4,)}
{'K': 2, 'interv': 2, 'params': (1,)}
{'K': 2, 'interv': 16, 'params': (2,)}
{'K': 2, 'interv': 16, 'params': (8,)}
{'K': 4, 'interv': 2.0, 'params': (1,)}
{'K': 4, 'interv': 4, 'params': (2,)}
{'K': 4, 'interv': 32, 'params': (4,)}
{'K': 4, 'interv': 32, 'params': (16,)}
{'K': 8, 'interv': 4.0, 'params': (2,)}
{'K': 8, 'interv': 8, 'params': (1,)}
{'K': 8, 'interv': 64, 'params': (8,)}
{'K': 8, 'interv': 64, 'params': (32,)}
{'K': 16, 'interv': 8.0, 'params': (1,)}
{'K': 16, 'interv': 8.0, 'params': (4,)}
{'K': 16, 'interv': 16, 'params': (8,)}
{'K': 16, 'interv': 128, 'params': (16,)}
{'K': 16, 'interv': 128, 'params': (64,)}
{'K': 32, 'interv': 16.0, 'params': (2,)}
{'K': 32, 'interv': 16.0, 'params': (8,)}
{'K': 32, 'interv': 32, 'params': (4,)}
{'K': 32, 'interv': 32, 'params': (16,)}
{'K': 32, 'interv': 256, 'params': (128,)}
{'K': 64, 'interv': 32.0, 'params': (4,)}
{'K': 64, 'interv': 32.0, 'params': (16,)}
{'K': 64, 'interv': 64, 'params': (8,)}
{'K': 256, 'interv': 128.0, 'params': (16,)}
{'K': 256, 'interv': 128.0, 'params': (64,)}
{'K': 256, 'interv': 256, 'params': (32,)}
{'K': 256, 'interv': 256, 'params': (128,)}
""", (prixs, haut, bas, volumes), "cree_MACD"
#""", (prixs, haut, bas, volumes, median), "cree_MACD"

chiffres = "CHIFFRE", """
{'K': 1, 'interv': 4, 'params': (10000,)}
{'K': 1, 'interv': 8, 'params': (10000,)}
{'K': 1, 'interv': 8, 'params': (10000,)}
{'K': 8, 'interv': 4.0, 'params': (10000,)}
{'K': 8, 'interv': 8, 'params': (10000,)}
{'K': 8, 'interv': 8, 'params': (10000,)}
{'K': 8, 'interv': 32, 'params': (10000,)}
{'K': 8, 'interv': 32, 'params': (10000,)}
{'K': 8, 'interv': 64, 'params': (10000,)}
{'K': 8, 'interv': 64, 'params': (10000,)}
{'K': 64, 'interv': 32.0, 'params': (10000,)}
{'K': 64, 'interv': 32.0, 'params': (10000,)}
{'K': 64, 'interv': 64, 'params': (10000,)}
{'K': 64, 'interv': 64, 'params': (10000,)}
{'K': 64, 'interv': 256, 'params': (10000,)}
{'K': 64, 'interv': 256, 'params': (10000,)}
{'K': 256, 'interv': 128.0, 'params': (10000,)}
{'K': 256, 'interv': 128.0, 'params': (10000,)}
{'K': 256, 'interv': 256, 'params': (10000,)}
{'K': 256, 'interv': 256, 'params': (10000,)}
""", (prixs, haut, bas), "cree_CHIFFRE"

awesome = "AWESOME", """
{'K': 1, 'interv': 8, 'params': (1,)}
{'K': 1, 'interv': 8, 'params': (4,)}
{'K': 2, 'interv': 2, 'params': (1,)}
{'K': 2, 'interv': 16, 'params': (2,)}
{'K': 2, 'interv': 16, 'params': (8,)}
{'K': 8, 'interv': 4.0, 'params': (2,)}
{'K': 8, 'interv': 8, 'params': (1,)}
{'K': 8, 'interv': 8, 'params': (4,)}
{'K': 8, 'interv': 64, 'params': (8,)}
{'K': 8, 'interv': 64, 'params': (32,)}
{'K': 32, 'interv': 16.0, 'params': (2,)}
{'K': 32, 'interv': 16.0, 'params': (8,)}
{'K': 32, 'interv': 32, 'params': (4,)}
{'K': 32, 'interv': 32, 'params': (16,)}
{'K': 32, 'interv': 256, 'params': (32,)}
{'K': 32, 'interv': 256, 'params': (128,)}
{'K': 64, 'interv': 32.0, 'params': (4,)}
{'K': 64, 'interv': 32.0, 'params': (16,)}
{'K': 64, 'interv': 64, 'params': (8,)}
{'K': 64, 'interv': 64, 'params': (32,)}
{'K': 128, 'interv': 64.0, 'params': (8,)}
{'K': 128, 'interv': 64.0, 'params': (32,)}
{'K': 128, 'interv': 128, 'params': (16,)}
{'K': 128, 'interv': 128, 'params': (64,)}
{'K': 256, 'interv': 128.0, 'params': (16,)}
{'K': 256, 'interv': 128.0, 'params': (64,)}
{'K': 256, 'interv': 256, 'params': (32,)}
{'K': 256, 'interv': 256, 'params': (128,)}
""", (prixs, haut, bas, volumes), "cree_AWESOME"
#""", (prixs, haut, bas, volumes, median), "cree_AWESOME"

pourcent_r = "POURCENT_R", """
{'K': 1, 'interv': 4, 'params': (4, 2)}
{'K': 1, 'interv': 8, 'params': (8, 2)}
{'K': 1, 'interv': 8, 'params': (8, 2)}
{'K': 4, 'interv': 2.0, 'params': (2.0, 2)}
{'K': 4, 'interv': 4, 'params': (4, 2)}
{'K': 4, 'interv': 16, 'params': (16, 2)}
{'K': 4, 'interv': 16, 'params': (16, 2)}
{'K': 4, 'interv': 32, 'params': (32, 2)}
{'K': 4, 'interv': 32, 'params': (32, 2)}
{'K': 16, 'interv': 8.0, 'params': (8.0, 2)}
{'K': 16, 'interv': 8.0, 'params': (8.0, 2)}
{'K': 16, 'interv': 16, 'params': (16, 2)}
{'K': 16, 'interv': 16, 'params': (16, 2)}
{'K': 16, 'interv': 64, 'params': (64, 2)}
{'K': 16, 'interv': 64, 'params': (64, 2)}
{'K': 16, 'interv': 128, 'params': (128, 2)}
{'K': 16, 'interv': 128, 'params': (128, 2)}
{'K': 32, 'interv': 16.0, 'params': (16.0, 2)}
{'K': 32, 'interv': 16.0, 'params': (16.0, 2)}
{'K': 32, 'interv': 32, 'params': (32, 2)}
{'K': 32, 'interv': 32, 'params': (32, 2)}
{'K': 32, 'interv': 128, 'params': (128, 2)}
{'K': 32, 'interv': 128, 'params': (128, 2)}
{'K': 32, 'interv': 256, 'params': (256, 2)}
{'K': 32, 'interv': 256, 'params': (256, 2)}
{'K': 64, 'interv': 32.0, 'params': (32.0, 2)}
{'K': 64, 'interv': 32.0, 'params': (32.0, 2)}
{'K': 64, 'interv': 64, 'params': (64, 2)}
{'K': 64, 'interv': 64, 'params': (64, 2)}
{'K': 64, 'interv': 256, 'params': (256, 2)}
{'K': 64, 'interv': 256, 'params': (256, 2)}
{'K': 256, 'interv': 128.0, 'params': (128.0, 2)}
{'K': 256, 'interv': 128.0, 'params': (128.0, 2)}
{'K': 256, 'interv': 256, 'params': (256, 2)}
{'K': 256, 'interv': 256, 'params': (256, 2)}
""", (prixs, haut, bas, volumes, median), "cree_POURCENT_R"

rsi = "RSI", """
{'K': 1, 'interv': 4, 'params': (4,)}
{'K': 1, 'interv': 8, 'params': (8,)}
{'K': 1, 'interv': 8, 'params': (8,)}
{'K': 4, 'interv': 2.0, 'params': (2.0,)}
{'K': 4, 'interv': 4, 'params': (4,)}
{'K': 4, 'interv': 16, 'params': (16,)}
{'K': 4, 'interv': 16, 'params': (16,)}
{'K': 4, 'interv': 32, 'params': (32,)}
{'K': 4, 'interv': 32, 'params': (32,)}
{'K': 16, 'interv': 8.0, 'params': (8.0,)}
{'K': 16, 'interv': 8.0, 'params': (8.0,)}
{'K': 16, 'interv': 16, 'params': (16,)}
{'K': 16, 'interv': 16, 'params': (16,)}
{'K': 16, 'interv': 64, 'params': (64,)}
{'K': 16, 'interv': 64, 'params': (64,)}
{'K': 16, 'interv': 128, 'params': (128,)}
{'K': 16, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 16.0, 'params': (16.0,)}
{'K': 32, 'interv': 16.0, 'params': (16.0,)}
{'K': 32, 'interv': 32, 'params': (32,)}
{'K': 32, 'interv': 32, 'params': (32,)}
{'K': 32, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 128, 'params': (128,)}
{'K': 32, 'interv': 256, 'params': (256,)}
{'K': 32, 'interv': 256, 'params': (256,)}
{'K': 64, 'interv': 32.0, 'params': (32.0,)}
{'K': 64, 'interv': 32.0, 'params': (32.0,)}
{'K': 64, 'interv': 64, 'params': (64,)}
{'K': 64, 'interv': 64, 'params': (64,)}
{'K': 64, 'interv': 256, 'params': (256,)}
{'K': 64, 'interv': 256, 'params': (256,)}
{'K': 256, 'interv': 128.0, 'params': (128.0,)}
{'K': 256, 'interv': 128.0, 'params': (128.0,)}
{'K': 256, 'interv': 256, 'params': (256,)}
{'K': 256, 'interv': 256, 'params': (256,)}
""", (prixs, haut, bas, volumes, median), "cree_RSI"

k = 0
for nom, lignes, sources, fonc_params in (directes, macds,):#, awesome,):# chiffres, pourcent_r, rsi):
	lignes = list(map(eval, lignes.strip('\n').split('\n')))
	print("\t// -------")
	for src in sources:
		for i in lignes:
			print(f"\t\tcree_ligne({src}, {nom}, {i['K']}, {i['interv']}, {fonc_params}{str(i['params']).replace(',)',')')}),")
			k += 1

print(f"\nlignes = {k}")