from os import system
from pybitget import Client

client = Client(
	'bg_b13daa6f909560dd8dfac7549e9ce83f',
	'd1c9e1c5c520a74b89133945f79246929e27908187cdb2fd51bab280b0dd98e9',
	'Henrya10ans')

argent = lambda : float(client.mix_get_account('BTCUSDT_UMCBL', 'USDT')['data']['usdtEquity'])

print(f"Le compte a {argent()} $")

SYMBOLE     = 'BTCUSDT_UMCBL'
productType = 'umcbl'
marginCoin  = 'USDT'

from def_futures_bitget import fermer_les_positions, executer_future, derniere_prediction

POURCENT = 0.10

fermer_les_positions()

#   === Achat / Vente ===

__pred = derniere_prediction()
print(f"Prediction : {__pred}")
#executer_future(client, POURCENT*abs(__pred), __pred)
