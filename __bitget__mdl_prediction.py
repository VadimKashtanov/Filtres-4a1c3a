from mdl import *

if __name__ == "__main__":
	from sys import argv

	with open(argv[1], "w") as co:
		co.write(str(Mdl("mdl.bin")()[-1]))