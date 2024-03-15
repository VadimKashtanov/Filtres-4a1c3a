#! /usr/bin/python3

from mdl import *

if __name__ == "__main__":
	mdl = Mdl("mdl.bin")

	mdl.incruster_DOT1D(26, 32)

	mdl.ecrire("mdl.bin")