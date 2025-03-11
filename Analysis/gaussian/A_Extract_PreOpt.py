# # # # # # # # # # # # #
#
# This script extracts the PreOptimized structure from a formatted checkpoint file (fchk) after a gaussian calculation.
#
# FW 08/08/23

import os

import numpy as np

from archonml.common import PSEIDICT

bohr  = 0.52917721

# Look for the formchk file
f = open('OPT.fchk','r')
Local = f.readlines()
f.close()

# Look for the atoms.
buffer = 0
while "Atomic numbers" not in Local[buffer]:
    buffer += 1
NAtm = int(Local[buffer].split()[-1])

# Allocate Array
Type = np.zeros((NAtm))
buffer += 1
count = -1

while count < NAtm-1:
    LocLen = len(Local[buffer].split())
    for ii in range(LocLen):
        count += 1
        Type[count] = int(Local[buffer].split()[ii])
    buffer += 1

while "Current cartesian coordinates" not in Local[buffer]:
    buffer += 1
NCoord = int(Local[buffer].split()[-1])
Coords = np.zeros((NCoord))
buffer += 1

# Read "flat" coordinates
count = 0
while count < NCoord:
    LocLen = len(Local[buffer].split())
    for ii in range(LocLen):
        Coords[count] = float(Local[buffer].split()[ii])*bohr
        count += 1
    buffer += 1
Geom = np.reshape(Coords, [NAtm,3])

t = open('PreOpted.xyz', 'w')
t.write(str(NAtm)+"\n\n")
for ii in range(NAtm):
    SSS = "{:2}     {: 12.9f}     {: 12.9f}     {: 12.9f}\n".format(PSEIDICT[Type[ii]], Geom[ii,0], Geom[ii,1], Geom[ii,2])
    t.write(SSS)
    buffer += 1
t.close()
