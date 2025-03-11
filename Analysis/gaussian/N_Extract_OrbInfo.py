# # # # # # # # # # # # #
#
# This script extracts Orbital Information and Mulliken charges data from a gaussian output file.
#
# FW 08/08/23

import os
from copy import deepcopy as dc

# Change this value if necessary. This will control how long the reduced output will be written.
MOWin = 10

# Look for the output file
f = open("OrbEns.out", "r")
Local = f.readlines()
f.close()
gid = open("OrbInfo", "w")
line = 0
while "alpha electrons" not in Local[line]:
    line += 1
NOcc = int(Local[line].split()[0])
gid.write(Local[line])
while "SCF Done" not in Local[line]:
    line += 1
PM6En = float((Local[line].split("=")[1]).split("A.U.")[0])
gid.write("{}\n".format(PM6En))
line = len(Local)-1
while "Population analysis using the SCF Density." not in Local[line]:
    line -= 1
line += 4
StartLine = 0 + line
OCCS = []
VIRT = []

# Import the Energies
while "eigenvalues" in Local[line]:
    if "occ" in Local[line]:
        PreOC = Local[line].split("Alpha  occ. eigenvalues --")[1].replace("-", " -")
        OC    = PreOC.split()
        OCLen = len(OC)
        for ii in range(OCLen):
            OCCS.append(float(OC[ii]))
    if "virt" in Local[line]:
        VR    = Local[line].split("Alpha virt. eigenvalues --")[1].split()
        VRLen = len(VR)
        for ii in range(VRLen):
            VIRT.append(float(VR[ii]))
    line += 1

# Write the Output
g = open("OrbEns", "w")
S = "{} {: 12.9f}\n"
for ii in range(len(OCCS)):
    ID = -int(len(OCCS))+ii+1
    g.write(S.format(ID, OCCS[ii]))
for ii in range(len(VIRT)):
    ID = ii
    g.write(S.format(ID, VIRT[ii]))
g.close()

# Jump to Molecular Orbital Section
while "Molecular Orbital Coefficients:" not in Local[line]:
    line += 1
gid.write(Local[line])
line += 1
Rewi = dc(line)

# count how many lines for each block
NLine = 3
line += 3
while "Eigenvalues" not in Local[line]:
    NLine += 1
    line  += 1
NLine -= 2
NAOs    = NLine - 2
NBlocks = int((NAOs - (NAOs % 5)) / 5)
if (NAOs % 5) > 0:
    NBlocks += 1
line = Rewi	# put buffer to the first line again

# Go to the first orbital and just copy all the stuff to file
for jj in range(NBlocks):
    Nu = 5*jj
    Theo = [1+Nu, 2+Nu, 3+Nu, 4+Nu, 5+Nu]
    if NOcc-MOWin in Theo:
        TLinSS = Rewi+NLine*jj
    if NOcc+MOWin in Theo:
        TLinES = Rewi+NLine*jj
        TLinE = TLinES + NLine
for PLine in range(TLinSS, TLinE):
    gid.write(Local[PLine])
gid.close()

# Get the Mulliken Charges
oid = open("Mulliken", "w")
line = 0
while "Mulliken charges:" not in Local[line]:
    line += 1
line += 2
ReadL = dc(line)
while "Sum of Mulliken charges" not in Local[line]:
    line += 1
StopL   = dc(line)
NLine   = StopL - ReadL
buffer  = ReadL
for ii in range(NLine):
    LocMull = Local[buffer+ii]
    oid.write(LocMull)
oid.close()
