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
FID = open('OrbEns.out', "r")
FLoc = FID.readlines()
FID.close()

gid = open("OrbInfo", "w")
# Get the number of occupied MOs. Assume that this is NEL/2 in all N calculations.
line = 0
while "Number of Electrons" not in FLoc[line]:
    line += 1
LocNOcc = int(FLoc[line].split()[-1])/2
SSS = "    {} alpha electrons       {} beta electrons\n".format(int(LocNOcc), int(LocNOcc))
gid.write(SSS)                                         # Stores NOcc

# Get the SEmp SCF Energy
while "Total Energy" not in FLoc[line]:
    line += 1
LocSEmpEn = float((FLoc[line].split(":")[1]).split("Eh")[0])
gid.write("{}\n".format(LocSEmpEn))                           # Stores SEmp Energy

# Here comes reading of orbital information.
g = open("OrbEns", "w")
# Get the Occupied and Virtual Orbital Energies
line = len(FLoc)-1
while "ORBITAL ENERGIES" not in FLoc[line]:
    line -= 1
line += 4
StartLine = 0 + dc(line)
LocOCCEn  = []
LocVIRTEn = []
while "------------------" not in FLoc[line]:
    if "2." in FLoc[line].split()[1]:                                                   # this is an occupied orbital
        LocOCCEn.append(float(FLoc[line].split()[2]))
    if "0." in FLoc[line].split()[1]:
        LocVIRTEn.append(float(FLoc[line].split()[2]))
    line += 1
# Construct the Merged Format output
SSS = "{} {: 12.9f}\n"
for ii in range(len(LocOCCEn)):
    LocID = -int(len(LocOCCEn)) + ii + 1
    g.write(SSS.format(LocID, LocOCCEn[ii]))
for ii in range(len(LocVIRTEn)):
    LocID = ii
    g.write(SSS.format(LocID, LocVIRTEn[ii]))
g.close()

# Here comes the rest of the Molecular orbital section.
while "MOLECULAR ORBITALS" not in FLoc[line]:
    line += 1
gid.write("Molecular Orbital Coefficients:\n")
line += 2
StartLine = dc(line)
# reconstruct the overall number of NAOs, MOs and blocks
NLine = 4
line += 4
while "--------" not in FLoc[line]:
    NLine += 1
    line  += 1
NLine -= 3
LocNAOs = NLine - 4
NBlocks = int((LocNAOs - (LocNAOs % 6)) / 6)
if (LocNAOs % 6) > 0:
    NBlocks += 1
# put buffer to the first line again
line = dc(StartLine)
# Generate, which lines contain the valuable orbital information.
for jj in range(NBlocks):
    Nu   = 6*jj                                      # Block Number
    Theo = [1+Nu, 2+Nu, 3+Nu, 4+Nu, 5+Nu, 6+Nu]      # theoretical block numbers in these lines
    if (LocNOcc-MOWin) in Theo:
        TLinSS = StartLine + NLine * jj
    if (LocNOcc+MOWin) in Theo:
        TLinES = StartLine + NLine * jj
        TLinE  = TLinES + NLine
for PLine in range(TLinSS, TLinE):
    gid.write(FLoc[PLine])
gid.close()

# Get the Mulliken Charges
oid = open("Mulliken", "w")
line = 0
while "MULLIKEN ATOMIC CHARGE" not in FLoc[line]:
    line += 1
line += 2
ReadL = dc(line)
while "Sum of atomic charges:" not in FLoc[line]:
    line += 1
StopL   = dc(line)
NLine   = StopL - ReadL
buffer  = ReadL
for ii in range(NLine):
    LocMull = FLoc[buffer+ii]
    oid.write(LocMull)
oid.close()
