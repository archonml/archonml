# # # # # # # # # # # # #
#
# This script extracts TDDFT data from an orca output file.
#
# FW 08/08/23

import os

import numpy as np

conv = 0.0367493

# Read output and look for nstates. Allocate Matrix.
f = open('TDST.out','r')
Local = f.readlines()
f.close()
buffer = 0
while "nroots" not in Local[buffer]:
    buffer += 1
NStates = int(Local[buffer].split()[-1])*2       # double to get 'TDST'.

# Find Total Energy. Necessary for calculating absolute energies of ES.
while "Total Energy" not in Local[buffer]:
    buffer += 1
TotalEN = float(Local[buffer].split()[3])
print("Total energy found as :", TotalEN)

# 0: EXC EN - in eV
# 1: MuX
# 2: MuY
# 3: MuZ
# 4: Mu**2
# 5: fOsc
# 6 : S**2 - has to be read from STATE stream.
TDS = np.zeros((NStates, 7))

# Step 1 - read all excitation energies and S**2 values.
line = 0
for ii in range(NStates):
    while "S**2" not in Local[line]:
        line += 1
    # Grab the state S**2 and Exc EN
    TDS[ii, 0] = float((Local[line].split("au")[0]).split()[-1])
    TDS[ii, 6] = float(Local[line].split()[-1])
    line += 1

# Step 2 - read the remaining stuff from the "end" output.
line = 0
while "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" not in Local[line]:
    line += 1
line += 5
for ii in range(NStates):
    if "spin forbidden" not in Local[line]:        # Only update the lines that do not contain "spin forbidden" - all else are 0.0 anyway.
        TDS[ii, 1] = float(Local[line].split()[5])    # 1: MuX
        TDS[ii, 2] = float(Local[line].split()[6])    # 2: MuY
        TDS[ii, 3] = float(Local[line].split()[7])    # 3: MuZ
        TDS[ii, 4] = float(Local[line].split()[4])    # 4: Mu**2
        TDS[ii, 5] = float(Local[line].split()[3])    # 5: fOsc
    line += 1

# Sort now w.r.t. excitation energies.
sortarr  = TDS[:, 0]
sorti    = np.argsort(sortarr)

# Write Output
t = open('TDData.log', 'w')
t.write("#State    ExcEn[eV]        MuX              MuY              MuZ              Mu**2            fOsc             AbsEN             S**2\n")

# Write GS with Asterisk on it, to signal this as reference optimized state with reference egnergy
SSS = '000*     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}    {: 3.2f}\n'.format(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TotalEN, 0.0)
t.write(SSS)
for ii in range(NStates):
    Aux = TDS[sorti[ii],0] * conv
    AbsEN = TotalEN + Aux
    SSS = '{:03d}      {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}    {: 3.2f}\n'
    t.write(SSS.format(ii+1, TDS[sorti[ii] ,0], TDS[sorti[ii] ,1], TDS[sorti[ii] ,2], TDS[sorti[ii] ,3], TDS[sorti[ii] ,4], TDS[sorti[ii] ,5], AbsEN, TDS[sorti[ii] ,6]))
t.close()
