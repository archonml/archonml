# # # # # # # # # # # # #
#
# This script extracts TDDFT data from a gaussian output file.
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
while "nstates=" not in Local[buffer]:
    buffer += 1
NStates = int((Local[buffer].split('nstates=')[1]).split(',')[0])*2

# Find Total Energy. Necessary for calculating absolute energies of ES.
while "SCF Done" not in Local[buffer]:
    buffer += 1
TotalEN = float(Local[buffer].split()[4])
print("Total energy found as :", TotalEN)

# 0: EXC EN
# 1: MuX
# 2: MuY
# 3: MuZ
# 4: Mu**2
# 5: fOsc
# 6 : S**2
TDS = np.zeros((NStates, 7))

# Store Transition Dipoles, Mu**2, fOsc
while "Excited states from <AA,BB:AA,BB> singles matrix" not in Local[buffer]:
  buffer += 1
buffer += 6
for ii in range(NStates):
    TDS[ii,1] = float(Local[buffer].split()[1])
    TDS[ii,2] = float(Local[buffer].split()[2])
    TDS[ii,3] = float(Local[buffer].split()[3])
    TDS[ii,4] = float(Local[buffer].split()[4])
    TDS[ii,5] = float(Local[buffer].split()[5])
    buffer += 1

# Get energies and S**2
count = 0
while count < NStates:
    # Jump to next state
    while "Excited State " not in Local[buffer]:
        buffer += 1
    # Get Energy
    Aux1 = Local[buffer].split("eV")[0] # get everything before the energy unit
    TDS[count,0] = float(Aux1.split()[-1])
    # Get S**2
    TDS[count,6] = float(Local[buffer].split("=")[-1])
    buffer += 1
    count += 1

# Write Output
t = open('TDData.log', 'w')
t.write("#State    ExcEn[eV]        MuX              MuY              MuZ              Mu**2            fOsc             AbsEN             S**2\n")
# Write GS with Asterisk on it, to signal this as reference optimized state with reference egnergy
SSS = '000*     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}    {: 3.2f}\n'.format(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TotalEN, 0.0)
t.write(SSS)
for ii in range(NStates):
    Aux = TDS[ii,0] * conv
    AbsEN = TotalEN + Aux
    SSS = '{:03d}      {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}    {: 3.2f}\n'
    t.write(SSS.format(ii+1, TDS[ii,0], TDS[ii,1], TDS[ii,2], TDS[ii,3], TDS[ii,4], TDS[ii,5], AbsEN, TDS[ii,6]))
t.close()
