# # # # # # # # # # # # #
# General utility functions used throughout all different modules.
#
# Not explicitly loaded in the package initialization, but instead, the modules themselves
# grab whatever general functions they need from here.
#
# The idea is, that the main modules just become less cluttered with functions.
#
# FW 07/13/23

from copy import deepcopy as dc

import os, psutil
import numpy as np
import mendeleev as md
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdqueries
from numba import jit, njit, types, vectorize

# Utility for printing currently used RAM. Useful for debugging purposes.
def procram():
    proc = psutil.Process(os.getpid())
    print("Taking up {} MB.".format(((proc.memory_info().rss)/1024)/1024))
    return None

# Utility for printing currently used RAM. Useful for debugging purposes.
def ramval():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss


# Distance between two points. Sped up using numba.
@njit
def dist(P_A, P_B):
    DX = P_A[0] - P_B[0]
    DY = P_A[1] - P_B[1]
    DZ = P_A[2] - P_B[2]
    DD = np.sqrt(DX**2 + DY**2 + DZ**2)
    return DD

# Formation of an Distance Array - i.e. all distances from point P_A inside a Geometry.
# Sped up using numba.
@njit
def darr(P_A, Geom):
    NRou = len(Geom)
    DRou = np.zeros((NRou))
    for ij in range(NRou):
        DRou[ij] = dist(P_A, Geom[ij,:])
    return DRou

# Converting seconds into general DD:HH:MM:SS format.
def timeconv(inptime):
    days    = inptime // (24 * 3600)
    time    = inptime - (days*(24 * 3600))
    hours   = time // 3600
    time    = time - (hours*3600)
    minutes = time // 60
    time    = time - (minutes*60)
    seconds = time
    return "{:02d}:{:02d}:{:02d}:{:02d}".format(int(days), int(hours), int(minutes), int(seconds))

# Determines the sorted Eigenvalues for the Euclidean Norm of Coulomb-Matrix(like) objects.
# Used in Descriptor Module
@njit(nogil=True)
def eig_coul(LocCMat):
    Aux1 = np.linalg.eigvalsh(LocCMat)
    Aux2 = np.zeros((len(Aux1)))
    for ii in range(len(Aux1)):
        Aux2[ii] = abs(Aux1[ii])
    Order = np.argsort(-Aux2)
    V = np.zeros((len(Aux1)))
    for ii in range(len(Aux1)):
        V[ii] = Aux1[Order[ii]]
    return V

# Generates a Coulomb-Matrix(like) object according to the original formulation.
# Used in Descriptor Module
@njit(nogil=True)
def coulomb_mat(LocGeom):
    NAt = len(LocGeom)
    LocCMat = np.zeros((NAt, NAt))
    for ii in range(NAt):
        for jj in range(NAt):
            if ii == jj:
                LocCMat[ii, jj] = 0.5 * float(LocGeom[ii, 3])**(2.4)
            else:
                Aux1 = float(LocGeom[ii, 3]) * float(LocGeom[jj, 3])
                LocCMat[ii, jj] = Aux1 / dist(LocGeom[ii, :], LocGeom[jj, :])
    return LocCMat

# Generates a Coulomb-Matrix(like) object with a signed Denominator according to an additional Array.
# Used in Descriptor Module
def coulomb_mat_sing_sign(LocGeom, Arr):
    NAt = len(LocGeom)
    LocCMat = np.zeros((NAt, NAt))
    for ii in range(NAt):
        for jj in range(NAt):
            if ii == jj:
                # Use only diagonal atom type
                Aux1 = abs(float(LocGeom[ii, 3]))
                LocCMat[ii, jj] = 0.5 * Aux1**(2.4)
            else:
                # Apply both charge and atom types
                Aux1 = (float(LocGeom[ii,3]) * Arr[ii]) * (float(LocGeom[jj, 3]) * Arr[jj])
                LocCMat[ii, jj] = Aux1 / dist(LocGeom[ii, :], LocGeom[jj, :])
    return LocCMat

# Generates a Coulomb-Matrix(like) object with additional factors according to an external Array. All values as absolutes.
# Used in Descriptor Module
def coulomb_mat_duo_abs(LocGeom, Arr):
    NAt = len(LocGeom)
    LocCMat = np.zeros((NAt, NAt))
    for ii in range(NAt):
        for jj in range(NAt):
            if ii == jj:
                # Use only diagonal atom type
                Aux1 = abs(float(LocGeom[ii, 3]) * float(Arr[ii]))   # Applies both el change and atom type
                LocCMat[ii, jj] = 0.5 * Aux1**(2.4)
            else:
                Aux1 = (float(LocGeom[ii,3]) * abs(Arr[ii])) * (float(LocGeom[jj, 3]) * abs(Arr[jj]))
                LocCMat[ii, jj] = Aux1 / dist(LocGeom[ii, :], LocGeom[jj, :])
    return LocCMat

# Generates a Coulomb-Matrix(like) object with additional factors according to an external Array. Denominator is doubly signed.
# Used in Descriptor Module
def coulomb_mat_duo_sign(LocGeom, Arr):
    NAt = len(LocGeom)
    LocCMat = np.zeros((NAt, NAt))
    for ii in range(NAt):
        for jj in range(NAt):
            if ii == jj:
                # Use only diagonal atom type
                Aux1 = abs(float(LocGeom[ii, 3]) * Arr[ii])   # Applies both el change and atom type
                LocCMat[ii, jj] = 0.5 * Aux1**(2.4)
            else:
                Aux1 = (float(LocGeom[ii,3]) * Arr[ii]) * (float(LocGeom[jj, 3]) * Arr[jj])
                LocCMat[ii, jj] = Aux1 / dist(LocGeom[ii, :], LocGeom[jj, :])
    return LocCMat


# Distance Matrices, Tensor Generators
# Used in DataTens Module.
@njit
def dist_mat_calc(InpArr):
    DistMat = np.zeros((len(InpArr), len(InpArr)))
    for ii in range(len(InpArr)):
        for jj in range(len(InpArr)):
            DistMat[ii,jj] = abs(InpArr[ii]-InpArr[jj])
    return DistMat

# Distance Matrices, Tensor Generators
# Used in DataTens Module.
@njit
def dist_mat_calc(InpArr):
    DistMat = np.zeros((len(InpArr), len(InpArr)))
    for ii in range(len(InpArr)):
        for jj in range(len(InpArr)):
            DistMat[ii,jj] = abs(InpArr[ii]-InpArr[jj])
    return DistMat

# Operations between 2 Molecule Objects' properties w.r.t. to the eigenvalues distances
# Takes care of differences in lengths of eigenvectors by supplementing with zeros.
# This is a subroutine to EiucDistMatCalc.
@njit(nogil=True)
def euc_dist(LocEig1, LocEig2, XpSig=None):
    LocLen1 = len(LocEig1)
    LocLen2 = len(LocEig2)
    LocSame = (LocLen1 == LocLen2)
    if not LocSame:
        if LocLen2 < LocLen1:
            for ii in range(LocLen1-LocLen2):
                Aux1 = np.append(LocEig2, 0.0)
                LocEig2 = Aux1
        else:
            for ii in range(LocLen2-LocLen1):
                Aux1 = np.append(LocEig1, 0.0)
                LocEig1 = Aux1
    EucD = 0
    Aux2 = 0
    for ii in range(len(LocEig1)):
        Aux1 = abs(LocEig1[ii] - LocEig2[ii])**2
        Aux2 += Aux1
    EucD = np.sqrt(Aux2)
    if XpSig != None:
        Aux1 = -1*(1/(2*XpSig**2))*(EucD)**2
        EuXp = np.exp(Aux1)
        return EuXp
    else:
        return EucD

# Calculation of Distance Matrices through formation of  Euclidean Norms.
# Used in DataTens Module.
def euc_dist_mat_calc(InpObj):
    DistMat = np.zeros((len(InpObj), len(InpObj)))
    for ii in range(len(InpObj)):
        for jj in range(len(InpObj)):
            DistMat[ii,jj] = euc_dist(InpObj[ii], InpObj[jj])
    return DistMat

# This function guesses an initial value for sigma_g for one specific descriptor "g".
# The guess is based off the distance matrix of the training data. Here, we look at the most often
# encountered distance in a histogram plot, and then search for a "full width at half maximum" around
# this maximum, which will become the initial guess.
# Note, that the histogram's binning resolution is dynamically adapted - such that the FWHM will only
# be directly accepted if at least a specific (user-defined) resolution lies between "left" and "right"
# values.
# If the resolution can never be met within ESC tries, the strictest FWHM (which does not meet the
# resolution) will be taken as guess instead.
# In case, NO FWHM can be found at all, the guess will take the most often encountered value "minus 90%
# of its value". Consider this to be a failsafe option for the function.
def guess_sigma(DMAT, WINTHRSH, ESC):
    FWHMS       = []
    BINS        = []
    TBINS       = 3           # This variable stores the current (dynamically changed) resolution for the overall histogram
    BINWIN      = 2           # This variable checks how many points lie between left and right value. initialized as 2 to stay under TBINS.
    EscapeCnt   = 0           # This variable counts how many attempts of dynamically changing the resolution have been made so far.


    while BINWIN < WINTHRSH:                          # While the current resolution between left and right edges is below the desired one:... 
        counts, bins = np.histogram(DMAT.flatten(), bins=TBINS)
        XHistMaxID = np.argmax(counts)
        XHistMax   = bins[XHistMaxID]                 # XHistMax is the most common distance between two molecules for this descriptor.
                                                      # Now, try to express "sigma" to both left and right sides.
        HalfMax = max(counts)/2

        # Go left
        Pos     = XHistMaxID
        buffer  = counts[Pos]
        LValid  = True
        while buffer > HalfMax:                        # Go left, until half the value has been found.
            Pos    -= 1
            buffer  = counts[Pos]
            LPos    = dc(Pos)
            LPosVal = dc(bins[Pos])
            if Pos < 0:                                # If there are no more points to the left, flag as invalid.
                LValid  = False                        # Also, overwrite the left edge position with the maximum's position.
                LPos    = dc(XHistMaxID)
                LPosVal = dc(bins[XHistMaxID])
                buffer  = dc(counts[XHistMaxID])
                break
        
        # Go right
        Pos    = XHistMaxID
        buffer = counts[Pos]
        RValid  = True
        while buffer > HalfMax:                         # Go right, until half the value has been found.
            Pos   += 1
            try:
                buffer   = counts[Pos]
                RPos     = dc(Pos)
                RPosVal  = dc(bins[Pos])
            except IndexError:                          # If no more points to the right, flag as invalid.
                RValid  = False                         # Also, overwrite the right edge position with the maximum's position.
                RPos     = dc(XHistMaxID)
                RPosVal  = dc(bins[XHistMaxID])
                buffer  = dc(counts[XHistMaxID])
                break

        if (LValid) and (RValid):                       # If valid points were found for both left and right, add
            FWHMS.append(RPosVal-LPosVal)               # as FWHM.
            BINWIN     = (RPos - LPos)                  # Determine the current resolution between the left and right edges.
            
        if (RValid) and not (LValid):                   # If only one side was valid, guess width as double the "half-width"
            FWHMS.append((RPosVal-LPosVal)*2)           # to the valid side.
            BINWIN     = (RPos - LPos)*2                # Determine the current resolution as double the resolution to the valid point.
            
        if (LValid) and not (RValid):                   # If only one side was valid, guess width as double the "half-width"
            FWHMS.append((RPosVal-LPosVal)*2)           # to the valid side.
            BINWIN     = (RPos - LPos)*2                # Determine the current resolution as double the resolution to the valid point.

        TBINS     += 1                                  # After one try is over, dynamically increase the histogram resolution...
        EscapeCnt += 1                                  # ...and increase the "Escape counter".

        if EscapeCnt > ESC:                             # If EscapeCount reached the Escape condition, break the loop "with whatever we have so far".
            break

    # now, outside the while loop.
    if EscapeCnt <= ESC:
        ReturnSig = FWHMS[-1]/(np.sqrt(2*np.log(2)))           # If loop was exited within the escape condition, this must mean that the desired resolution was reached
                                                               # at the last step. Thus, take the last stored FWHM as return value.
        
    if EscapeCnt > ESC:                                        # If loop was exited outside the escape condition...
        if len(FWHMS) != 0:                                    # ...and there were any FWHMs found at all, use the smallest (i.e. strictest) one as guess.
            ReturnSig = ((min(FWHMS))/(np.sqrt(2*np.log(2))))
        else:                                                  # ...and there were NO valid FWHMS at all, estimate a guess as 10% of the most encountered distance.
            EscapeVal   = bins[XHistMaxID]
            EscapeDist  = 0.1 * EscapeVal
            ReturnSig   = EscapeDist/(np.sqrt(2*np.log(2)))
            if EscapeDist < 0.0001:                            # ...in case the distance is practically 0, use sigma of 0.1 to "not divide by 0 sigma".
                ReturnSig = 0.1
    return ReturnSig




# Local Generation function of "Submatrices", using Training and Validation IDs. Formerly known as the "MatGen" functions.
# Uses the Locbin Global variable for setting up the Sigma Thresholds.
# NOTE - TrainIDs and ValiIDs are NUMERICAL INTEGERS here, and do not refer to the name of specific structures.
# Used in DataTens Module.
def get_sub_mat(TrainIDs, ValiIDs, DistMat, BINWINTHRSH=10, BINWINESCAPE=100):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            TrainDistMat[ii,jj] = DistMat[TrainIDs[ii], TrainIDs[jj]]
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            ValiDistMat[ii,jj] = DistMat[ValiIDs[ii], TrainIDs[jj]]
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(TrainDistMat, BINWINTHRSH, BINWINESCAPE)
    return TrainDistMat, ValiDistMat, abs(ReturnSig)


# Local Generation function of "Submatrices", using Training and Validation IDs. Formerly known as the "MatGen" functions.
# This on-the-fly version will not use the pre-calculated distance matrix but instead calculate the sub_matrices on-the-fly.
# NOTE - TrainIDs and ValiIDs are NUMERICAL INTEGERS here, and do not refer to the name of specific structures.
# Used in DataTens Module.
def otf_sub_distmat(TrainIDs, ValiIDs, MLObjList, curDesc, BINWINTHRSH=10, BINWINESCAPE=100):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                               # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc] - MLObjList[curJID].Desc[curDesc])        # Calculation of normal distance matrix elements.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc] - MLObjList[curJID].Desc[curDesc])         # Calculation of normal distance matrix elements.
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(TrainDistMat, BINWINTHRSH, BINWINESCAPE)
    return TrainDistMat, ValiDistMat, abs(ReturnSig)


# Euclidean Norm variant
def otf_sub_eucdistmat(TrainIDs, ValiIDs, MLObjList, curDesc, BINWINTHRSH=10, BINWINESCAPE=100):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                               # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc], MLObjList[curJID].Desc[curDesc])    # Calculation of Euclidean Norm goes here.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc], MLObjList[curJID].Desc[curDesc])     # Calculation of Euclidean Norm goes here.
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(TrainDistMat, BINWINTHRSH, BINWINESCAPE)
    return TrainDistMat, ValiDistMat, abs(ReturnSig)



# Normal Distance variant - layered
def otf_sub_distmat_layer(TrainIDs, ValiIDs, MLObjList, curDesc, Dep, BINWINTHRSH=10, BINWINESCAPE=100):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                                         # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc][Dep] - MLObjList[curJID].Desc[curDesc][Dep])        # Calculation of normal distance matrix elements.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                          # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc][Dep] - MLObjList[curJID].Desc[curDesc][Dep])         # Calculation of normal distance matrix elements.
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(TrainDistMat, BINWINTHRSH, BINWINESCAPE)
    return TrainDistMat, ValiDistMat, abs(ReturnSig)


# Euclidean Norm variant - layered
def otf_sub_eucdistmat_layer(TrainIDs, ValiIDs, MLObjList, curDesc, Dep, BINWINTHRSH=10, BINWINESCAPE=100):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                                         # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc][Dep], MLObjList[curJID].Desc[curDesc][Dep])    # Calculation of Euclidean Norm goes here.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                          # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc][Dep], MLObjList[curJID].Desc[curDesc][Dep])     # Calculation of Euclidean Norm goes here.
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(TrainDistMat, BINWINTHRSH, BINWINESCAPE)
            
    return TrainDistMat, ValiDistMat, abs(ReturnSig)

# Local Generation function of "Submatrices", using Training and Validation IDs. Formerly known as the "MatGen" functions.
# This on-the-fly version will not use the pre-calculated distance matrix but instead calculate the sub_matrices on-the-fly.
# Used in DataTens Module - for final training only.
def otf_sub_distmat_only(TrainIDs, ValiIDs, MLObjList, curDesc):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                               # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc] - MLObjList[curJID].Desc[curDesc])        # Calculation of normal distance matrix elements.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc] - MLObjList[curJID].Desc[curDesc])         # Calculation of normal distance matrix elements.
    return TrainDistMat, ValiDistMat


# Euclidean Norm variant
def otf_sub_eucdistmat_only(TrainIDs, ValiIDs, MLObjList, curDesc):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                               # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc], MLObjList[curJID].Desc[curDesc])    # Calculation of Euclidean Norm goes here.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc], MLObjList[curJID].Desc[curDesc])     # Calculation of Euclidean Norm goes here.
    return TrainDistMat, ValiDistMat


# Normal Distance variant - layered
def otf_sub_distmat_layer_only(TrainIDs, ValiIDs, MLObjList, curDesc, Dep):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                                         # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc][Dep] - MLObjList[curJID].Desc[curDesc][Dep])        # Calculation of normal distance matrix elements.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                          # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = abs(MLObjList[curIID].Desc[curDesc][Dep] - MLObjList[curJID].Desc[curDesc][Dep])         # Calculation of normal distance matrix elements.
    return TrainDistMat, ValiDistMat

# Euclidean Norm variant - layered
def otf_sub_eucdistmat_layer_only(TrainIDs, ValiIDs, MLObjList, curDesc, Dep):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            curIID = TrainIDs[ii]                                                                                         # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            TrainDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc][Dep], MLObjList[curJID].Desc[curDesc][Dep])    # Calculation of Euclidean Norm goes here.
    # Distance Matrix for Validation during Training
    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            curIID = ValiIDs[ii]                                                                                          # mapping to positions inside the Descriptor-Vector.
            curJID = TrainIDs[jj]                                                                                         # mapping to positions inside the Descriptor-Vector.
            ValiDistMat[ii,jj] = euc_dist(MLObjList[curIID].Desc[curDesc][Dep], MLObjList[curJID].Desc[curDesc][Dep])     # Calculation of Euclidean Norm goes here.
    return TrainDistMat, ValiDistMat

# Local Generation function of "Submatrices", using Training and Validation IDs. Formerly known as the "MatGen" functions.
# This one only for final training, where the sigma don't need to be determined any more.
# Used in DataTens Module.
def get_sub_mat_only(TrainIDs, ValiIDs, DistMat):
    TrainDistMat = np.zeros((len(TrainIDs), len(TrainIDs)))
    for ii in range(len(TrainIDs)):
        for jj in range(len(TrainIDs)):
            TrainDistMat[ii,jj] = DistMat[TrainIDs[ii], TrainIDs[jj]]

    ValiDistMat = np.zeros((len(ValiIDs), len(TrainIDs)))
    for ii in range(len(ValiIDs)):
        for jj in range(len(TrainIDs)):
            ValiDistMat[ii,jj] = DistMat[ValiIDs[ii], TrainIDs[jj]]
    return TrainDistMat, ValiDistMat

# Local Generation function of "Submatrices", using Training and Validation IDs. Formerly known as the "MatGen" functions.
# This one only gives the final Distance matrices for Predicitons
# Used in DataTens Module.
def get_sub_mat_pred(TrainIDs, PredIDs, DistMat):
    PredDistMat = np.zeros((len(PredIDs), len(TrainIDs)))
    for ii in range(len(PredIDs)):
        for jj in range(len(TrainIDs)):
            PredDistMat[ii,jj] = DistMat[PredIDs[ii], TrainIDs[jj]]
    return PredDistMat

# This function sums up Tensor Matrices scaled by one slice of Sigma values (hence SIGMA_VEC) to yield a Kernel-Matrix.
# Used in MLModel.
def kernel_gen(SIGMA_VEC, LocTensor):
    MLen = LocTensor[0][:, :].shape[0]
    NLen = LocTensor[0][:, :].shape[1]
    Aux  = np.zeros((MLen, NLen))
    for ii in range(len(SIGMA_VEC)):
        Aux += (1/(2*(SIGMA_VEC[ii]**2)))*(np.square(LocTensor[ii][:, :]))
    LocKMAT = np.exp(-Aux)
    return LocKMAT


# The following are on-the-fly-versions of calculating a single (layer of a) distance rank 3 tensor and then decorating it with its
# respective sigma value to yield the kernel matrix precursor.
# These are used in the krr_tens module.
def calc_pred_distmat(MLObjList, curDesc, TrainIDs, PredIDs, LocSigma):
    Aux = np.zeros((len(PredIDs), len(TrainIDs)))
    for ii in range(len(PredIDs)):
        for jj in range(len(TrainIDs)):
            curPID = PredIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curTID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            Aux[ii, jj] = abs(MLObjList[curPID].Desc[curDesc] - MLObjList[curTID].Desc[curDesc])                # Calculation of normal distance matrix elements.
    Out = (1/(2*(LocSigma**2)))*(np.square(Aux))                                                                # "embedding" with the respective sigma value.
    return Out

def calc_pred_eucdistmat(MLObjList, curDesc, TrainIDs, PredIDs, LocSigma):
    Aux = np.zeros((len(PredIDs), len(TrainIDs)))
    for ii in range(len(PredIDs)):
        for jj in range(len(TrainIDs)):
            curPID = PredIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curTID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            Aux[ii, jj] = euc_dist(MLObjList[curPID].Desc[curDesc], MLObjList[curTID].Desc[curDesc])            # Calculation of Euclidean Norm goes here.
    Out = (1/(2*(LocSigma**2)))*(np.square(Aux))                                                                # "embedding" with the respective sigma value.
    return Out

def calc_pred_distmat_layer(MLObjList, curDesc, Dep, TrainIDs, PredIDs, LocSigma):
    Aux = np.zeros((len(PredIDs), len(TrainIDs)))
    for ii in range(len(PredIDs)):
        for jj in range(len(TrainIDs)):
            curPID = PredIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curTID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            Aux[ii, jj] = abs(MLObjList[curPID].Desc[curDesc][Dep] - MLObjList[curTID].Desc[curDesc][Dep])      # Calculation of normal distance matrix elements.
    Out = (1/(2*(LocSigma**2)))*(np.square(Aux))                                                                # "embedding" with the respective sigma value.
    return Out

def calc_pred_eucdistmat_layer(MLObjList, curDesc, Dep, TrainIDs, PredIDs, LocSigma):
    Aux = np.zeros((len(PredIDs), len(TrainIDs)))
    for ii in range(len(PredIDs)):
        for jj in range(len(TrainIDs)):
            curPID = PredIDs[ii]                                                                                # mapping to positions inside the Descriptor-Vector.
            curTID = TrainIDs[jj]                                                                               # mapping to positions inside the Descriptor-Vector.
            Aux[ii, jj] = euc_dist(MLObjList[curPID].Desc[curDesc][Dep], MLObjList[curTID].Desc[curDesc][Dep])  # Calculation of Euclidean Norm goes here.
    Out = (1/(2*(LocSigma**2)))*(np.square(Aux))                                                                # "embedding" with the respective sigma value.
    return Out


# The following are on-the-fly-versions of calculating a single (layer of a) distance rank 3 tensor. Then determining the sigma on the fly and
# (using the nth_slice information and Min/MaxMods, on-the-fly generating the auxiliary matrix.
# respective sigma value to yield the kernel matrix precursor.
# These are used in the krr_tens module.
# These have to be used only for the Train x Train matrices - the other ones can be used from the ones above, after we get the local sigma values.
def otf_distmat(MLObjList, DataTensObj, curDesc, slice_n):
    LTrainIDs     = DataTensObj.CVTrainIDs
    LBinWinThrsh  = DataTensObj.BinWinThrsh
    LBinWinEscape = DataTensObj.BinWinEscape
    DMAT          = np.zeros((len(LTrainIDs), len(LTrainIDs)))
    for ii in range(len(LTrainIDs)):
        for jj in range(len(LTrainIDs)):
            curPID = LTrainIDs[ii]
            curTID = LTrainIDs[jj]
            DMAT[ii,jj] = abs(MLObjList[curPID].Desc[curDesc] - MLObjList[curTID].Desc[curDesc])
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(DMAT, LBinWinThrsh, LBinWinEscape)
    # Determine local Sigma based off the nth slice.
    Aux1 = abs(ReturnSig)*DataTensObj.MinMod
    Aux2 = abs(ReturnSig)*DataTensObj.MaxMod
    sig_vec  = np.linspace(Aux1, Aux2, num=DataTensObj.CVGridPts)
    LocSigma = sig_vec[slice_n]
    Out = (1/(2*(LocSigma**2)))*(np.square(DMAT))                                                                # "embedding" with the respective sigma value.
    return Out, LocSigma

def otf_eucdistmat(MLObjList, DataTensObj, curDesc, slice_n):
    LTrainIDs     = DataTensObj.CVTrainIDs
    LBinWinThrsh  = DataTensObj.BinWinThrsh
    LBinWinEscape = DataTensObj.BinWinEscape
    DMAT           = np.zeros((len(LTrainIDs), len(LTrainIDs)))
    for ii in range(len(LTrainIDs)):
        for jj in range(len(LTrainIDs)):
            curPID = LTrainIDs[ii]
            curTID = LTrainIDs[jj]
            DMAT[ii, jj] = euc_dist(MLObjList[curPID].Desc[curDesc], MLObjList[curTID].Desc[curDesc])            # Calculation of Euclidean Norm goes here.
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(DMAT, LBinWinThrsh, LBinWinEscape)
    # Determine local Sigma based off the nth slice.
    Aux1 = abs(ReturnSig)*DataTensObj.MinMod
    Aux2 = abs(ReturnSig)*DataTensObj.MaxMod
    sig_vec  = np.linspace(Aux1, Aux2, num=DataTensObj.CVGridPts)
    LocSigma = sig_vec[slice_n]
    Out = (1/(2*(LocSigma**2)))*(np.square(DMAT))                                                                # "embedding" with the respective sigma value.
    return Out, LocSigma

def otf_distmat_layer(MLObjList, DataTensObj, curDesc, Dep, slice_n):
    LTrainIDs     = DataTensObj.CVTrainIDs
    LBinWinThrsh  = DataTensObj.BinWinThrsh
    LBinWinEscape = DataTensObj.BinWinEscape
    DMAT           = np.zeros((len(LTrainIDs), len(LTrainIDs)))
    for ii in range(len(LTrainIDs)):
        for jj in range(len(LTrainIDs)):
            curPID = LTrainIDs[ii]
            curTID = LTrainIDs[jj]
            DMAT[ii, jj] = abs(MLObjList[curPID].Desc[curDesc][Dep] - MLObjList[curTID].Desc[curDesc][Dep])      # Calculation of normal distance matrix elements.
    # Flexible determination of SigGuess
    ReturnSig = guess_sigma(DMAT, LBinWinThrsh, LBinWinEscape)
    # Determine local Sigma based off the nth slice.
    Aux1 = abs(ReturnSig)*DataTensObj.MinMod
    Aux2 = abs(ReturnSig)*DataTensObj.MaxMod
    sig_vec  = np.linspace(Aux1, Aux2, num=DataTensObj.CVGridPts)
    LocSigma = sig_vec[slice_n]
    Out = (1/(2*(LocSigma**2)))*(np.square(DMAT))                                                                # "embedding" with the respective sigma value.
    return Out, LocSigma

def otf_eucdistmat_layer(MLObjList, DataTensObj, curDesc, Dep, slice_n):
    LTrainIDs     = DataTensObj.CVTrainIDs
    LBinWinThrsh  = DataTensObj.BinWinThrsh
    LBinWinEscape = DataTensObj.BinWinEscape
    DMAT           = np.zeros((len(LTrainIDs), len(LTrainIDs)))
    for ii in range(len(LTrainIDs)):
        for jj in range(len(LTrainIDs)):
            curPID = LTrainIDs[ii]
            curTID = LTrainIDs[jj]
            DMAT[ii, jj] = euc_dist(MLObjList[curPID].Desc[curDesc][Dep], MLObjList[curTID].Desc[curDesc][Dep])  # Calculation of Euclidean Norm goes here.
    # Flexible determination of SigGuess
    ReturnSig = GuessSigma(DMAT, LBinWinThrsh, LBinWinEscape)
    # Determine local Sigma based off the nth slice.
    Aux1 = abs(ReturnSig)*DataTensObj.MinMod
    Aux2 = abs(ReturnSig)*DataTensObj.MaxMod
    sig_vec  = np.linspace(Aux1, Aux2, num=DataTensObj.CVGridPts)
    LocSigma = sig_vec[slice_n]
    Out = (1/(2*(LocSigma**2)))*(np.square(DMAT))                                                                # "embedding" with the respective sigma value.
    return Out, LocSigma

