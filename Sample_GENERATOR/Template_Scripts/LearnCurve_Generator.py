import os, sys, time
from copy import deepcopy as dc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from archonml.entries import MLEntry
from archonml.desctools import Descriptor
from archonml.krr_tens import DataTens
from archonml.krr_model import MLModel
from archonml.utils import timeconv, procram

plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['figure.dpi'] = 200
np.set_printoptions(suppress=True)
MLEntry.get_db_specs()

LType         = "Training"
LQC_Pack      = ""                                                                                      # The software package that was used goes here.
LQC_low       = "PM6"                                                                                   # The quality of descriptor calculaitons goes here.
LQC_high      = "TD-DFT"                                                                                # The quality of the label data goes here. (unused)
LLabelL       = [""]                                                                                    # The name of the label to train agaionst goes here. Brackets are needed.
LDataL        = ["Geometry", "Mulliken_F", "SEmpOrbInfo_F"]                                             # The Keywords for what raw data to merge are needed here.
MLEntry.MOWin = 4                                                                                       # The definition for the MOWin size goes here.

LDescL    = ["SEmpOrbEnDiffs", "SEmpOccs", "SEmpVirs", "SEmpTups", "SEmpEigCoul", "SEmpOccEigCoul",     # The keywords for the desired descriptors to use goes here.
             "SEmpNEl", "SEmpOccPCMEigCoul", "SEmpVirPCMEigCoul", "SEmpTransPCMEigCoul",
             "SEmpOccVirPTransSum", "SEmpHOLUPDiff"]
DescInst  = Descriptor(LDescL)
MLOBJLIST = []

FID = open("", "r")                                                                                     # The name of the Sample-Calculations file for the training set goes here.
FLoc = FID.readlines()
FID.close()
LID      = []
LFingerP = []
for line in range(len(FLoc)):
    LID.append(FLoc[line].split()[0])
    Aux     = FLoc[line].split()[1]
    FingLen = len(Aux.split(","))
    LFing = []
    for jj in range(FingLen):
        LFing.append(int(Aux.split(",")[jj]))
    LFingerP.append(LFing)
    LObj = MLEntry(LID[-1], LType, LFingerP[-1], LQC_Pack, LQC_low, LQC_high, LDataL, LLabelL)
    MLOBJLIST.append(dc(LObj))

######################################################################################################
######################################################################################################
######################################################################################################

# CV Step Automization
STEP = [0, 1, 2, 3, 4]                                                                                   # Counters for the CV steps. _Must_ be set up with respect to your CVSize.
VAS  = [0, 1, 2, 3, 4]                                                                                   # Indices for which validation block to use as local testing set at each CV step.
TRS  = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]                            # The remaining validation blocks to put in local training sets at each CV step.
                                                                                                         # These _Must_ exclude the testing block of the respective CV step.
                                                                                                         # (i.e. "0" is missing in the first set, as it is the VAS[0])

PRC  = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]                                          # The percentages to use go here.
PRCS = ["100", "090", "080", "070", "060", "050", "040", "030", "020", "010", "005"]                     # These are the same percentages in a different, string format.
                                                                                                         # They will be added to output files of the learncurve generator.

# Initialize the DataTens Object. Below are the default options for data management.
DataTens.CVGridPts    = 64
DataTens.CVSize       = 5
DataTens.RandomStrat  = True
DataTens.FixedSeed    = False          # Note that FixedSeed should be deavtivated for LearnCurves!!!
DataTens.MinMod       = 0.01           
DataTens.MaxMod       = 10             
DataTens.Lambda_Bot   = -6
DataTens.Lambda_Top   =  2
DataTens.BinWinThrsh  = 10
DataTens.BinWinEscape = 100

# At initialization, it will precalculate all distance matrices.
for JJ in range(len(PRC)):
    DataInst = DataTens(MLOBJLIST, mem_mode="high")
    DataInst.train_test_split(0.8)
    DataInst.train_test_split(0.8, UPDATE=True, ActPerc=PRC[JJ])
    DataInst.stratify(TARGET)
    KRR = MLModel(TARGET, "R2")
    for II in STEP:
        CV_VaIDs = DataInst.CVIDBlocks[VAS[II]]
        CV_TrIDs = DataInst.CVIDBlocks[TRS[II][0]] + DataInst.CVIDBlocks[TRS[II][1]] + DataInst.CVIDBlocks[TRS[II][2]] + DataInst.CVIDBlocks[TRS[II][3]]
        DataInst.update_cv_tens(CV_TrIDs, CV_VaIDs, MLOBJLIST)
        KRR.cv_para(DataInst, MLOBJLIST, optimizer="vectorized", useGUI=False)
        # RENAME FILES
        curNO = "CVStats_{}.out".format(len(KRR.RCurLambdas))
        curNS = "CVStats_{}.svg".format(len(KRR.RCurLambdas))
        newNO = "CVStats_C{}_P{}_S{}.out".format(CNum, PRCS[JJ], II)
        newNS = "CVStats_C{}_P{}_S{}.svg".format(CNum, PRCS[JJ], II)
        os.rename(curNO, newNO)
        os.rename(curNS, newNS)
    # FINALIZE MODEL
    KRR.final_cv(DataInst, MLOBJLIST, useGUI=False)
    # RENAME FILES
    curO = "CVFinal.out"
    curS = "CVFinal.svg"
    newO = "CVFinal_C{}_P{}.out".format(CNum, PRCS[JJ])
    newS = "CVFinal_C{}_P{}.svg".format(CNum, PRCS[JJ])
    os.rename(curO, newO)
    os.rename(curS, newS)
    # Save the 100 % Model
    if JJ == 0:
        KRR.save_model("{}_C{}".format(TARGET, CNum))

    # Output the predicted values and label values - for post processing analysis stuff.
    OID = open("CVFinStats_C{}_P{}.out".format(CNum, PRCS[JJ]), "w")
    OID.write("# Num        Label            PredVal_M        PredVal_R\n")
    for ii in range(len(KRR.FinTestLabelVals)):
        OID.write("{:5d}       {: 3.7f}       {: 3.7f}       {: 3.7f}\n".format(ii, KRR.FinTestLabelVals[ii], KRR.FinPredMVals[ii], KRR.FinPredRVals[ii]))
    OID.close()
