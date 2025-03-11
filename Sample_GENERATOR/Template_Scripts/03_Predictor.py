import os, sys, time
import numpy as np
from archonml.entries import MLEntry
from archonml.desctools import Descriptor
from archonml.krr_tens import DataTens
from archonml.krr_model import MLModel
from archonml.utils import timeconv
from archonml.common import PSEDict, PSEiDict
from copy import deepcopy as dc

# Load an existing Model and setup the MLOBJLIST as wells as the Descriptor instance.
MLEntry.get_db_specs()
LLabelL   = [""]                                           # Enter Name of property to predict here.
KRR = MLModel(LLabelL)
MLOBJLIST, DescInst = KRR.get_model("")                    # Enter Name of the Model to load here.

# Load the Data to Predict. This assumes, that all outputs have been "merged" already.
LType     = "Prediction"
LQC_Pack  = ""                                            # Enter Name of the Program package here (g16, orca)
LQC_low   = ""                                            # Enter what QC method was used for the semi-empirical results (e.g. PM6 or PM3)
LQC_high  = "TD-DFT"                                      # Enter what QC method was used for the high-quality results (e.g. TD-DFT)
LDataL    = ["Geometry", "Mulliken_F", "SEmpOrbInfo_F"]   # Enter what raw data to use for generating the Descriptors.
                                                          # Should be the same as during Training!
PredLib   = ""                                            # Enter file name of a Calculation Library to predict.
OutFile   = "{}.Predictions".format(LLabelL[0])""         # Enter name of output file for the predicted data

# Read the Prediction IDs and Fingerprints from File. Append to the MLEntries.
FID = open(PredLib, "r")
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
    
# Get the MLEntry.Data from MergeFiles and immediately describe.
for ii in range(len(MLOBJLIST)):
    MLOBJLIST[ii].get_merged_data()
    DescInst.describe(MLOBJLIST[ii])

# Instantiate the DataTens Object to get the precalculated Distance matrices. For predictions, no splitting or stratifying needed, of course.
DataInst = DataTens(MLOBJLIST, mem_mode="low")
# Perform predictions from loaded model.
KRR.predict_from_loaded(DataInst, MLOBJLIST)
# Save Predictions to file.
FID = open(OutFile, "w")
cnt = 0
for ii in range(len(MLOBJLIST)):
    if MLOBJLIST[ii].Type == "Prediction":
        Aux = ""
        for jj in range(len(MLOBJLIST[ii].FingerP)):
            Aux  += "{},".format(MLOBJLIST[ii].FingerP[jj])
            LFing = Aux.rstrip(Aux[-1])
        SSS = "{}   {}   {: 12.9f}\n".format(MLOBJLIST[ii].ID, LFing, KRR.Predictions[cnt])
        FID.write(SSS)
        cnt += 1
FID.close()
