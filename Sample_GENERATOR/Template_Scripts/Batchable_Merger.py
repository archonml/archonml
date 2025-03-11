import os, sys, time
import numpy as np
from archonml.entries import MLEntry
from archonml.desctools import Descriptor
from archonml.krr_tens import DataTens
from archonml.krr_model import MLModel
from archonml.utils import timeconv
from archonml.common import PSEDICT, PSEIDICT
from copy import deepcopy as dc

# Load an existing Model and setup the MLOBJLIST as wells as the Descriptor instance.
MLEntry.get_db_specs()

# Load the Data to Predict.
LType     = "Prediction"
LLabelL   = [""]
LQC_Pack  = "g16"                                         # Enter Name of the Program package here (g16, orca)
LQC_low   = "PM6"                                         # Enter what QC method was used for the semi-empirical results (e.g. PM6 or PM3)
LQC_high  = "TD-DFT"                                      # Enter what QC method was used for the high-quality results (e.g. TD-DFT)
LDataL    = ["Geometry", "Mulliken_F", "SEmpOrbInfo_F"]   # Enter what raw data to use for generating the Descriptors.
                                                          # Should be the same as during Training!

PredLib   = "../BATCHES/XXXAXXX"                          # Enter file name of a Calculation Library to predict.
                                                          # (these files need the same format as a "SampleCalc" file,
                                                          #  as produced from the DBGenerator)

# Activate this section in addition, if the native file format of the predictions does not exist yet.
# Read the Prediction IDs and Fingerprints from File.
FID = open (PredLib, "r")                                      # Enter file name of a Calculation Library to predict.
FLoc = FID.readlines()
FID.close()
LID      = []
LFingerP = []
PrepList = []
for line in range(len(FLoc)):
 LID.append(FLoc[line].split()[0])
 Aux     = FLoc[line].split()[1]
 FingLen = len(Aux.split(","))
 LFing = []
 for jj in range(FingLen):
  LFing.append(int(Aux.split(",")[jj]))
 LFingerP.append(LFing)
 LObj = MLEntry(LID[-1], LType, LFingerP[-1], LQC_Pack, LQC_low, LQC_high, LDataL, LLabelL)
 PrepList.append(dc(LObj))
# Only needed once per "newly calculated entry": Merge Data into my native format
for ii in range(len(PrepList)):
 PrepList[ii].merge_data()
