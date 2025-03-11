# # # # # # # # # # # # #
# This is the Module containing the MLEntry class and some functions used within.
# The MLEntry class instances individual ML entries with their respective raw data and meta data (original QC package that generated the data).
# It is used by all other classes as an intermediary to calculate features from the raw data and assign data types (Prediction, Training).
#
# Its class attributes define the database architecture in a tree-like subfolder format and further define paths to the Calculation Folders
# and the MergeFiles (i.e. ArchOnML's native format of raw data).
#
# FW 07/13/23

import re
from copy import deepcopy as dc

import numpy as np

from .common import PSEDICT, PSEIDICT, EV


class MLEntry:
    """
    An object class that stores and transports all relevant data for one database entry.

    The MLEntry class is carries all useful data for one molecule/conformation of
    your database. It interacts with both the Descriptor class (which calculates the
    entry's descriptors), as well as the Data Tensor class (which calculates and
    manages the distance matrices for kernel ridge regression).

    --- Attributes ---
    ------------------
    Fold_L1 : int
        An integer describing how the database is structured. Used to (re-)create the
        paths to the individual calculations based off the fingerprint.
        "First level" of the path string.

    Fold_L2 : int
        An integer describing how the database is structured. Used to (re-)create the
        paths to the individual calculations based off the fingerprint.
        "Second level" of the path string.

    MergePath : str
        A string that stores where the merged files (i.e. the native format of raw
        data) for all entries are stored. Exact string is re-recreated from Fingerprint.

    DataPath : str
        A string that stores where the database files (i.e. the quantum-chemistry calculations)
        are found. Exact string is re-created from Fingerprint.

    SpaceTyp : str
        A string that affects, whether a chemical space screen ("Chem") or a conformational
        space screen ("Conf") is to be performed.

    ID : str
        A string that stores the unique name of this database entry. The name is composed of
        the project name (i.e. "SUCROSE") plus a running integer number (i.e. "SUCROSE124").

    Type : str
        A string that encodes whether this entry is considered "Training" data or "Prediction" data.
        Depending on the type, the entry will be treated differently by the Data Tensor object.

    FingerP : array of float
        An array of floats that encodes the "chemistry" of the entry. In case of a chemical space screen,
        the i-th float corresponds to the n-th substitution fragment as given in the substituents library.

    Path : str
        A string that stores the full path to this entry's stem folder - i.e. where the calculations are
        performed.

    QC_Pack : str
        A string that stores which program package was used to perform the external calculations.

    QC_low : str
        A string storing which level of theory was used for obtaining the raw data - which is to
        be turned into descriptors eventually.

    QC_high : str
        A string that stores which level of theory was used for obtaining the label data. Currently
        unused, but may be used for further automization in the future.

    DataL : list of str
        A list of strings that encode which raw data is supposed to be taken from the QC calculations.
        These will determine, which Descriptors are able to be built.

    DescL : list of str
        A list of strings that request specific descriptors to be built from raw data. Names here must match
        the names inside the desctool module.

    Data  : dict w.r.t DataL strings
        A dictionary that stores the raw data. The keys for the data match the names given in the DataL.

    Desc  : dict w.r.t. DescL strings
        A dictionary that stores the descriptors. This is filled by an instance of the Describe class.
        the keys for the data match the names given in the DescL.

    DescType : dict w.r.t. DescL strings
        A dictionary for the type (and operation) for each descriptor. First entry reads (either "unused" or) the
        data type for the current descriptor. Second entry (if not "unused") reads, what operation the Data Tensor
        class' instance should use when trying to build a distance matrix for this descriptor.

    LabelL : list of str
        A list of strings containing all loaded labels. Note that the names given here are expected to be the same
        of the files that contain the label data. Label data files are expected to be found in the MergePath location
        by default.

    Label : dict w.r.t. LabelL strings
        A dictionary that contains the label values for this entry. Keys match the names provided in the LabelL list.

    LabelType : dict w.r.t. LabelL strings
        A dictionary that describes the data type of the labels. Currently unused, but may be used for predicting
        more complex shaped properties in the future.

    NAt : int
        An integer that carries the number of atoms for this entry. Useful for many situations outside of descriptors
        and thus received its own attribute.

    GeomH : matrix of float
        A matrix object that contains the entry's geometry (including H atoms) - plus additional data. Each row
        carries information on one atom - reading
        [0] x-position, [1] y-positon, [2], z-position, [3] atomic number, [4] placeholder for mulliken charge
        [5] wildcard placeholder for different descriptors.

    --- Methods ---
    ---------------
    get_db_specs(cls)
        A class method that will read in an existing DBSpecs.cfg file to initialize the relevant background
        information for this database.
        (returns None)

    merge_data(self)
        A method that will store all raw data in ArchOnML's native format inside the respective MergePath.
        The data stored depends on what DataL were set earlier.
        (returns None)

    get_merged_data(self)
        A method that will fetch the raw data for this entry from the MergePath.
        (returns None)
    """
    # Initialize Database specific, global attributes. The values are defaults but can be read from the DBSpecs.cfg file as well. 
    Fold_L1   = 1                                   # Folder Levels for first layer in Database. This is an INTEGER. Strings are constructed on the fly, later.
    Fold_L2   = 3                                   # Folder Levels for second layer in Database. This is an INTEGER. Strings are constructed on the fly, later.
    MergePath = "../SemiEmpData/"                   # MergePath for the merged files.
    DataPath  = "../../DATABASE/CHLA/"              # Path of the Calculation Folders.
    
    MOWin     = 10                                  # Length of the MO window for I/O purposes. This may be longer than the actual later used window.
                                                    # (This is useless, in case of reduced I/O, where the window is predefined elsewhere)
    @classmethod
    def get_db_specs(cls):
        """
        A class method that will read in an existing DBSpecs.cfg file to initialize the relevant background
        information for this database.
        (returns None)
        """
        # Read from the project-specific DBSpecs.cfg file, generated during DB generation.
        FID  = open("./DBSpecs.cfg", "r")
        FLoc = FID.readlines()
        FID.close()
        MLEntry.Fold_L1   = int(FLoc[1].split()[0])    # Folder Levels for first layer in Database. This is an INTEGER. Strings are constructed on the fly, later.
        MLEntry.Fold_L2   = int(FLoc[2].split()[0])    # Folder Levels for second layer in Database. This is an INTEGER. Strings are constructed on the fly, later.
        MLEntry.MergePath = FLoc[3].split()[0]         # MergePath for the merged files.
        MLEntry.DataPath  = FLoc[4].split()[0]         # Path of the Calculation Folders.
        MLEntry.SpaceTyp  = FLoc[5].split()[0]         # Whether this is a chemical or conformer space search.
        return None

    # Initialize object-specific attributes.
    def __init__(self, iID, iType, iFingerP, iQC_Pack, iQC_low, iQC_high, iDataL, iLabelL):
        """
        --- Parameters ---
        ------------------
        iID : str
            The unique name of the current database entry.

        iType : str
            The type of data entry - i.e. whether this is "Training" or "Prediction" data.

        iFingerP : array of floats
            The unique chemical encoding for this database entry. In case of a chemical space screen,
            this code gives each substituent position's currently assigned substituent ID.
            In case of a conformational space screen, the first and second entry reflect the folder
            hierarchy, while the third refers the the n-th structure as given in the conformation
            file.

        iQC_Pack : str
            The keyword for a specific quantum-chemistry package used in the calculations.

        iQC_high : str
            A keyword for a specific level of theory used in obtaining the label values. Currently unused.

        iDataL : list of str
            A list of requested raw data that is to be collected when merging the data. Only requested data
            will be accessible to later building the descriptor values.

        iLabelL : list of str
            A list of attachable label values. When loading the entry, ArchOnML will look for a file that has
            the exact name provided here to read this ID's label value from. The file should (line by line)
            contain one of the unique IDs, as well as the designated label value. Ordering does not matter.
        """
        # Data-Storage relevant Object information
        self.ID          = iID                         # Internal, unique instance ID used in library.
        self.Type        = iType                       # Type of Entry. Discern between "Training" and "Prediction". Note that Testing is a subcategory of Training.
                
        self.FingerP     = iFingerP                    # Fingerprint of this ID. Stored as an array of integers.

        # Construct the Path to the Calculation
        if self.SpaceTyp == "Chem":
            Aux = ""
            for ii in range(MLEntry.Fold_L1):
                Aux += "{}_".format(str(self.FingerP[ii]))
            L1_Path = Aux.rstrip(Aux[-1])                  # remove trailing underscore
            Aux = ""
            for ii in range(MLEntry.Fold_L2):
                Aux += "{}_".format(str(self.FingerP[ii]))
            L2_Path = Aux.rstrip(Aux[-1])                  # remove trailing underscore

        if self.SpaceTyp == "Conf":
            L1_Path = "{}".format(str(self.FingerP[0]))
            L2_Path = "{}_{}".format(str(self.FingerP[0]), str(self.FingerP[1]))

        self.Path = MLEntry.DataPath + "/" + L1_Path + "/" + L2_Path + "/" + self.ID + "/"

        # Parser relevant Object information. Storing this individually allows for constructing mixed Databases.
        self.QC_Pack     = iQC_Pack                    # What code was used to generate data. Will impact the Parser.
        self.QC_low      = iQC_low                     # Method used to generate low-accuracy QC data. This will determine how to Parse the data.
        self.QC_high     = iQC_high                    # Method used to generate high-accuracy ML data. This will determine how to Parse the data - and IS the suffix of the folders.
  
        self.DataL       = iDataL                      # LIST of Data types read from the low-accuracy QC data.
                                                       # Note, that Data may have a _F_ull I/O and a _R_educed I/O version.
                                                       # Full I/O will read directly from the original QC output files.
                                                       # Reduced I/O will read from an already extracted, shortened file (useful for saving disk space).

        # Initialized empty - filled somewhere else.
        self.DescL       = []                          # LIST of Descriptors constructed from the Data. This will be filled up by a Descriptor object.
        self.Data        = {}                          # Dictionary of raw Merge Data.
        self.Desc        = {}                          # Dictionary of low-level descriptors.
        self.DescType    = {}                          # Instructions on how to deal with a given descriptor (Euclidean distances; direct comparisons, data format).
                                                       # This is practically how to set up the individual similarity functions for the exponential, later.
      
        self.LabelL      = iLabelL                     # LIST of labels read from the high-accuracy ML data. This is forwarded during initialization.
        self.Label       = {}                          # Dictionary of high-level Labels. If Training Data, this will be filled with the real data during the Descriptor,
                                                       # if Prediction data, this will be filled out in the Predictor.
      
        self.LabelType   = {}                          # Dictionary of Label Types and formats. This is widely unused still, as the module still mostly covers individual predicted values
                                                       # instead of fitting multiple Labels at the same time.
        
        self.NAt         = []                          # Useful recurring integer for number of atoms. Set during the reading of Merge File.


    # This function will Merge the data for this entry in a specified folder and format.
    # This will not need to store anything into the MLEntry object. Reading should always be done from merged files, afterwards.
    def merge_data(self):
        """
        merge_data(self)
            A method that will store all raw data in ArchOnML's native format inside the respective MergePath.
            The data stored depends on what DataL were set earlier.
            (returns None)

        """
        # Depending on which program and method was used to generate data, build a specific parser.
        LocGeom = []
        LocMull = []
        if self.QC_Pack == "g16":       # Gaussian Merger as Package.
            # Import the CTypes and FTypes for the gaussian package to flexibly create the required name paths.
            from archonml.calcgen.gaussian import CType, FType
            
            # Read the Geometry of the PreOpt calculation - i.e. Folder Structure "A".
            if "Geometry" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["PreOpt"]["FolName"].format(self.QC_low)
                IName   = "PreOpted.xyz"
                IOFile  = Aux.format(FName, IName)
            
                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
                LocNAt  = int(FLoc[0].split()[0])
                LocGeom = np.zeros((LocNAt, 4))
                Loccnt  = 0
                for line in range(2, len(FLoc)):
                    LocGeom[Loccnt, 0] = float(FLoc[line].split()[1])
                    LocGeom[Loccnt, 1] = float(FLoc[line].split()[2])
                    LocGeom[Loccnt, 2] = float(FLoc[line].split()[3])
                    LocGeom[Loccnt, 3] = PSEDICT[FLoc[line].split()[0]]
                    Loccnt += 1
                if not self.NAt:
                    self.NAt = LocNAt

            # Read Mulliken Charges from an OrbEns output - i.e. Folder Structure "N".
            if "Mulliken_F" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = CType["OrbEns"]["OutName"]
                IOFile  = Aux.format(FName, IName)
            
                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
                line = 0
                while "Mulliken charges:" not in FLoc[line]:
                    line += 1
                line += 2
                ReadL = dc(line)
                while "Sum of Mulliken charges" not in FLoc[line]:
                    line += 1
                StopL   = dc(line)
                NLine   = StopL - ReadL
                LocMull = np.zeros((NLine))
                buffer  = ReadL
                for ii in range(NLine):
                    LocMull[ii] = float(FLoc[buffer+ii].split()[-1])

            # Read Mulliken Charges from REDUCED File "Mulliken" - i.e. Folder Structure "N".
            if "Mulliken_R" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = "Mulliken"
                IOFile  = Aux.format(FName, IName)
            
                FID    = open(self.Path+IOFile, "r")
                FLoc   = FID.readlines()
                FID.close()
                NLine = len(FLoc)
                LocMull = np.zeros((NLine))
                for ii in range(NLine):
                    LocMull[ii] = float(FLoc[ii].split()[-1])

            # Read the MO Info from the OrbEns output - i.e. Folder Structure "N".
            if "SEmpOrbInfo_F" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = CType["OrbEns"]["OutName"]
                IOFile  = Aux.format(FName, IName)
            
                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
              
                LCAOOut   = []
                OrbEnsOut = []
                # Get the number of occupied MOs
                line = 0
                while "alpha electrons" not in FLoc[line]:
                    line += 1
                LCAOOut.append(FLoc[line])                                         # Stores NOcc
                LocNOcc = int(FLoc[line].split()[0])
                # Get the SEmp SCF Energy
                while "SCF Done" not in FLoc[line]:
                    line += 1
                LocSEmpEn = float((FLoc[line].split("=")[1]).split("A.U.")[0])
                LCAOOut.append("{}\n".format(LocSEmpEn))                           # Stores SEmp Energy
                # Get the Occupied and Virtual Orbital Energies
                line = len(FLoc)-1
                while "Population analysis using the SCF Density." not in FLoc[line]:
                    line -= 1
                line += 4
                StartLine = 0 + dc(line)
                LocOCCEn  = []
                LocVIRTEn = []
                while "eigenvalues" in FLoc[line]:
                    if "occ" in FLoc[line]:
                        PreOC = FLoc[line].split("Alpha  occ. eigenvalues --")[1].replace("-", " -")
                        OC    = PreOC.split()
                        OCLen = len(OC)
                        for ii in range(OCLen):
                            LocOCCEn.append(float(OC[ii]))
                    if "virt" in FLoc[line]:
                        VR    = FLoc[line].split("Alpha virt. eigenvalues --")[1].split()
                        VRLen = len(VR)
                        for ii in range(VRLen):
                            LocVIRTEn.append(float(VR[ii]))
                    line += 1
                # Construct the Merged Format output
                SSS = "{} {: 12.9f}\n"
                for ii in range(len(LocOCCEn)):
                    LocID = -int(len(LocOCCEn)) + ii + 1
                    OrbEnsOut.append(SSS.format(LocID, LocOCCEn[ii]))
                for ii in range(len(LocVIRTEn)):
                    LocID = ii
                    OrbEnsOut.append(SSS.format(LocID, LocVIRTEn[ii]))
                # Get the LCAO-MO coefficients.
                while "Molecular Orbital Coefficients:" not in FLoc[line]:
                    line += 1
                LCAOOut.append(FLoc[line])
                line += 1
                StartLine = dc(line)
                # reconstruct the overall number of NAOs, MOs and blocks
                NLine = 3
                line += 3
                while "Eigenvalues" not in FLoc[line]:
                    NLine += 1
                    line  += 1
                NLine -= 2
                LocNAOs    = NLine - 3
                NBlocks = int((LocNAOs - (LocNAOs % 5)) / 5)
                if (LocNAOs % 5) > 0:
                    NBlocks += 1
                # put buffer to the first line again
                line = dc(StartLine)
                # Generate, which lines contain the valuable orbital information.
                for jj in range(NBlocks):
                    Nu   = 5*jj                                # Block Number
                    Theo = [1+Nu, 2+Nu, 3+Nu, 4+Nu, 5+Nu]      # theoretical block numbers in these lines
                    if (LocNOcc-MLEntry.MOWin) in Theo:
                        TLinSS = StartLine + NLine * jj
                    if (LocNOcc+MLEntry.MOWin) in Theo:
                        TLinES = StartLine + NLine * jj
                        TLinE  = TLinES + NLine
                for PLine in range(TLinSS, TLinE):
                    LCAOOut.append(FLoc[PLine])

            # the MO Info from the reduced output files "OrbEns" and "OrbInfo" - i.e. Folder Structure "N".
            if "SEmpOrbInfo_R" in self.DataL:
                # Read in the "OrbEns" file.
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = "OrbEns"
                IOFile  = Aux.format(FName, IName)

                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
                OrbEnsOut = []
                for line in range(len(FLoc)):
                    OrbEnsOut.append(FLoc[line])
                # Read in the OrbInfo file
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = "OrbInfo"
                IOFile  = Aux.format(FName, IName)

                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
                LCAOOut = []
                for line in range(len(FLoc)):
                    LCAOOut.append(FLoc[line])
            
        if self.QC_Pack == "orca":       # Gaussian Merger as Package.
            # Import the CTypes and FTypes for the gaussian package to flexibly create the required name paths.
            from archonml.calcgen.orca import CType, FType

            # Read the Geometry of the PreOpt calculation - i.e. Folder Structure "A".
            if "Geometry" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["PreOpt"]["FolName"].format(self.QC_low)
                IName   = "PreOpted.xyz"
                IOFile  = Aux.format(FName, IName)

                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
                LocNAt  = int(FLoc[0].split()[0])
                LocGeom = np.zeros((LocNAt, 4))
                Loccnt  = 0
                for line in range(2, len(FLoc)):
                    LocGeom[Loccnt, 0] = float(FLoc[line].split()[1])
                    LocGeom[Loccnt, 1] = float(FLoc[line].split()[2])
                    LocGeom[Loccnt, 2] = float(FLoc[line].split()[3])
                    LocGeom[Loccnt, 3] = PSEDICT[FLoc[line].split()[0]]
                    Loccnt += 1
                if not self.NAt:
                    self.NAt = LocNAt

            # Read Mulliken Charges from an OrbEns output - i.e. Folder Structure "N".
            if "Mulliken_F" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = CType["OrbEns"]["OutName"]
                IOFile  = Aux.format(FName, IName)

                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()
                line = 0
                while "MULLIKEN ATOMIC CHARGE" not in FLoc[line]:
                    line += 1
                line += 2
                ReadL = dc(line)
                while "Sum of atomic charges:" not in FLoc[line]:
                    line += 1
                StopL   = dc(line)
                NLine   = StopL - ReadL
                LocMull = np.zeros((NLine))
                buffer  = ReadL
                for ii in range(NLine):
                    LocMull[ii] = float(FLoc[buffer+ii].split()[-1])

            # Read Mulliken Charges from REDUCED File "Mulliken" - i.e. Folder Structure "N".
            if "Mulliken_R" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = "Mulliken"
                IOFile  = Aux.format(FName, IName)

                FID    = open(self.Path+IOFile, "r")
                FLoc   = FID.readlines()
                FID.close()
                NLine = len(FLoc)
                LocMull = np.zeros((NLine))
                for ii in range(NLine):
                    LocMull[ii] = float(FLoc[ii].split()[-1])

            # Read the MO Info from the OrbEns output - i.e. Folder Structure "N".
            if "SEmpOrbInfo_F" in self.DataL:
                Aux     = "/{}/{}"
                FName   = CType["OrbEns"]["FolName"].format(self.QC_low)
                IName   = CType["OrbEns"]["OutName"]
                IOFile  = Aux.format(FName, IName)

                FID     = open(self.Path+IOFile, "r")
                FLoc    = FID.readlines()
                FID.close()

                LCAOOut   = []
                OrbEnsOut = []
                # Get the number of occupied MOs. Assume that this is NEL/2 in all N calculations.
                line = 0
                while "Number of Electrons" not in FLoc[line]:
                    line += 1

                LocNOcc = int(FLoc[line].split()[-1])/2
                SSS = "    {} alpha electrons       {} beta electrons\n".format(int(LocNOcc), int(LocNOcc))
                LCAOOut.append(SSS)                                         # Stores NOcc

                # Get the SEmp SCF Energy
                while "Total Energy" not in FLoc[line]:
                    line += 1
                LocSEmpEn = float((FLoc[line].split(":")[1]).split("Eh")[0])
                LCAOOut.append("{}\n".format(LocSEmpEn))                           # Stores SEmp Energy
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
                    OrbEnsOut.append(SSS.format(LocID, LocOCCEn[ii]))
                for ii in range(len(LocVIRTEn)):
                    LocID = ii
                    OrbEnsOut.append(SSS.format(LocID, LocVIRTEn[ii]))
                # Get the LCAO-MO coefficients.
                while "MOLECULAR ORBITALS" not in FLoc[line]:
                    line += 1
                LCAOOut.append("Molecular Orbital Coefficients:\n")

                # Here begins the actual translation part.
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
                    if (LocNOcc-MLEntry.MOWin) in Theo:
                        TLinSS = StartLine + NLine * jj
                    if (LocNOcc+MLEntry.MOWin) in Theo:
                        TLinES = StartLine + NLine * jj
                        TLinE  = TLinES + NLine

                # Read and store EACH PIECE of information between these lines.
                # Read MOIDs, MOENs, MOStats from only the "headline sections" of the relevant blocks.
                MOID       = []
                MOEN       = []
                MOStat     = []
                buffer     = TLinSS
                while buffer < TLinE:
                    Aux = FLoc[buffer].split()
                    for ii in range(len(Aux)):
                        MOID.append(Aux[ii])
                    Aux = FLoc[buffer+1].split()
                    for ii in range(len(Aux)):
                        MOEN.append(float(Aux[ii]))
                    Aux = FLoc[buffer+2].split()
                    for ii in range(len(Aux)):
                        if Aux[ii][0] == "2":
                            MOStat.append("O")
                        if Aux[ii][0] == "0":
                            MOStat.append("V")
                    buffer += NLine

                # Read NAOIDs, NAOAtIDs, NAOAtTyps, NAOTyps - this only needs to be done along one block.
                AOID      = []
                AOAtID    = []
                AOAtTyp   = []
                AOPrincip = []
                AOAzimuth = []
                buffer  = TLinSS
                buffer += 4
                regroups = re.compile("([0-9]+)([a-zA-Z]+)")
                for ii in range(LocNAOs):
                    # Just save NAOID as ii + 1
                    AOID.append(ii+1)
                    # Split the first string block into integer numbers and letters and store the atom ID and type.
                    # During writing later, check if new atom each line.
                    Aux      = FLoc[buffer].split()[0]
                    Tmp      = regroups.match(Aux).groups()
                    Aux1     = int(Tmp[0])+1
                    Aux2     = Tmp[1]
                    AOAtID.append(Aux1)
                    AOAtTyp.append(Aux2)
                    # Split the second string block into integer numbers and letters. Store principal quantum number and "azimuthal information"
                    Aux      = FLoc[buffer].split()[1]
                    Tmp      = regroups.match(Aux).groups()
                    Aux1     = int(Tmp[0])
                    Aux2     = Tmp[1]
                    AOPrincip.append(Aux1)
                    AOAzimuth.append(Aux2.upper())
                    buffer += 1

                # Read all MO-AO coefficients. This has to be done M times recursively, block-wise.
                MOAOCoeff = []
                buffer    = TLinSS
                while buffer < TLinE:
                    # Split how many orbitals there are in this block.
                    MLen = len(FLoc[buffer].split())
                    Anchor = dc(buffer)
                    # start reading for one M in this block.
                    for ii in range(MLen):
                        # Reset buffer position and empty Aux.
                        Aux      = []
                        buffer   = dc(Anchor)
                        curSplit = ii + 2
                        buffer  += 4
                        for jj in range(LocNAOs):
                            Aux.append(float(FLoc[buffer+jj].split()[curSplit]))
                        MOAOCoeff.append(Aux)
                    # After all M MOs have been read this way, jump to the next block.
                    buffer = Anchor + NLine
      
                # Finally, turn this into the native output format and write.
                # Needs to be Column-Wise, recursive writing as well, since number of columns may not always be divisible by 5...
                # NOTE - for now, use a STATIC indentation scheme.
                MOsDone      = 0
                MOsThisBlock = 0
                NewBlock     = True
                NewMO        = True
                AuxLines     = []                          # This will store on the fly.
                                                           # need to "append from left to right", when adding an MO block.
                                                           # NOTE - the "\n" need to be added LAST for each line.
     
                BlockLines   = []
                while MOsDone != len(MOID):
                    # Finalize this block
                    if MOsThisBlock == 5:
                        for ii in range(len(AuxLines)):
                            AuxLines[ii] += "\n"
                            BlockLines.append(AuxLines[ii])
                        AuxLines = []
                        MOsThisBlock = 0
                        NewBlock = True
                    # Start a new Block!
                    if NewBlock:
                        # Initialize the header section.
                        AuxLines.append("                       ")
                        AuxLines.append("                       ")
                        AuxLines.append("     Eigenvalues --    ")
                        # Write general AO information.
                        LastAtomID = 0
                        NewAtom    = False
                        for ii in range(LocNAOs):
                            curAOID      = AOID[ii]
                            curAOAtID    = AOAtID[ii]
                            curAOAtTyp   = AOAtTyp[ii]
                            curAOPrincip = AOPrincip[ii]
                            curAOAzimuth = AOAzimuth[ii]
                            # Check if this atom has been there before. If not, initialize new atom and update LastAtomID.
                            if curAOAtID != LastAtomID:
                                NewAtom      = True
                                LastAtomID   = dc(curAOAtID)
                            # Continue at the current atom.
                            if not NewAtom:
                                SSS = "{:>4}        {:1}{:10}".format(curAOID, curAOPrincip, curAOAzimuth)
                            # Initialize a new atom and set NewAtom to false immediately.
                            if NewAtom:
                                SSS = "{:>4} {:3} {:2} {:1}{:10}".format(curAOID, curAOAtID, curAOAtTyp, curAOPrincip, curAOAzimuth)
                                NewAtom = False
                            AuxLines.append(SSS)
                        NewBlock = False

                    # Write a new MO!
                    if not NewBlock:
                        if MOsThisBlock < 5:
                            AuxLines[0] += "{:>5}     ".format(str(int(MOID[MOsDone])+1))    # orca counts MO indices from 0! -> add +1
                            AuxLines[1] += "{:>5}     ".format(str(MOStat[MOsDone]))
                            AuxLines[2] += "{: 8.5f}  ".format(float(MOEN[MOsDone]))
                            for ii in range(LocNAOs):
                                AuxLines[3 + ii] += "{: 8.5f}  ".format(float(MOAOCoeff[MOsDone][ii]))
                            MOsDone += 1
                            MOsThisBlock += 1

                # If AuxLines is not empty, finalize and flush remaining lines.
                if AuxLines != []:
                    for ii in range(len(AuxLines)):
                        AuxLines[ii] += "\n"
                        BlockLines.append(AuxLines[ii])
                    AuxLines = []
     
                for ii in range(len(BlockLines)):
                    LCAOOut.append(BlockLines[ii])
            
        # Dump the Merged Data File.
        # Construct the file path from ID, Layers, Fingerprint and MergePath.
        if self.SpaceTyp == "Chem":
            Aux = ""
            for ii in range(MLEntry.Fold_L1):
                Aux += "{}_".format(str(self.FingerP[ii]))
            L1_Path = Aux.rstrip(Aux[-1])                           # remove last underscore
            Aux = ""
            for ii in range(MLEntry.Fold_L2):
                Aux += "{}_".format(str(self.FingerP[ii]))
            L2_Path = Aux.rstrip(Aux[-1])                           # remove last underscore

        if self.SpaceTyp == "Conf":
            L1_Path = "{}".format(str(self.FingerP[0]))
            L2_Path = "{}_{}".format(str(self.FingerP[0]), str(self.FingerP[1]))

        IOFile  = MLEntry.MergePath + "/" + L1_Path + "/" + L2_Path + "/" + self.ID + ".dat"
        FID     = open(IOFile, "w")
        # Write only Geometry.
        if (LocGeom.size != 0) and (LocMull.size == 0):
            FID.write("{}\n\n".format(self.NAt))
            for ii in range(self.NAt):
                SSS = "{:2}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}\n"
                FID.write(SSS.format(PSEIDICT[LocGeom[ii, 3]], LocGeom[ii, 0], LocGeom[ii, 1], LocGeom[ii, 2], 0.0, 0.0))
        # Write Geometry and Mulliken Charges.
        if (LocGeom.size != 0) and (LocMull.size != 0):
            FID.write("{}\n\n".format(self.NAt))
            for ii in range(self.NAt):
                SSS = "{:2}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}     {: 12.9f}\n"
                FID.write(SSS.format(PSEIDICT[LocGeom[ii, 3]], LocGeom[ii, 0], LocGeom[ii, 1], LocGeom[ii, 2], LocMull[ii], 0.0))
        # Write the SEmp Orbital Energies.
        if (OrbEnsOut != []):
            for ii in range(len(OrbEnsOut)):
             FID.write(OrbEnsOut[ii])
        # Write the LCAO-MO section.
        if (LCAOOut != []):
            for ii in range(len(LCAOOut)):
                FID.write(LCAOOut[ii])
        FID.close()
        return None


    # This function will get the data from the Merged File in the specified folder.
    # Depending on the Descriptor List (DescL), this function initializes the data for the "Describe" function.
    # Note that this is now the "abstract" format that may have been produced from any QC code.
    # Also, the Describe function should "flush" this raw data from memory, later.
    def get_merged_data(self):
        """
        get_merged_data(self)
            A method that will fetch the raw data for this entry from the MergePath.
            (returns None)
        """
        if self.SpaceTyp == "Chem":
            Aux = ""
            for ii in range(MLEntry.Fold_L1):
                Aux += "{}_".format(str(self.FingerP[ii]))
            L1_Path  = Aux.rstrip(Aux[-1])                           # remove last underscore
            Aux = ""
            for ii in range(MLEntry.Fold_L2):
                Aux += "{}_".format(str(self.FingerP[ii]))
            L2_Path  = Aux.rstrip(Aux[-1])                           # remove last underscore

        if self.SpaceTyp == "Conf":
            L1_Path = "{}".format(str(self.FingerP[0]))
            L2_Path = "{}_{}".format(str(self.FingerP[0]), str(self.FingerP[1]))

        IOFile   = MLEntry.MergePath + "/" + L1_Path + "/" + L2_Path + "/" + self.ID + ".dat"
        FID      = open(IOFile, "r")
        FLoc     = FID.readlines()
        FID.close()

        # Descriptor "Geometry" specified.
        if "Geometry" in self.DataL:
            self.NAt = int(FLoc[0].split()[0])
            line = 2
            # GeomH - Molecular Geometry _including_ hydrogen atoms. Storage format for each of the self.NAt atoms:
            # [0] : x (float)
            # [1] : y (float)
            # [2] : z (float)
            # [3] : atom type (stored as float)
            self.GeomH = np.zeros((self.NAt, 4))
            Aux1       = np.zeros((self.NAt))
            Aux2       = np.zeros((self.NAt))
            for ii in range(self.NAt):
                self.GeomH[ii, 0] = float(FLoc[line].split()[1])
                self.GeomH[ii, 1] = float(FLoc[line].split()[2])
                self.GeomH[ii, 2] = float(FLoc[line].split()[3])
                self.GeomH[ii, 3] = PSEDICT[FLoc[line].split()[0]]
                Aux1[ii]          = float(FLoc[line].split()[4])
                Aux2[ii]          = float(FLoc[line].split()[5])
                line += 1
            self.Data['GeoMullQ'] = dc(Aux1)                   # Store Mulliken Charges at each atom.
            self.Data['GeoSpinP'] = dc(Aux2)                   # Store Spin Density at each atom.

        # Descriptor "SEmpOrbInfo"  specified -- adapted from original "ReadReMerged" function.
        # Assumes that the orbital section is ALWAYS written directly after the last atoms.
        if ("SEmpOrbInfo_F" in self.DataL) or ("SEmpOrbInfo_R" in self.DataL):
            # Part 1 - Get orbital energies and energy differences.
            GeoEndSec = self.NAt+2
            LocOccLen = -int(FLoc[GeoEndSec].split()[0])+1
            LocVirLen = 0
            line      = GeoEndSec + LocOccLen
            Items     = 2
            while Items == 2:
                LocVirLen += 1
                line += 1
                Items = len(FLoc[line].split())
            LocTransList = []
            LocOccOrbs   = np.zeros((LocOccLen, 2))
            LocVirOrbs   = np.zeros((LocVirLen, 2))
            LocOCCS      = np.zeros((MLEntry.MOWin))
            LocVIRS      = np.zeros((MLEntry.MOWin))
            LocTUPS      = []
            LocTransList = []
            # Read all Occ and Vir Orbs
            line = dc(GeoEndSec)
            for ii in range(LocOccLen):
                LocOccOrbs[ii, 0] = float(FLoc[line].split()[0])
                LocOccOrbs[ii, 1] = float(FLoc[line].split()[1])
                line += 1
            for ii in range(LocVirLen):
                LocVirOrbs[ii, 0] = float(FLoc[line].split()[0])
                LocVirOrbs[ii, 1] = float(FLoc[line].split()[1])
                line += 1
            OrbInfoLine = dc(line)
            # Next, for each "Jump" generate the "LenPerJump" lowest energy differences and store
            CurJump = 0
            Triggered = False
            while CurJump < MLEntry.MOWin:
                for ii in range(LocOccLen):
                    for jj in range(LocVirLen):
                        if int(abs(LocOccOrbs[ii, 0])+abs(LocVirOrbs[jj, 0])) == CurJump:
                            LocTUPS.append([int(abs(LocOccOrbs[ii,0])), int(LocVirOrbs[jj,0])])
                            Aux = (LocVirOrbs[jj, 1] - LocOccOrbs[ii, 1]) * EV
                            LocTransList.append(Aux)
                            LocOCCS[int(abs(LocOccOrbs[ii,0]))] = LocOccOrbs[ii,1]
                            LocVIRS[int(abs(LocVirOrbs[jj,0]))] = LocVirOrbs[jj,1]
                CurJump += 1
            line = OrbInfoLine
            while "alpha electrons" not in FLoc[line]:
                line += 1
            NOcc = int(FLoc[line].split()[0])
            self.Data['OccMO']     = dc(LocOCCS)       # HOMO energies.
            self.Data['VirMO']     = dc(LocVIRS)       # LUMO energies.
            self.Data['OccVirTup'] = dc(LocTUPS)       # List of OCC/VIR combinations for the energy differences.
            self.Data['OccVirDif'] = dc(LocTransList)  # Energy differences w.r.t. the OccVirTup list.
            self.Data['NOcc'] = dc(NOcc)               # Number of occupied orbitals.
            Aux = []

            # Part 2 - Get data for the Coulomb-Matrix-like descriptors.
            # A. P-Orbital shapes of occupied and virtual orbitals within the given MO window.
            while "Molecular Orbital Coefficients:" not in FLoc[line]:
                line += 1
            line += 1
            Rewi  = dc(line)
            NLine = 3
            line += 3
            while "Eigenvalues" not in FLoc[line]:
                NLine += 1
                line  += 1
            NLine  -= 2
            NAOs    = NLine - 3
            NBlocks = int((NAOs - (NAOs % 5)) / 5)
            if (NAOs % 5) > 0:
                NBlocks += 1
                line      = Rewi
            IDOccRead = []
            IDVirRead = []
            for ii in reversed(range(NOcc-MLEntry.MOWin+1, NOcc+1)):
                IDOccRead.append(ii)
            for ii in range(NOcc+1, NOcc+MLEntry.MOWin+1):
                IDVirRead.append(ii)
            # Read the Occs
            AuxStore1    = []
            OCCID        = []
            for ii in IDOccRead:
                CurPs = np.zeros((self.NAt))
                # Find starting line and starting block
                OCCID.append(ii-self.Data['NOcc'])
                for jj in range(NBlocks):
                    try:
                        auxlen = len(FLoc[Rewi+jj*NLine].split())
                        for kk in range(auxlen):
                            if str(ii) == FLoc[Rewi+jj*NLine].split()[kk]:
                                line = Rewi+jj*NLine
                                curBlocks = len(FLoc[Rewi+jj*NLine].split())
                    except IndexError:
                        break
                for jj in range(len(FLoc[line].split())):
                    if str(ii) == FLoc[line].split()[jj]:
                        NthBlock = jj
                ### Start Reading
                line   += 3
                ATIDX   = -1
                for _ in range(NAOs):
                    if len(FLoc[line].split()) > curBlocks + 2:
                        if "D 0" not in FLoc[line]:
                            ATIDX += 1
                    if "PX" in FLoc[line]:
                        CurPs[ATIDX] += float(FLoc[line].split()[NthBlock+2])
                    if "PY" in FLoc[line]:
                        CurPs[ATIDX] += float(FLoc[line].split()[NthBlock+2])
                    if "PZ" in FLoc[line]:
                        CurPs[ATIDX] += float(FLoc[line].split()[NthBlock+2])
                    line += 1
                ### Store as array along atoms for each Orbital.
                AuxStore1.append(CurPs)
            self.Data['OccPChar'] = dc(AuxStore1)
            AuxStore1             = []
            # Read the Virs
            AuxStore1    = []
            VIRID        = []
            for ii in IDVirRead:
                CurPs = np.zeros((self.NAt))
                # Find starting line and starting block
                VIRID.append(ii-NOcc-1)
                for jj in range(NBlocks):
                    try:
                        auxlen = len(FLoc[Rewi+jj*NLine].split())
                        for kk in range(auxlen):
                            if str(ii) == FLoc[Rewi+jj*NLine].split()[kk]:
                                line = Rewi+jj*NLine
                                curBlocks = len(FLoc[Rewi+jj*NLine].split())
                    except IndexError:
                        break
                # This "NthBlock" determines, which column to read from.
                for jj in range(len(FLoc[line].split())):
                    if str(ii) == FLoc[line].split()[jj]:
                        NthBlock = jj
                ### Start Reading
                line   += 3
                ATIDX   = -1
                for _ in range(NAOs):
                    if len(FLoc[line].split()) > curBlocks + 2:
                        if "D 0" not in FLoc[line]:
                            ATIDX += 1
                    if "PX" in FLoc[line]:
                        CurPs[ATIDX] += float(FLoc[line].split()[NthBlock+2])
                    if "PY" in FLoc[line]:
                        CurPs[ATIDX] += float(FLoc[line].split()[NthBlock+2])
                    if "PZ" in FLoc[line]:
                        CurPs[ATIDX] += float(FLoc[line].split()[NthBlock+2])
                    line += 1
                ### Store as array along atoms for each Orbital.
                AuxStore1.append(CurPs)
            self.Data['VirPChar'] = dc(AuxStore1)

            # B. For each HOMO-LUMO combination (from the self.Data['OccVirTup']), generate a "Transition Overlap" with the p-orbital shapes.
            AuxStore1 = []
            AuxStore2 = []
            CurJump   = 0
            while CurJump < MLEntry.MOWin:
                for ii in range(len(self.Data['OccPChar'])):
                    for jj in range(len(self.Data['VirPChar'])):
                        if int(abs(OCCID[ii])+abs(VIRID[jj])) == CurJump:
                            Aux    = np.zeros((self.NAt))
                            AuxSum = 0
                            for kk in range(self.NAt):
                                Aux[kk] = self.Data['VirPChar'][jj][kk] * self.Data['OccPChar'][ii][kk]
                                AuxSum += abs((self.Data['VirPChar'][jj][kk] * self.Data['OccPChar'][ii][kk])**2)
                            AuxStore1.append(Aux)
                            AuxStore2.append(AuxSum)
                CurJump += 1
            self.Data["OccVirPTransChar"] = dc(AuxStore1)
            self.Data["OccVirPTransSum"]  = dc(AuxStore2)

            # C. Generate a guess for the "LUMO**2 - HOMO**2" density difference.
            Aux    = np.zeros((self.NAt))
            for ii in range(self.NAt):
                Aux[ii] = (self.Data['VirPChar'][0][ii]**2) - (self.Data['OccPChar'][0][ii]**2)
            self.Data["HOLUPDiff"] = dc(Aux)
        return None
