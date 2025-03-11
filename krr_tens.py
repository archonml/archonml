# # # # # # # # # # # # #
# The DataTens class flexibly collects the descriptors of the  MLEntries, calculates "master" distance matrices
# and brings them into the required kernel form for the KRR model.
#
# Further, it handles splitting the data into Test/Training sets and can prepare K-fold Cross-Validation for optimizing
# the KRR hyperparameters "Sigma" and "Lambda". Note, that for EACH descriptor, there is an initial guess for Sigma made,
# that is then (globally) scaled with a single hyperparameter. The guess is made from estimating the length of the full width half maximum
# in an idealised histogram picture. Binning of histograms is performed flexibly, such that a minimum resolution (number of bins) for the FWHM
# can be requested.
#
# Note also, that the DataTens class does not know the names of any descriptors that it will compute the distance matrices of.
# That is possible, since each descriptor comes attached with information of "how to deal with it" when calculating the distance matrix in the end.
#
# FW 07/13/23

import time
from copy import deepcopy as dc

import numpy as np
import psutil

from .utils import dist_mat_calc, euc_dist_mat_calc, get_sub_mat, get_sub_mat_only, get_sub_mat_pred
from .utils import calc_pred_distmat, calc_pred_eucdistmat, calc_pred_distmat_layer, calc_pred_eucdistmat_layer
from .utils import otf_sub_distmat, otf_sub_eucdistmat, otf_sub_distmat_layer, otf_sub_eucdistmat_layer
from .utils import otf_sub_distmat_only, otf_sub_eucdistmat_only, otf_sub_distmat_layer_only, otf_sub_eucdistmat_layer_only
from .utils import ramval

class DataTens:
    """
    A class that holds and manages the data for training / prediction with the kernel
    ridge regression model.

    --- Class Attributes ---
    ------------------------
    CVGridPts : int
        The resoltion of the hyperparameters. Strongly affects training time.

    CVSize : int
        The number 'K' of cross-validation steps used for K-fold stratified
        corss-validation.

    RandomStrat : bool
        A boolean controlling whether stratification bins should be randomized.
        Only useful during debugging.

    FixedSeed : bool
        A boolean to control whether a fixed seed should be used in all methods
        that use random number generators.
        Useful for designing your model, as training/test split and
        cross-validation sets stay the same while you may adjust your model.
        Can be deactivated to create several training/test splits and run
        statistics on your learning curves.

    MinMod : float
        A multiplier for the lower range of the initial sigma guess. Should
        be adjusted for every new model, core structure and label.

    MaxMod : float
        A multiplier for the upper range of the initial sigma guess. Should
        be adjusted for every new model, core structure and label.

    Lambda_Bot : float
        A lower bound of the LOGARITHMIC range of lambda values in the
        hyperparameter optimization.

    Lambda_Top : float
        An upper bound of the LOGARITHMIC range of lambda values in the
        hyperparameter optimization.

    BinWinThrsh : int
        The convergence criterion for binning-resolution when the initial
        sigma values are guessed. See documentation for more details on
        this process.

    BinWinEscape : int
        The number of iterations that the initial sigma value search will
        adjust the binning resolution before taking the strictest guess.
        See documentation for more details on this process.

    --- Object Attributes ---
    -------------------------

    memmode : str
        An option that specifies how distance matrices will be built / stored. If specified as
        'high', a look-up rank 3 tensor ('DistMats') will be constructed and kept in memory for
        fast updating of the individual cross-validation steps.
        If specified as 'medium', smaller local rank 3 tensors will be constructed on-the-fly at
        each cross-validation step.
        If specified as 'low', all kernel matrices will be constructed on-the-fly, whenever needed.
        (This is computationally very, very expensive - but allows training of 10000+ entries with
        50+ effective descriptors in only a few hundred megabytes)

    DBSpecs : list of list of str
        A list containing the database specifications, stored along the final model  as to locate
        all merged data during predictions.

    NEntries : int
        Stores the number of overall MLEntries in the provided MLObject list.

    ID : list of str
        Stores the unique name of each MLEntry (i.e. "PROJ123").

    FingerP : list of arrays of float
        Stores each individual fingerprint of each MLEntry.

    PrepType : list of str
        Stores whether each entry of the MLObjList is designated as 'Training' or 'Predicting'
        type.

    LabelL : list of str
        Stores the list of available labels.

    Labels : dicts of lists of float
        Stores the label values for each named Label contained in the LabelL list. This is
        only filled for the MLEntries that are designated as 'Training' type.

    DescL : list of str
        Stores the list of requested descriptors.

    DescType : list of dicts
        Stores the data type and operation for each of the DescL.

    NDescLay : int
        Number of overall 'layers', when considering the sum of individual descriptor
        contributions to the kernel matrix. This is the effective number of descriptors.

        While some descriptors will result in a single distance matrix, other descriptors
        will request several distance matrices at once (leading to rank 3 tensors). For
        example, the comparison of the number of electrons will yield a single distance matrix;
        requesting the occupied orbital energies (with an MOWin of 4) will result in four
        distance matrices (pairwise comparing HOMO-0, HOMO-1, HOMO-2, ...).

    DistMats : dict of (rank 3) tensors
        Stores the 'abstract distance (rank 3) tensor' for each descriptor. This
        look-up dictionary is only constructed if the memory mode is set to 'high'.

    CVTDistMats : dict of (rank 3) tensors
        Stores the 'abstract distance (rank 3) tensor' for each descriptor. This
        dictionary is a 'local instance' used for constructing the 'training
        Kernel matrix' during each cross-validation step.

    CVVDistMats : dict of (rank 3) tensors
        Stores the 'abstract distance (rank 3) tensor' for each descriptor. This
        dictionary is a 'local instance' used for constructing the 'validation
        Kernel matrix' during each cross-validation step.

    LAMBDA_GRID : array of float
        An array of float with CVGridPts values that is scanned during hyperparameter
        optimization. Should (typically) _not_ be updated in-between cross-validation steps.

    SIGMA_MAT : rank 3 tensor of float
        A matrix of floats containing the grid of sigma values for each of the NDescLay
        individual distance matrices. Takes the shape [NDescLay, CVGridPts].

    TrainTens : rank 3 tensor of float
        A rank 3 tensor containing the 'abstract distances' within the training entries.
        Shape follows [NDescLay, NCVTrain, NCVTrain]

    ValiTens : rank 3 tensor of float
        A rank 3 tensor containing the 'abstract distances' for predicting the validation
        data. Shape follows [NDescLay, NCVValidation, NCVTrain]

    TrainLabels : array of float
        The label values of the training set for the _current_ cross-validation step.
        Is updated at each step of the cross-validation according to what IDs is provided
        to the update_cv_tens method.

    ValiLabels : array of float
        The label values of the validation set for the _current_ cross-validation step.
        Is updated at each step of the cross-validation according to what IDs is provided
        to the update_cv_tens method.

    FinTrainTens : rank 3 tensor of float
        A rank 3 tensor containing the 'abstract distances' for training the final model
        with the then optimized hyperparameters.
        Shape follows [NDescLay, NCVValidation, NCVTrain]

    FinTestTens : rank 3 tensor of float
        A rank 3 tensor containing the 'abstract distances' for testing the final model
        with the then optimized hyperparameters. Note, that the test data set has never been
        shown to the model before, as it is split off by the train_test_split method.
        Shape follows [NDescLay, NCVValidation, NCVTrain]

    FinTrainLabels : array of float
        The label values of the full training set for the final model training.

    FinTestLabels : array of float
        The label values of the test set for the final model training.

    CVTrainIDs : array of int
        The _numerical pointers_ with respect to the MLObjList for each MLEntry used
        in the training set of the _current_ cross-validation step.
        (in other words - NOT "PROJ123" but 11 th entry of MLObjList)

    CVValiIDs : array of int
        The _numerical pointers_ with respect to the MLObjList for each MLEntry used
        in the validation set of the _current_ cross-validation step.
        (in other words - NOT "PROJ123" but 11 th entry of MLObjList)

    Status : list of str
        A list of string that determines how each MLEntry of the MLObjList is to be
        treated during training. Note that this is _in addition_ to the PrepType.
        Is assigned as 'Test' per default and then updated by the train_test_split
        method.

        Possible values:
        'Test'
            This Entry is selected to be testing data. Will not be used during
            cross-validation, but for determining model performance.
        'TrainAct'
            This Entry is selected as active training data. It will be part of
            one the cross-validation blocks.
        'TrainInAct'
            This Entry is temporarily deactivated and will not be part of the
            cross-validation. Useful for creating learning curves.

    IniStat : str
        Initial status of this MLEntry. This status is kept frozen to allow running
        train_test_split several times.

    NTrain : int
        Number of MLEntries assigned as PrepType 'Training'.

    NPred : int
        Number of MLEntries assigned as PrepType 'Predicting'.

    NActive : int
        Number of MLEntries assigned as Status 'TrainAct'.

    NInactive : int
        Number of MLEntries assigned as Status 'TrainInAct'.

    NTest : int
        Number of MLEntries assigned as Status 'Test'.

    NTrainT : int
        Number of MLEntries assigned as Status of 'TrainAct' or 'TrainInAct'.

    CVIDBlocks : list of lists of int
        Lists containing the _numerical pointers_ to the entries in the MLObjList
        that were sorted into the cross-validation blocks during the stratify
        method.

    --- Methods ---
    ---------------
    train_test_split(self, iPerc, UPDATE=False, ActPerc=1.0)
        A method to split off a testing set from the known training data.
        Changing the update option to true allows to then activate /deactivate
        parts of the remaining training data (AFTER splitting) for the
        cross-validation.
        (returns None)

    stratify(self, iLabelL)
        A method to put all active training data into bins of evenly distributed
        data. Even distribution will be done with respect to the given iLabelL.
        The number of bins is the same as CVSize.
        (returns None)

    update_cv_tens(self, TrainIDs, ValiIDs, MLObjList=[])
        A method that will fill the two required distance tensors for the cross-validation
        training depending on which CVIDBlocks were chosen as training and validation
        sets. Data management depends on the memmode settings.
        The method will also flexibly determine the grid of sigma hyperparameters for
        later forming the kernel matrices during cross-validation.
        (returns None)

    final_tens(self, TrainIDs, TestIDs, MLObjList=[])
        A method that fills the final kernel matrices without determination of
        sigma values.
        (returns None)

    get_pred_tens(self, MLObjList, TrainIDs, PredIDs, Sigmas)
        A method that fills a kernel matrix for predictions. Reads in the
        sigma of a previously finished and saved model.
        (returns prediction kernel matrix)
    """
    CVGridPts = 64                                                                   # A class attribute for the number of Gridpoints for the Lambda and Sigma Grid generation of the later
                                                                                     # Hyperparameter Optimization.
     
    CVSize      = 5                                                                  # Number of Cross-Validation Blocks / Steps.
    RandomStrat = True                                                               # Activate or Deactivate Randomized Stratification.
                                                                                     # Note that this will result in differently trained models across machines,
                                                                                     # but is the scientifically correct method. Deactivation should only be for debugging across
                                                                                     # different machines.
             
    FixedSeed   = True                                                               # Use a fixed seed during random stratification and Test/Train splitting.
                                                                                     # This will ensure that (at least on one machine) the randomization should stay the
                                                                                     # same all the time.
 
    MinMod    = 0.01                                                                 # Global attributes for the Sigma-Range broadening inside the Hyperparameter Optimization.
    MaxMod    = 50000
 
    Lambda_Bot =  -6                                                                 # Cross-Validation Lambda-space. This is logarithmic.
    Lambda_Top =   2
 
    BinWinThrsh = 10                                                                 # Minimum number of bins in the histogrammatic determination of sigma-windows.
                                                                                     # Try to find a FWHM of at least N bins resolution.
    BinWinEscape = 100                                                               # Escape the Flexible FWHM search after this amount of tries.
 
 
    # Initialization of the Tensor. Will automatically produce the Distance Matrices, unless low memory flag is set to true.
    def __init__(self, MLObjList, mem_mode="high"):                                    # MLObjList is a list of MLObjects.
        """
        --- Parameters ---
        ------------------
        MLObjList : list of MLEntry objects
            The list of all MLEntries after adding the descriptors.

        mem_mode : str
            An option that specifies how distance matrices will be built / stored. If specified as
            'high', a look-up rank 3 tensor ('DistMats') will be constructed and kept in memory for
            fast updating of the individual cross-validation steps.
            If specified as 'medium', smaller local rank 3 tensors will be constructed on-the-fly at
            each cross-validation step.
            If specified as 'low', all kernel matrices will be constructed on-the-fly, whenever needed.
            (This is computationally very, very expensive - but allows training of 10000+ entries with
            50+ effective descriptors in only a few hundred megabytes)
        """
                                                                                     # Note that the lists during creation are quite large and should be flushed as soon as possible.
        # Take the _first_ MLObject of the list and extract the DescL, DescType and LabelL Lists.
        # Initialize the empty dictionary of Distance Matrices.
        self.NEntries    = len(MLObjList)                                            # Number of Entries total.
        self.ID          = []                                                        # copies the ID of the MLObjects - this will make storage/loading less interdependent...
        self.FingerP     = []                                                        # copies the Fingerprint of the MLObjects - this will make storage/loading less interdependent...
        self.DescL       = dc(MLObjList[0].DescL)
        self.DescType    = dc(MLObjList[0].DescType)
        self.DistMats    = {}                                                        # This dictionary will later contain all different distance matrices and rank 3 distance tensors. Used in high memory mode. 
        self.CVTDistMats = {}                                                        # This dictionary will later contain the sub-matrics for the train x train CV block. Used in medium memory mode.
        self.CVVDistMats = {}                                                        # This dictionary will later contain the sub-matrics for the vali x train CV block. Used in medium memory mode.
        self.NDescLay    = 0                                                         # Number of Layers of "flat" descriptor matrices.
        self.LabelL      = dc(MLObjList[0].LabelL)
        self.Labels      = {}                                                        # This dictionary collects _ALL_ labels for _ALL_ MLEntries!
        self.LAMBDA_GRID = np.logspace(DataTens.Lambda_Bot, DataTens.Lambda_Top
                                 , num=DataTens.CVGridPts)
        self.PrepType    = "Training"                                                # Behaviour types for the DataTens object. Can be either "Training" or "Predicting".
        self.memmode     = mem_mode

        # Initialize Labels as Dictionaries.
        self.TrainLabels = {}
        self.ValiLabels  = {}
  
        # Initialize tensor placeholders. The "Fin" Tensors are for the final training, and the "Pred" Tensor is for Predictions after loading a model.
        self.FinTrainTens = []
        self.FinTestTens  = []
  
        # Initiliaze finalization Labels as Dictionaries.
        self.FinTrainLabels = {}
        self.FinTestLabels  = {}

        # Initialize current cross-validation indices. These will get updated during the update_cv_tens method.
        # These have to be stored so that they can be dynamically accessed by the krr_model module in case of on-the-fly low-memory mode.
        self.CVTrainIDs = []
        self.CVValiIDs  = []
  
        # FOR NOW - force that all data for one model comes from the same software / low-QC method!
        # Initialize the three Database Specs (QC-Package, low-QC and LDataL)
        self.DBSpecs        = [[dc(MLObjList[0].QC_Pack)], [dc(MLObjList[0].QC_low)], [dc(MLObjList[0].QC_high)], [dc(MLObjList[0].DataL)]]
  
        # initialize the IDs and initial data status
        Aux1, Aux2, Aux3 = [], [], []
        self.NTrain      = 0                                                            # Number of Training Entries - NOTE: irrespective of active / inactive / Test.
        self.NPred       = 0                                                            # Number of Prediction Entries.
        for ii in range(self.NEntries):
            Aux1.append(MLObjList[ii].ID)
            Aux2.append(MLObjList[ii].FingerP)
            Aux3.append(MLObjList[ii].Type)
            if MLObjList[ii].Type == "Training":
                self.NTrain += 1
            if MLObjList[ii].Type == "Prediction":
                self.NPred  += 1
        self.ID      = dc(Aux1)
        self.FingerP = dc(Aux2)
        self.Status  = dc(Aux3)                                                         # This is the current Status of entries. May be changed.
        self.IniStat = dc(self.Status)                                                  # This is the initialized Status. Shall not be changed ever.

        if self.NPred > 0:
            self.PrepType = "Predicting"

        if self.memmode == "auto":
            print("Now estimating memory requirements assuming worst-case train-test-splits to account for potential overhead...")
            BGMem = ramval()                                                # Background Memory - mostly from the module itself.
            LayCnt= 0                                                       # Count the number of descriptor layers now!
            for ii in self.DescL:
                if self.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = self.DescType[ii][0]
                    LocDataO = self.DescType[ii][1]                                         # Get the Distance operator
                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            LayCnt += 1
                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            LayCnt += 1
                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            LayCnt += LocArrLen
                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            LayCnt += LocArrLen
            print("... working with {} descriptor layers for {} total molecules.\n".format(LayCnt, self.NEntries))
            # HIGH MEMORY - storing the full distance matrices.
            ####       byte-per-float  *  number of descriptors  *  number of molecules
            DMatMem =        8         *       float(LayCnt)     *  float(self.NEntries)**2          # memory for one copy of the full distance matrix - this is "DistMat"

            # HIGH / MEDIUM MEMORY - storing the Train / Validation Distance Sub Matrices.
            # These can not be "flushed" during a step of CV, since they will be dynamically "decorated" with the Sigma values to form KMatrices...
            # These will be flushed at the start of "final_cv"!
            # Assume that no train/test split was done - i.e. all data for the training.
            CVBlockSize = float(self.NEntries)/float(self.CVSize)
            CVNTrain    = (CVBlockSize * (float(self.CVSize)-1))**2
            CVNVali     = CVBlockSize  * (CVBlockSize * (float(self.CVSize)-1))

            ####       byte-per-float  *  number of descriptors  *  number of molecules
            CVTrainMat =     8         *       float(LayCnt)     *  CVNTrain                         # memory for one copy of the CV-Train sub matrix - this is the "TrainTens"
            CVValiMat  =     8         *       float(LayCnt)     *  CVNVali                          # memory for one copy of the CV-Vali sub matrix - this is the "ValiTens"

            # HIGH / MEDIUM MEMORY - final Training Tensors 
            # assume same size as CV tensors.
            FinTrainMat =     CVTrainMat                         
            FinValiMat  =     CVValiMat

            # HIGH / MEDIUM / LOW MEMORY - storing the KMatrices; for training and prediction
            KMatMem     = CVTrainMat / float(LayCnt)
            PMatMem     = CVValiMat / float(LayCnt)

            # Assume, that dynamic loading into functions doubles all values.
            # Finally, add a 10 % overhead anyways...
            HighMemReq    = (2 * (DMatMem + FinTrainMat + FinValiMat + KMatMem + PMatMem) * 1.1) + BGMem
            MediumMemReq  = (2 * (CVTrainMat + FinValiMat + KMatMem + PMatMem) * 1.1) + BGMem 
            LowMemReq     = (2 * (KMatMem + PMatMem) * 1.1) + BGMem
            print("High   Memory would require {:8.2f} MB.".format(HighMemReq/(1024*1024)))
            print("Medium Memory would require {:8.2f} MB.".format(MediumMemReq/(1024*1024)))
            print("Low    Memory would require {:8.2f} MB.".format(LowMemReq/(1024*1024)))

            print("")
            PhysMem = psutil.virtual_memory().total
            print("There is {:8.2f} MB available on this machine ...".format(PhysMem/(1024*1024)))
            print("")

            self.memmode = "high"
            if HighMemReq > PhysMem:
                self.memmode = "medium"
                if MediumMemReq > PhysMem:
                    self.memmode = "low"

        if self.memmode == "high":
            print("High memory mode was chosen for training!")
            print("... distance matrices will be stored as look-up-tables now!")
        if self.memmode == "medium":
            print("Medium memory mode was chosen for training!")
            print("... distance matrices will be stored only for the sub-blocks during each individual")
            print("    cross-validation step and for the final step!")
        if self.memmode == "low":
            print("Low memory mode was chosen for training!")
            print("... distance matrices will ALWAYS be generated on-the-fly!")

        # Implemented a 4-layered self.Status for handling the data - containing "Prediction", "Test", "TrainActive", "TrainInactive" statuses.
        # "Prediction" data may not be entered into the Cross-Validation Training.
  
        # General "Training Data" (from MLObject) shall be split into the other 3 groups according to the Train/Test Splitting.
        # Test data may not be entered into the Cross-Validation Training.
        # Active Training Data will be entered into the Cross-Validation Training Block.
        # Inactive Training Data is kept outside the CV-Training Block automatically - this allows for making a learn-curve while keeping the Test Data fixed.
  
        # NOTE - changing the percentage of active / inactive training data will give a NEW random sample each time;
        #        however, this behaviour is NOT desired when splitting off the Test data. Thus, "UPDATING" should not perform a Train/Test split,
        #        but just change the fraction of active inactive!
        # Go through every descriptor, allocate the data for the distance matrix and calculate the matrix.        
        # Initialize the Descriptors for the data tensors

        # The DescTens class has two distinct behaviours - one for training mode, and one for prediction mode.

        # TRAINING MODE
        # When preparing for training, it makes sense to generate all distance matrices in memory as a look-up-table.
        # That way, when changing through the different data-blocks during individual cross-validation steps, one can just
        # look up the currently needed IDs for Training and Testing and build the "local" Kernel matrix.
        #
        # Note also, that no on-the-fly "embedding" can be performed here. "Embedding" is what I refer to as putting together the
        # distance matrices with their respective sigma values to form the Kernel matrix. Since the sigma have not been optimized yet,
        # on-the-fly summation is not available (yet).


        # PREDICTION MODE
        # When preparing for predictions, it does not make sense to build a look-up table, since only one Kernel matrix will be needed anyway.
        # Hence, in this case, the distance matrices can be "embedded" immediately with the pre-optimized sigmas.

        # Pre-generate all distance matrices in case of "Training mode". Only do that, if low-mem flag is set to false.
        if self.PrepType == "Training":
            for ii in self.DescL:
                if self.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = self.DescType[ii][0]                                         # Get the Data Format
                    LocDataO = self.DescType[ii][1]                                         # Get the Distance operator
    
                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            if self.memmode == "high":
                                print("Initializing Distance Matrix for descriptor {}.".format(LocDescL))
                                print("   ...using Scalar data to form a direct Distance Matrix.")
                                LocScals = np.zeros((self.NEntries))                            # Grab Data - here an array of scalars
                                for jj in range(self.NEntries):
                                    LocScals[jj] = dc(MLObjList[jj].Desc[LocDescL])
                                LocDistMat = dist_mat_calc(LocScals)                            # Initialize the Distance Matrix
                                self.DistMats[LocDescL] = dc(LocDistMat)                        # Store Distance Matrix
                                LocDistMat, LocScals = [], []                                   # Flush temporary Memory
                            self.NDescLay += 1
      
                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            if self.memmode == "high":
                                print("Initializing Distance Matrix for descriptor {}.".format(LocDescL))
                                print("   ...using eigenvectors of CM-like object data to form Distance Matrix of Euclidean Norms.")
                                LocObjs = []                                                    # Grab Data - here a list of arrays with possibly different lengths
                                for jj in range(self.NEntries):
                                    LocObjs.append(dc(MLObjList[jj].Desc[LocDescL]))
                                LocDistMat = euc_dist_mat_calc(LocObjs)                         # Initialize the Distance Matrix
                                self.DistMats[LocDescL] = dc(LocDistMat)                        # Store Distance Matrix
                                LocDistMat, LocObjs = [], []                                    # Flush temporary Memory
                            self.NDescLay += 1

                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            if self.memmode == "high":
                                print("Initializing Distance rank 3 tensor for descriptor {}.".format(LocDescL))
                                print("   ...using arrays of Scalar data to form {} direct Distance Matrices.".format(LocArrLen))
                                LocDistR3Tens = np.zeros((LocArrLen, self.NEntries, self.NEntries))
                                # Form each individual Distance Matrix -                        for example: Orbital energies...
                                for jj in range(LocArrLen):                                     # ...take Orbital number jj of the OrbWin around HOMO-LUMO...
                                    LocScals = np.zeros((self.NEntries))
                                    for kk in range(self.NEntries):                             # ...read its energy for each molecule kk...
                                        LocScals[kk] = dc(MLObjList[kk].Desc[LocDescL][jj])
                                    LocDistR3Tens[jj, :, :] = dist_mat_calc(LocScals)           # Initialize the Distance Matrix 
                                self.DistMats[LocDescL] = dc(LocDistR3Tens)                     # Store Distance rank 3 tensor
                                LocDistSupertMat, LocScals = [], []                             # Flush temporary Memory
                            self.NDescLay += LocArrLen
             
                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            if self.memmode == "high":
                                print("Initializing Distance rank 3 tensor for descriptor {}.".format(LocDescL))
                                print("   ...using arrays of eigenvectors of CM-like object data to form {} Distance Matrices of Euclidean Norms.".format(LocArrLen))
                                LocDistR3Tens = np.zeros((LocArrLen, self.NEntries, self.NEntries))
                                # Form each individual Distance Matrix -                        for example: P-orbital characters...
                                for jj in range(LocArrLen):                                     # ...take Orbital number jj of the OrbWin around HOMO-LUMO...
                                    LocObjs = []
                                    for kk in range(self.NEntries):                             # ...read the CM-like eigenvectors for each molecule kk...
                                        LocObjs.append(dc(MLObjList[kk].Desc[LocDescL][jj]))
                                    LocDistR3Tens[jj, :, :] = euc_dist_mat_calc(LocObjs)        # Initialize the Distance Matrix 
                                self.DistMats[LocDescL] = dc(LocDistR3Tens)                     # Store Distance rank 3 tensor
                                LocDistSupertMat, LocObjs = [], []                              # Flush temporary Memory
                            self.NDescLay += LocArrLen

            ################################## Descriptor Section over - following the Label Section ######################################
            #
            # NOTE - this section will be performed regardless of low-memory flag.
            #
            # For the time being, just assume that Labels are just one value per structure. This can be changed later in a similar
            # DataF / DataO fashion below here, if desired.
            ###############################################################################################################################

            # For the time being, assume just singular scalar values per Label. Could be split up in similar DataF / DataO in the future here.
            for ii in range(len(self.LabelL)):
                LocLabelL  = self.LabelL[ii]
                LocEntries = []
                for jj in range(self.NEntries):
                    if MLObjList[jj].Type == "Training":
                        LocEntries.append(dc(MLObjList[jj].Label[LocLabelL]))
                    if MLObjList[jj].Type == "Prediction":
                        LocEntries.append([])
                self.Labels[LocLabelL] = LocEntries
                LocEntries = []


    # The TrainTestSplit method will assign each entry a self.status "TrainAct", "TrainInact", "Test" - using a RANDOM distribution according to the percentage iPerc.
    # "TrainAct" entries will be used to make the CV Data-Block (which in turn then has K-Fold, stratified sub-blocks).
    # "TrainInact" entries are NOT used for CV Data-Block; they can be activated/deactivated for making Learning Curves.
    # "Test" entries are ONLY for the test of the final model AFTER K-fold CV.    
    # Using the TrainTestSplit with UPDATE=True will change the active / inactive percentage randomly according to the ActPerc percentage - but will NOT affect the test/train split any more.
    # RANDOMNESS WARNING - this will result in differently trained ML models on different machines. Therefore, consider SAVING which entries were used when and how for 100 % reproducibility.
    def train_test_split(self, iPerc, Include=None, UPDATE=False, ActPerc=1.0):
        """
        A method to split off a testing set from the known training data.
        Changing the update option to true allows to then activate /deactivate
        parts of the remaining training data (AFTER splitting) for the
        cross-validation.

        The two-fold design is intended, as it allows to keep the test set while
        changing the percentage of active training data - which practically
        allows to make learning curves.
        (returns None)

        --- Parameters ---
        ------------------
        iPerc : float [0.0 < iPerc < 1.0]
            The percentage of data that is being kept as Training data, while the rest
            is split off as Testing data. At initialization (i.e. UPDATE=False) all
            Training data will be set as active (i.e. 'TrainAct')

        UPDATE : bool
            A boolean switch that will determine if the method will split off testing data,
            or whether it should change the active percentage of Training data.

        ActPerc : float [0.0 < ActPerc < 1.0]
            The percentage of traiing data that is kept as active. This parameter is only
            used in case UPDATE is set to True.

        --- Raises ---
        --------------
        IndexError:
            Raises an IndexError in case a set with 0 entries would be created.
        """
        print("Reminder: This function contains a random number generator. Make sure to transfer the outcome consistently across machines for full reproducibility.")
        # initialize Randomness according to fixed seed policy.
        if DataTens.FixedSeed:
            print("...using a fixed seed!")
            rng = np.random.RandomState(seed=1337)
        if not DataTens.FixedSeed:
            print("...using system time for seeding!")
            rng = np.random.RandomState(seed=int(time.time()))                          # If not using a fixed seed, use system time for seeding.

        # Initialize the Train-Test-Split.
        if not UPDATE:
            self.NActive   = 0                                                          # Number of Active Training Entries.
            self.NInactive = 0                                                          # Number of Inactive Training Entries.
            self.NTest     = 0                                                          # Number of Test Entries.
                                                                                        # (Remember : all three belong to the "Training Data" - as opposed to prediction data)
            self.NTrainT   = 0                                                          # Number of training entries irrespective of active / inactive (BUT excludiing NTest!!)
            # Get the integer number closest to the desired iPerc percentage times the total "Training Data". This is the number of (initially) active Training entries.
            # i.e. iPerc of 80 % will yield 80 % of "Training Data" as active training data - but ignores Prediction data.
            LocN_TTSplit = int(round((self.NTrain * iPerc), 0))
            if LocN_TTSplit == 0:
                raise IndexError("Test / Trainig set of 0 entries detected! Something must have gone wrong somewhere...")
            # Now, assign all initial "Training" entries the status "Test" for easier management. NOTE that IniStat is always kept frozen - such that the routine may be repeated to give new Train/Test sets.
            for ii in range(len(self.IniStat)):
                CurIniStat = self.IniStat[ii]
                if CurIniStat == "Training":
                    self.Status[ii] = "Test"
                    self.NTest += 1
            # Randomly Assign "TrainAct" self.Status for entries that have been. NOTE that status is initialized with all entries as "Test".
            # NOTE - NTrain is the fixed reference point for the active/inactive splitting.
            #        It is only updated when the training / testing split is changed.
            Loccnt = 0
            if Include is None:
                print("Running a new initialization - this will create a NEWLY PICKED Test set!")
                while Loccnt < LocN_TTSplit:
                    LocPick = rng.randint(0, self.NEntries)
                    if self.Status[LocPick] == "Test":
                        self.Status[LocPick] = "TrainAct"
                        self.NTest   -= 1
                        self.NTrainT += 1
                        self.NActive += 1
                        Loccnt += 1
            # Add the to-include entries first - then resume the while loop.
            if Include is not None:
                print("Running a new initialization - this will create a NEWLY PICKED Test set!\n")
                print("Detected an Include List - will add these entries to the active training space beforehand.")
                for ii in range(len(self.Status)):             # go along all entries
                    for jj in range(len(Include)):
                        Picked = Include[jj]
                        if self.ID[ii] == Include[jj]:         # if current ID is an exact match on the include list, set it as active.
                            self.Status[ii] = "TrainAct"       # (double-loop was used, since I'm not sure if "is in Include" statement
                                                               # would recognize non-exact matches, like ID = "PROJ123" and Include has something like
                                                               # "PROJ1234" in it.)
                            self.NTest     -= 1                
                            self.NTrainT   += 1
                            self.NActive   += 1
                            Loccnt += 1
                while Loccnt < LocN_TTSplit:
                    LocPick = rng.randint(0, self.NEntries)
                    if self.Status[LocPick] == "Test":
                        self.Status[LocPick] = "TrainAct"
                        self.NTest   -= 1
                        self.NTrainT += 1
                        self.NActive += 1
                        Loccnt += 1
     
        if UPDATE:
            LocN_ActInAct = int(round((self.NTrainT * ActPerc), 0))
            if LocN_ActInAct == 0:
                raise IndexError("Active / Inactive set of 0 entries detected! Something must have gone wrong somewhere...")
            # First - initialize all training entries as "TrainInact" - to get an equal starting point each time.
            self.NActive     = 0
            for ii in range(self.NEntries):
                if (self.Status[ii] != "Test"):
                    if (self.Status[ii] != "Prediction"):
                        self.Status[ii] = "TrainInact"
                        self.NInactive += 1

            # Randomly Overwrite "TrainAct" self.Status over the inactive entries.
            Loccnt = 0
            while Loccnt < LocN_ActInAct:
                LocPick = rng.randint(0, self.NEntries)
                if self.Status[LocPick] == "TrainInact":
                    self.Status[LocPick] = "TrainAct"
                    self.NActive   += 1
                    self.NInactive -= 1
                    Loccnt += 1
        return None


    # The Stratify Method will store (numerical) IDs that stratify the labels into N Blocks for the Cross Validation.
    # Note that for "fully specialized training", each ML-Model should be Cross-Validated for it's own prediction target label.
    # Further, note that this routine will throw away overhanging data for the time being, to make N blocks of same size.
    # It will print out a reminder message for that, though.
    # Remember, that stratification is only necessary ONCE per iLabel. 
    def stratify(self, iLabelL):
        """
        A method to put all active training data into bins of evenly distributed
        data. Even distribution will be done with respect to the given iLabelL.
        The number of bins is the same as CVSize.

        Needs to be performed AFTER splitting. Note, that this also means that
        the test data is _not stratified_, which is probably more realistic as you can
        not be sure beforehand whether any unknown set would behave according to the
        'typical' behaviour of what we would call 'typical' so far.

        --- Parameters ---
        ------------------
        iLabel : str
            The name of a Label that should be taken into account for stratification.
        """
        print("Reminder: This function contains a random number generator. Make sure to transfer the outcome consistently across machines for full reproducibility.")
        # Pre-Select the Active Training Set only.
        ActID = []
        for ii in range(self.NEntries):
            if self.Status[ii] == "TrainAct":
                ActID.append(ii)
        StratTarget = [self.Labels[iLabelL][ActID[ii]] for ii in range(len(ActID))]
        KFold = dc(DataTens.CVSize)
        Rest  = len(StratTarget) % KFold
        StratLabels   = np.zeros((len(StratTarget)-Rest))
        StratTrainIDs = []

        # Grab K-divisible Labels.
        print("Performing a {}-fold Stratification with respect to Label {}.".format(DataTens.CVSize, iLabelL))
        print("To obtain equal-sized blocks, {} entries will be omitted during the cross-validation.".format(Rest))
        for ii in range(len(StratTarget)-Rest):
            StratLabels[ii] = dc(StratTarget[ii])                                                         # Store each label...
            StratTrainIDs.append(ActID[ii])                                                               # ... together with the MAPPED active ID.

        # Data sorting for KFold stratification
        LocSortLabels = np.sort(StratLabels)
        LocSortIDs    = [StratTrainIDs[np.argsort(StratLabels)[ii]] for ii in range(len(StratTrainIDs))]     # These IDs are the mapped, arg-sorted numerical IDs belonging to the active-only Training set

        # Initialize K Bins of Labels and numerical IDs to evenly put data inside while going along all sorted IDs.
        # Note that the outcome is NOT randomly stratified (yet).
        LocIDBins    = []
        LocLabelBins = np.zeros(((int(len(StratLabels)/KFold)), KFold))
        NthStep = 0
        Bin     = 0
        Aux     = []
        for ii in range(len(LocSortIDs)):
            LocLabelBins[NthStep, Bin] = LocSortLabels[ii]
            Aux.append(LocSortIDs[ii])
            Bin += 1
            if Bin == KFold:                                               # If N entries have been binned into one line, start next line.
                                                                           # This ensures that each step, the sorted Labels are stored in K bins
                Bin      = 0
                NthStep += 1
                LocIDBins.append(Aux)
                Aux      = []

        # Perform randomized Stratification, if specified. This is the more correct way to do it - however, stay aware that the "random" functions are machine dependent.
        # Turning this on _will_ result in differently trained models across computers, unless the IDs of the CV-Blocks are _diligently_ carried over each time.
        # Using a Fixed Seed will allow the results to be true to randomization while staying the same at the same computer, at least.
        if DataTens.RandomStrat:
            if DataTens.FixedSeed:
                print("...using a fixed seed!")
                rng = np.random.RandomState(seed=1337)
            if not DataTens.FixedSeed:
                print("...using system time for seeding!")
                rng = np.random.RandomState(seed=int(time.time()))         # If not using a fixed seed, use system time for seeding.

            # The Randomization follows a scheme, where at first, a random bin for each step is taken and stored in the "Taken" List.
            # In the second run, at each step, the current Taken List for a specific step is appended randomly with the remaining possible entries.
            # The result is a List similar to the LocIDBins, where the IDs at each step have been shuffled "across columns" randomly.
            # Initialization of Taken List.
            Taken = []                                  # The Taken list is for checking, which bins have been used at the step already.
            ReOrg = []                                  # The ReOrg list will be filled with the corresponding numerical IDs.
            for ii in range(len(LocIDBins)):
                Pick = rng.randint(0, KFold)
                Taken.append([Pick])
                ReOrg.append([LocIDBins[ii][Pick]])

            # Next, go through all steps again, picking the remaining IDs at random - while making sure that no entry is taken twice.
            for ii in range(len(LocIDBins)):
                for _ in range(1, KFold):
                    Valid = False
                    while not Valid:                 # continue picking randomly until the picked item is not already inside the Taken list at that step.
                        Pick = rng.randint(0, KFold)
                        if Pick not in Taken[ii]:
                            Valid = True
                            Taken[ii].append(Pick)
                            ReOrg[ii].append(LocIDBins[ii][Pick])

            # Overwrite the LocIDBins with the randomized LocIDBins.
            LocIDBins = dc(ReOrg)

        # Finally, store the numerical IDs along K columns instead of rows.
        CVIDCols = []
        for ii in range(KFold):                 # Focus on ii'th COLUMN
            Aux = []
            for jj in range(len(LocIDBins)):    # Go down all jj steps of this column
                Aux.append(LocIDBins[jj][ii])   # Note that the "seemingly swapped" jj and ii are intended.
            CVIDCols.append(Aux)

        # Store the CVIDBlocks
        self.CVIDBlocks = dc(CVIDCols)
        return None


    # Local generation of the full Training and Validation Tensors (= array of Distance Matrices) alongside the Sigma Matrix (= one range of Sigmas for each DescL)
    # This will need to be updated between each Cross-validation step.
    # Note that the function is NOT universal though, such that it can't be used for final training after CV.
    # NOTE - TrainIDs and ValiIDs are NUMERICAL INTEGERS here, and do not refer to the name of a specific structure.
    def update_cv_tens(self, TrainIDs, ValiIDs, MLObjList=[]):
        """
        A method that will fill the two required distance tensors for the cross-validation
        training depending on which CVIDBlocks were chosen as training and validation
        sets. Data management depends on the memmode settings.
        The method will also flexibly determine the grid of sigma hyperparameters for
        later forming the kernel matrices during cross-validation.

        In case of 'high' memory setting, it will just use the pre-calculated DistMats
        look-up tensor. Otherwise, it will either calculate the local distance tensors
        now ('medium') and then allow for fast formation of the kernel matrices or
        it will fully form the kernel matrix directly on-the-fly and repeat this
        every time ('low').

        If possible, you should not use the 'low' option, as it is not practical
        for training - but exceptional for running the predictions (which is
        why it exists in the first place).

        --- Parameters ---
        ------------------
        TrainIDs : list of int
            A list of the _numerical pointers_ for picking out the MLEntries from
            the MLObjList that belong to the training data.

        ValiIDs : list of int
            A list of the _numerical pointers_ for picking out the MLEntries from
            the MLObjList that belong to the validation data.

        MLObjList : a list of MLEntry objects
            This is the full list of MLEntries - required, in case of memmode 'medium'
            or 'low'.
        """
        self.CVTrainIDs = dc(TrainIDs)
        self.CVValiIDs  = dc(ValiIDs)
        # for each Descriptor, form the local, submatrix of the overall Distance Matrix and generate its local sigmas
        LocTrainTens   = []
        LocValiTens    = []
        LocSigmas      = []
        if self.memmode == "high":
            for ii in self.DescL:
                if self.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = self.DescType[ii][0]                                         # Get the Data Format
                    LocDataO = self.DescType[ii][1]                                         # Get the Distance operator

                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            print("Embedding Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B, C = get_sub_mat(TrainIDs, ValiIDs, self.DistMats[LocDescL], BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                            LocTrainTens.append(A)
                            A = []
                            LocValiTens.append(B)
                            B = []
                            LocSigmas.append(C)
                            C = []
      
                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            print("Embedding Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B, C = get_sub_mat(TrainIDs, ValiIDs, self.DistMats[LocDescL], BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                            LocTrainTens.append(A)
                            A = []
                            LocValiTens.append(B)
                            B = []
                            LocSigmas.append(C)
                            C = []

                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            print("Embedding Distance rank 3 tensor for descriptor {}...".format(LocDescL))
                            LocArrLen = len(self.DistMats[LocDescL])
                            for jj in range(LocArrLen):
                                A, B, C = get_sub_mat(TrainIDs, ValiIDs, self.DistMats[LocDescL][jj], BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                                LocTrainTens.append(A)
                                A = []
                                LocValiTens.append(B)
                                B = []
                                LocSigmas.append(C)
                                C = []

                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            print("Embedding Distance rank 3 tensor for descriptor {}...".format(LocDescL))
                            LocArrLen = len(self.DistMats[LocDescL])
                            for jj in range(LocArrLen):
                                A, B, C = get_sub_mat(TrainIDs, ValiIDs, self.DistMats[LocDescL][jj], BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                                LocTrainTens.append(A)
                                A = []
                                LocValiTens.append(B)
                                B = []
                                LocSigmas.append(C)
                                C = []

        # In medium memory mode, store only the _current_ CV submatrices of the train x train and train x vali sets.
        if self.memmode == "medium":
            if MLObjList == []:
                print("Medium memory mode was selected. In this case, you need to provide the MLObjList to update_cv_tens as well.")
            for ii in self.DescL:
                if self.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = self.DescType[ii][0]                                         # Get the Data Format
                    LocDataO = self.DescType[ii][1]                                         # Get the Distance operator

                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B, C = otf_sub_distmat(TrainIDs, ValiIDs, MLObjList, LocDescL, BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                            LocTrainTens.append(A)
                            A = []
                            LocValiTens.append(B)
                            B = []
                            LocSigmas.append(C)
                            C = []
      
                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B, C = otf_sub_eucdistmat(TrainIDs, ValiIDs, MLObjList, LocDescL, BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                            LocTrainTens.append(A)
                            A = []
                            LocValiTens.append(B)
                            B = []
                            LocSigmas.append(C)
                            C = []

                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            for jj in range(LocArrLen):
                                A, B, C = otf_sub_distmat_layer(TrainIDs, ValiIDs, MLObjList, LocDescL, jj, BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                                LocTrainTens.append(A)
                                A = []
                                LocValiTens.append(B)
                                B = []
                                LocSigmas.append(C)
                                C = []

                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            for jj in range(LocArrLen):
                                A, B, C = otf_sub_eucdistmat_layer(TrainIDs, ValiIDs, MLObjList, LocDescL, jj, BINWINTHRSH=DataTens.BinWinThrsh, BINWINESCAPE=DataTens.BinWinEscape)
                                LocTrainTens.append(A)
                                A = []
                                LocValiTens.append(B)
                                B = []
                                LocSigmas.append(C)
                                C = []


        # After initialization of Submatrices and Sigmas, get the Local SigmaMat over the Gridpoints for the HyperOpt.
        # CVGridPts is a gloabl attribute of this class.
        # Store SIGMA_MAT not as a _self_ attribute, but rather allow to return it for deciding to update or not update during Cross-Validation.
        if (self.memmode == "high") or (self.memmode == "medium"):
            self.SIGMA_MAT = np.zeros((len(LocSigmas), DataTens.CVGridPts))
            for ii in range(len(LocSigmas)):
                Aux1 = LocSigmas[ii]*DataTens.MinMod
                Aux2 = LocSigmas[ii]*DataTens.MaxMod
                self.SIGMA_MAT[ii, :] = np.linspace(Aux1, Aux2, num=DataTens.CVGridPts)
            self.TrainTens = dc(LocTrainTens)
            self.ValiTens  = dc(LocValiTens)

        if self.memmode == "low":
            self.SIGMA_MAT = np.zeros((self.NDescLay, DataTens.CVGridPts))

        ################################## Descriptor Section over - following the Label Section ######################################
        #
        # NOTE - this section is active regardless of low-memory flag.
        #
        # For the time being, just assume that Labels are just one value per structure. This can be changed later in a similar
        # DataF / DataO fashion below here.
        ###############################################################################################################################

        for ii in range(len(self.LabelL)):
            LocLabelL = self.LabelL[ii]
            LocList   = self.Labels[LocLabelL]
            LocAux    = []
            for jj in TrainIDs:
                LocAux.append(LocList[jj])
            self.TrainLabels[LocLabelL] = dc(LocAux)
            LocAux    = []
            for jj in ValiIDs:
                LocAux.append(LocList[jj])
            self.ValiLabels[LocLabelL]  = dc(LocAux)

        return None


    # This method is used to set up the final testing set. Sigmas not needed anymore, here.
    def final_tens(self, TrainIDs, TestIDs, MLObjList=[]):
        """
        A method that fills the final kernel matrices without determination of
        sigma values.

        --- Parameters ---
        ------------------
        TrainIDs : list of int
            A list of the _numerical pointers_ for picking out the MLEntries from
            the MLObjList that belong to the training data.

        ValiIDs : list of int
            A list of the _numerical pointers_ for picking out the MLEntries from
            the MLObjList that belong to the validation data.

        MLObjList : a list of MLEntry objects
            This is the full list of MLEntries - required, in case of memmode 'medium'
            or 'low'.
        """
        # At this point, CV has finished for good. Thus, flush the Vali/Train Tensors, before doing anything else.
        self.TrainTens = []
        self.ValiTens  = []

        # for each Descriptor, form the local, submatrix of the overall Distance Matrix and generate its local sigmas
        LocTrainTens   = []
        LocTestTens    = []
        if self.memmode == "high":
            print("High-memory mode is active. Will generate the final tensors from look-up-table.")
            for ii in self.DescL:
                if self.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = self.DescType[ii][0]                                         # Get the Data Format
                    LocDataO = self.DescType[ii][1]                                         # Get the Distance operator

                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            print("Embedding Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B = get_sub_mat_only(TrainIDs, TestIDs, self.DistMats[LocDescL])
                            LocTrainTens.append(A)
                            A = []
                            LocTestTens.append(B)
                            B = []

                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            print("Embedding Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B = get_sub_mat_only(TrainIDs, TestIDs, self.DistMats[LocDescL])
                            LocTrainTens.append(A)
                            A = []
                            LocTestTens.append(B)
                            B = []

                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            print("Embedding Distance rank 3 tensor Matrices for descriptor {}...".format(LocDescL))
                            LocArrLen = len(self.DistMats[LocDescL])
                            for jj in range(LocArrLen):
                                A, B = get_sub_mat_only(TrainIDs, TestIDs, self.DistMats[LocDescL][jj])
                                LocTrainTens.append(A)
                                A = []
                                LocTestTens.append(B)
                                B = []

                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            print("Embedding Distance rank 3 tensor Matrices for descriptor {}...".format(LocDescL))
                            LocArrLen = len(self.DistMats[LocDescL])
                            for jj in range(LocArrLen):
                                A, B = get_sub_mat_only(TrainIDs, TestIDs, self.DistMats[LocDescL][jj])
                                LocTrainTens.append(A)
                                A = []
                                LocTestTens.append(B)
                                B = []

        if self.memmode == "medium":
            print("Medium-memory mode is active. Will generate the final tensors on-the-fly...")
            for ii in self.DescL:
                if self.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = self.DescType[ii][0]                                         # Get the Data Format
                    LocDataO = self.DescType[ii][1]                                         # Get the Distance operator

                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B = otf_sub_distmat_only(TrainIDs, TestIDs, MLObjList, LocDescL)
                            LocTrainTens.append(A)
                            A = []
                            LocTestTens.append(B)
                            B = []
      
                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            A, B = otf_sub_eucdistmat_only(TrainIDs, TestIDs, MLObjList, LocDescL)
                            LocTrainTens.append(A)
                            A = []
                            LocTestTens.append(B)
                            B = []

                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            for jj in range(LocArrLen):
                                A, B = otf_sub_distmat_layer_only(TrainIDs, TestIDs, MLObjList, LocDescL, jj)
                                LocTrainTens.append(A)
                                A = []
                                LocTestTens.append(B)
                                B = []

                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            print("On-the-fly calculation of Distance Matrix for descriptor {}...".format(LocDescL))
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            for jj in range(LocArrLen):
                                A, B = otf_sub_eucdistmat_layer_only(TrainIDs, TestIDs, MLObjList, LocDescL, jj)
                                LocTrainTens.append(A)
                                A = []
                                LocTestTens.append(B)
                                B = []

        self.FinTrainTens = dc(LocTrainTens)
        LocTrainTens      = []
        self.FinTestTens  = dc(LocTestTens)
        LocTestTens       = []

        ################################## Descriptor Section over - following the Label Section ######################################
        # For the time being, just assume that Labels are just one value per structure. This can be changed later in a similar
        # DataF / DataO fashion below here.
        ###############################################################################################################################

        for ii in range(len(self.LabelL)):
            LocLabelL = self.LabelL[ii]
            LocList   = self.Labels[LocLabelL]
            LocAux    = []
            for jj in TrainIDs:
                LocAux.append(LocList[jj])
            self.FinTrainLabels[LocLabelL] = dc(LocAux)
            LocAux    = []
            for jj in TestIDs:
                LocAux.append(LocList[jj])
            self.FinTestLabels[LocLabelL]  = dc(LocAux)
        return None

    # This function gets the IDS of Training and Prediction data and generates the distance matrices, decorating them with the loaded, optimized Sigmas on-the-fly to yield the Kernel matrix.
    def get_pred_tens(self, MLObjList, TrainIDs, PredIDs, Sigmas):
        """
        A method that fills a kernel matrix for predictions. Reads in the
        sigma of a previously finished and saved model.
        
        --- Parameters ---
        ------------------
        MLObjList : list of MLEntry objects
            The list of all MLEntries after adding the descriptors.

        TrainIDs : list of int
            A list of the _numerical pointers_ for picking out the MLEntries from
            the MLObjList that belong to the training data.

        PredIDs : list of int
            A list of the _numerical pointers_ for picking out the MLEntries from
            the MLObjList that belong to the prediction data.

        Sigmas : list of float
            A list of floats for the final sigma values after hyperparameter
            optimization. These are required to decorate the individual
            distance matrices and ultimately form the kernel matrix.

        --- Returns ---
        ---------------
        np.exp(-Aux)
            This operation finalizes the kernel matrix from the sum of decorated
            distance matrices contained in Aux.
        """
        # for each Descriptor, form the local, submatrix of the overall Distance Matrix and generate its local sigmas
        LocPredTens   = []
        curLayer      = 0
        Aux = np.zeros((len(PredIDs), len(TrainIDs)))
        for ii in self.DescL:
            if self.DescType[ii] != "unused":
                LocDescL = ii
                LocDataF = self.DescType[ii][0]                                         # Get the Data Format
                LocDataO = self.DescType[ii][1]                                         # Get the Distance operator

                # Scalar - DistMat
                if LocDataF == "Scalar":
                    if LocDataO == "DistMat":
                        print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                        Aux += calc_pred_distmat(MLObjList, LocDescL, TrainIDs, PredIDs, Sigmas[curLayer])
                        curLayer += 1

                # EigCoulVec - EucDistMat
                if LocDataF == "EigCoulVec":
                    if LocDataO == "EucDistMat":
                        print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                        Aux += calc_pred_eucdistmat(MLObjList, LocDescL, TrainIDs, PredIDs, Sigmas[curLayer])
                        curLayer += 1

                # ArrOfScal - DistMat
                if LocDataF == "ArrOfScal":
                    if LocDataO == "DistMat":
                        print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                        LocArrLen = len(MLObjList[0].Desc[LocDescL])
                        for jj in range(LocArrLen):
                            Aux += calc_pred_distmat_layer(MLObjList, LocDescL, jj, TrainIDs, PredIDs, Sigmas[curLayer])
                            curLayer +=1

                # ArrOfArr - EucDistMat
                if LocDataF == "ArrOfArr":
                    if LocDataO == "EucDistMat":
                        print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                        LocArrLen = len(MLObjList[0].Desc[LocDescL])
                        for jj in range(LocArrLen):
                            Aux += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, TrainIDs, PredIDs, Sigmas[curLayer]) 
                            curLayer +=1
        return np.exp(-Aux)
