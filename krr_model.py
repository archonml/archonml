# # # # # # # # # # # # #
# The MLModel class will take Tensors from a DataTens Object (and it's SIGMA slices) to train a KRR model.
# It does not need any information about the length or meaning of individual Descriptors, since it will just use the summed, sigma-scaled exponential Distance matrices.
#
# Initialization loads the Model with a predictor target. This will then take the desired Label data from the overall DataTens-Objects.
# Goal is, to have singular, parallelized cross-validation cycles that will alter the ___object-specific___ sigma and lambda vectors.
# (This way, you can define S1Model = MLModel() - and have a compact model for each label)
#
# FW 07/13/23

# sklearn Related
import os
import time
from copy import deepcopy as dc

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression as pearsonr

from .entries import MLEntry
from .desctools import Descriptor
from .utils import kernel_gen
from .utils import otf_distmat, otf_eucdistmat
from .utils import otf_distmat_layer, otf_eucdistmat_layer
from .utils import calc_pred_distmat, calc_pred_eucdistmat
from .utils import calc_pred_distmat_layer, calc_pred_eucdistmat_layer

class MLModel:
    """
    An object for training, saving and loading a Kernel Ridge Regression Model. Using a finished model,
    it can also perform the predictions. Will update itself during the cross-validation steps - until a
    finalization call is made which wraps up the training and compares the resulting model with the
    testing data that was kept exclusive for finalization.

    Note, that in it's current form, the object will train two models at the same time - one that
    maximizes either the coefficient of determination (R2) or Pearson Correlation (pR, pearsonr, Pearson)
    and one that minimizes the mean absolute error (MAE). Thus, several attributes are doubled, at the
    moment - but only explained once below.

    --- Attributes ---
    ------------------
    Target : str

    CorrScore : str (anticipating "R2" or "Pearson")
        A string that determines whether the model will train/evaluate versus R2 or Pearson correlation.

    MCurSigmas, RCurSigmas : list of arrays of float
        A list that collects the array of optimized sigma values after each cross-validation step.

    MCurLambdas, RCurLambdas : list of floats
        A list that collects all optimized lambda values after each cross-validation step.

    FinSigmas_M, FinSigmas_R : array of float
        The final list of sigma values after cross-validation. This is to be stored later.

    FinLambda_M, FinLambda_R : float
        The final lambda value after cross-validation. This is to be stored later.

    FinR2_M, FinR2_R : float
        Stores the final model test performance in terms of R2.

    FinR2_MT, FinR2_RT : float
        Stores the final model training performance in terms of R2.

    FinMAE_M, FinMAE_R : float
        Stores the final model test performance in terms of MAE.

    FinMAE_MT, FinMAE_RT : float
        Stores the final model training performance in terms of MAE.

    FinModel_M, FinModel_R : sklearn objects
        Stores the final model for saving.

    FinTrainNames : list of str
        List of the unique entry IDs used during training (i.e. "PROJ123"). Is saved along the model,
        as these need to be reloaded for later predictions in the _exact same order_.

    FinTrainFPrints : list of str
        List of the fingerprints of the training set. Saved along the model, as it is useful for
        on-the-fly restoration of the data tensors (i.e. used to save memory in predictions)

    FinDescL : list of str
        List of the final descriptors (and their ordering). Is saved along the model to restore the
        exact data tensors in their exact ordering.

    FinDBSpecs : list of list of str
        A list containing the database specifications, stored along the final model  as to locate
        all merged data during predictions.

    LoadSigmas : array of float
        An array containing the loaded sigma for preparing the prediction kernel matrix.

    LoadLamba : float
        A float value for the loaded lambda of the current model. Not actually used - but
        added for completeness.

    LoadModel : sklearn object
        The model loaded from an earlier save. Contains the lambda - but not the sigma for
        creating the prediction kernel matrix.

    Predictions : array of float
        An array containing the prediction values as determined from applying the loaded
        model to the prediction kernel matrix.

    --- Methods ---
    ---------------
    correval(self, LabelVal, PredVal)
        A method that returns a correlation value - currently either uses R2 or Pearson coefficient.
        (returns float)

    cv_para(self, DataTensObj, MLObjList=[], optimizer="vectorized", convR=0.005, convM=0.0001, MaxIter=100, ignoreConv=False, useGUI=True)
        A method to initialize one round of hyperparameter screening in parallelized cross-validation fashion. Features
        several basic optimizer options to reduce computational costs. This will automatically update the current list
        of optimized hyperparameters.
        (returns None)

    final_cv(self, DataTensObj, MLObjList=[])
        A method to finalize the hyperparameter optimization and compare the test set's "true" values versus "predicted" values.
        A learning curve may be constructed, when additionally controlling the amount of active training data, as explained
        in the Data Tensor class. Will store the finalized model to the instance itself.
        (returns None)

    save_model(self, iName, Loss="R2", Method="Auto")
        A method to save the finalized model, along other vital information about the database, as to run predictions later.
        Will store the model inside the provided ModelPath.
        (returns None)

    get_model(self, iName)
        A method that loads a previously saved model, initializes a descriptor instance and collects the reference training data
        required to build the prediction kernel matrix.
        (returns LocObjList, LocDescInst)

    predict_from_loaded(self, DataTensObj, MLObjList)
        A method that predicts the data contained inside the Data Tensor Object. Uses the MLObjList to calculate the kernel matrix
        on-the-fly (i.e. more memory efficient). Stores the output inside the MLModel instance itself.
        (returns None)
    """
    # Global Path to store/load Models from
    ModelPath = "../MLModels/"                          # Path for storing trained Models and training specifics.
    # Note, that the "M" and "R" Suffixes and Prefixes in some places point towards the two trained models - i.e. when training against MAE as loss function or R2 as loss function.
    def __init__(self, iLabel, iCorr):
        """
        --- Parameters ---
        ------------------
        iLabel : str
            A string that contains the target label to train for. This needs to be identical to the
            so-far used label names when loading the data.
        """
        self.Target      = iLabel                       # One Model Object per Target Value to predict
        self.CorrScore   = iCorr

        # Placeholders for during training.
        self.MCurSigmas  = []                           # Stores the SIGMA_MAT vector that performed best at each best Sigma Pointer (MAE training).
        self.MCurLambdas = []                           # Stores the SIGMA_MAT vector that performed best at each best Sigma Pointer (MAE training).
        self.RCurSigmas  = []                           # Stores the SIGMA_MAT vector that performed best at each best Sigma Pointer (R training).
        self.RCurLambdas = []                           # Stores the Lambda value that performed best at each best Sigma Pointer (R training).

        # Placeholders for parameters of the model finalization.
        self.FinSigmas_M = []                           # Stores final Sigmas after CV which are used in the testing for the MAE trained model.
        self.FinLambda_M = []                           # Stores final Lambda after CV which are used in the testing for the MAE trained model.
        self.FinSigmas_R = []                           # Stores final Sigmas after CV which are used in the testing for the R trained model.
        self.FinLambda_R = []                           # Stores final Lambda after CV which are used in the testing for the R trained model.
        self.FinR2_M     = []                           # Stores the R2 score of the testing with the MAE trained model.
        self.FinR2_MT    = []                           # Stores the R2 score of the training with the MAE trained model.
        self.FinpR_M     = []                           # Stores the pearson R score of the testing with the MAE model.
        self.FinpR_MT    = []                           # Stores the pearson R score of the training with the MAE model.
        self.FinMAE_M    = []                           # Stores the MAE score of the testing with the MAE trained model.
        self.FinMAE_MT   = []                           # Stores the MAE score of the training with the MAE trained model.
        self.FinR2_R     = []                           # Stores the R2 score of the testing with the R trained model.
        self.FinR2_RT    = []                           # Stores the R2 score of the training with the R trained model.
        self.FinpR_R     = []                           # Stores the pearson R score of the testing with the R model.
        self.FinpR_RT    = []                           # Stores the pearson R score of the training with the R model.
        self.FinMAE_R    = []                           # Stores the MAE score of the testing with the R trained model.
        self.FinMAE_RT   = []                           # Stores the MAE score of the training with the R trained model.
        self.FinModel_M  = []                           # Stores Final Model after Training with respect to MAE.
        self.FinModel_R  = []                           # Stores Final Model after Training with respect to R.

        # Placeholders for further final model values.
        self.FinTestLabelVals = []                      # Label Values used in the last finalization.
        self.FinPredMVals     = []                      # Label Values as predicted by the MAE Model.
        self.FinPredRVals     = []                      # Label Values as predicted by the R model.


        # Placeholders for final model meta-data. This is crucial for saving a reproducible model.
        # Note - these are "freshly" passed on by a DataTens-Object during the final testing - but _could_ also be stored in a shared file, ultimately...
        self.FinTrainNames   = []                       # Stores the names of the Final training set in non-numerical form for later storing and loading.
        self.FinTrainFPrints = []                       # Stores the Fingerprints of the Final training set in non-numerical form for later storing and loading.
        self.FinDescL        = []                       # Stores the names of the Final Descriptors.
        self.FinDBSpecs      = []                       # Stores the DBSpecs of the DataTensObject for storing / loading.

        # Placeholders for loading a pre-trained model and performing predictions.
        self.LoadSigmas  = []                           # When loading any model, place Sigmas here
        self.LoadLamba   = []                           # When loading any model, place Lambda here
        self.LoadModel   = []                           # When loading any model, place the Model itself here
        self.Predictions = []                           # Container for the Predicted values.

    # This method shall serve as a helper function to standardize correlation calculation below.
    def correval(self, LabelVal, PredVal):
        """
        A method that returns a correlation value - currently either uses R2 or Pearson coefficient.
        (returns float)

        --- Parameters ---
        ------------------
        LabelVal : array of float
            An array of the label data in float.

        PredVal : array of float
            An array of the predicted values in float.

        --- Returns ---
        ---------------
        CVal : float
            A value for the correlation coefficient. Number depoends on which mehtod was chosen.

        """

        CVal = 0

        if self.CorrScore == "R2":
            CVal = r2_score(LabelVal, PredVal)

        if self.CorrScore == "Pearson":
            Aux  = np.asarray(LabelVal).reshape(-1, 1)
            with np.errstate(invalid='ignore', divide='ignore'):
                AuxVal = pearsonr(Aux, PredVal)
                CVal = AuxVal[0]
            if np.isnan(CVal):
                CVal = 0.0
            if np.isinf(CVal):
                CVal = 0.0

        return CVal





    # This function runs a hyper-parameter optimization for a specific CV Block.
    def cv_para(self, DataTensObj, MLObjList=[], optimizer="vectorized", convR=0.005, convM=0.0001, MaxIter=100, ignoreConv=False, useGUI=True):
        """
        A method to initialize one round of hyperparameter screening in parallelized cross-validation fashion. Features
        several basic optimizer options to reduce computational costs. This will automatically update the current list
        of optimized hyperparameters.
        (returns None)

        --- Parameters ---
        ------------------
        DataTensObj : data tensor object instance
            Contains the training / prediction data in the required format - including the information which parts are
            (active / inactive) training data for the cross-validation and which parts are testing data.

        MLObjList : list of MLEntry object instances
            A list of the MLEntries.  Providing it is required for medium and low-memory training options, as only then,
            the on-the-fly methods can be applied.

        optimizer : str
            A specification of which optimizer to use during hyperparameter cross-validation. See the documentation for
            all options.

        convR : float
            The convergence criterion with respect to the R2 value. Only used in "descent" optimizer, currently.

        convM : float
            The convergence criterion with respect to the MAE value. Only used in "descent" optimizer, currently.

        MaxIter : int
            Number of maximum iterations of the "descent" optimizer.

        ignoreConv : bool
            A boolean to deactivate convergence checking. Useful for very high-resolution grids that are
            rather shallow everywhere - and in case of very 'flat' hyperparameter landscapes.

        useGUI : bool
            A boolean to save the final picture and model performance to files, rather than printing (to the ipython interface).
        """
        if optimizer == "vectorized":
            # Get the memory option. This determines if Sigma Matrices and Vali/Train Tensors have been pre-calculated or need to be evaluated on-the-fly.
            memmode = DataTensObj.memmode
            AllT    = time.time()
            if memmode == "low":
                if MLObjList==[]:
                    print("In low-memory mode, it is necessary to add the MLObject List to cv_para!")
                print("Low-memory on-the-fly mode was selected.")
                print("Initialising Loops over Hyper-Parameters - with on-the-fly calculation of distance matrices.")
                HypMAEs = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                HypMSEs = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                HypR2SK = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                # Pre-allocate KMatrices for training-only and (Vali x Train) sets.
                AuxK = np.zeros((len(DataTensObj.CVTrainIDs), len(DataTensObj.CVTrainIDs)))
                AuxP = np.zeros((len(DataTensObj.CVValiIDs), len(DataTensObj.CVTrainIDs)))

                # For each nth slice of the sigma vector, generate on the fly each distance matrix, get the sigma range and embed into the auxiliary KMAT.
                for nth_slice in range(DataTensObj.CVGridPts):
                    print(" ...{} slice of the Sigma grid...".format(nth_slice))
                    AuxK = np.zeros((len(DataTensObj.CVTrainIDs), len(DataTensObj.CVTrainIDs)))
                    AuxP = np.zeros((len(DataTensObj.CVValiIDs), len(DataTensObj.CVTrainIDs)))
                    CurDescCnt = 0
                    # Go through the descriptors
                    for ii in DataTensObj.DescL:
                        if DataTensObj.DescType[ii] != "unused":
                            LocDescL = ii
                            LocDataF = DataTensObj.DescType[ii][0]                                         # Get the Data Format
                            LocDataO = DataTensObj.DescType[ii][1]                                         # Get the Distance operator

                            # Scalar - DistMat
                            if LocDataF == "Scalar":
                                if LocDataO == "DistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    Aux, LSig = otf_distmat(MLObjList, DataTensObj, LocDescL, nth_slice)
                                    DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                    AuxK += Aux
                                    Aux = []
                                    AuxP += calc_pred_distmat(MLObjList, LocDescL, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                    CurDescCnt += 1

                            # EigCoulVec - EucDistMat
                            if LocDataF == "EigCoulVec":
                                if LocDataO == "EucDistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    Aux, LSig = otf_eucdistmat(MLObjList, DataTensObj, LocDescL, nth_slice)
                                    DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                    AuxK += Aux
                                    Aux = []
                                    AuxP += calc_pred_eucdistmat(MLObjList, LocDescL, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                    CurDescCnt += 1

                            # ArrOfScal - DistMat
                            if LocDataF == "ArrOfScal":
                                if LocDataO == "DistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    LocArrLen = len(MLObjList[0].Desc[LocDescL])
                                    for jj in range(LocArrLen):
                                        Aux, LSig = otf_distmat_layer(MLObjList, DataTensObj, LocDescL, jj, nth_slice)
                                        DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                        AuxK += Aux
                                        Aux = []
                                        AuxP += calc_pred_distmat_layer(MLObjList, LocDescL, jj, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                        CurDescCnt += 1

                            # ArrOfArr - EucDistMat
                            if LocDataF == "ArrOfArr":
                                if LocDataO == "EucDistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    LocArrLen = len(MLObjList[0].Desc[LocDescL])
                                    for jj in range(LocArrLen):
                                        Aux, LSig = otf_eucdistmat_layer(MLObjList, DataTensObj, LocDescL, jj, nth_slice)
                                        DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                        AuxK += Aux
                                        Aux = []
                                        AuxP += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                        CurDescCnt += 1

                    KMAT           = np.exp(-AuxK)
                    AuxK           = []
                    KPred          = np.exp(-AuxP)
                    AuxP           = []
                    LocTrainLabels = DataTensObj.TrainLabels[self.Target]
                    LocValiLabels  = DataTensObj.ValiLabels[self.Target]
                    for ii in range(len(DataTensObj.LAMBDA_GRID)):
                        LAMBDA = DataTensObj.LAMBDA_GRID[ii]
                        kr = KernelRidge(kernel="precomputed", alpha=LAMBDA)
                        kr.fit(KMAT, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_kr = kr.predict(KPred)
                        Y_Bar = np.mean(LocValiLabels)
                        HypR2SK[nth_slice, ii] = self.correval(LocValiLabels, y_kr)                          # R value
                        HypMAEs[nth_slice, ii] = sum(abs(y_kr - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                        HypMSEs[nth_slice, ii] = np.mean((y_kr - LocValiLabels)**2)                     # Mean Squared Error

            if (memmode == "high") or (memmode == "medium"):
                print("High-memory or medium-memory mode was selected.")
                AllT = time.time()
                print("Initialising Loops over Hyper-Parameters...")
                HypMAEs = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                HypMSEs = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                HypR2SK = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                for nth_slice in range(len(DataTensObj.SIGMA_MAT[0, :])):
                    # Form Kernel matrices for training and validation
                    LocSigmas      = DataTensObj.SIGMA_MAT[:, nth_slice]
                    KMAT           = kernel_gen(LocSigmas, DataTensObj.TrainTens)
                    KPred          = kernel_gen(LocSigmas, DataTensObj.ValiTens)
                    LocTrainLabels = DataTensObj.TrainLabels[self.Target]
                    LocValiLabels  = DataTensObj.ValiLabels[self.Target]
                    # Remember, the larger LAMBDA becomes, the less sensitive the method becomes to the actual descriptors.
                    for jj in range(len(DataTensObj.LAMBDA_GRID)):
                        LAMBDA = DataTensObj.LAMBDA_GRID[jj]
                        kr = KernelRidge(kernel="precomputed", alpha=LAMBDA)
                        kr.fit(KMAT, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_kr = kr.predict(KPred)
                        Y_Bar = np.mean(LocValiLabels)
                        HypR2SK[nth_slice, jj] = self.correval(LocValiLabels, y_kr)                     # R value
                        HypMAEs[nth_slice, jj] = sum(abs(y_kr - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                        HypMSEs[nth_slice, jj] = np.mean((y_kr - LocValiLabels)**2)                     # Mean Squared Error

            AllTs = time.time() - AllT
            if useGUI == True:
                print("Full Grid done after ", AllTs, "seconds.")
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#                sns.heatmap(np.log(HypR2SK), cmap='jet', ax=axes[0])
                sns.heatmap(HypR2SK, cmap='jet', ax=axes[0])
                axes[0].set_xlabel('LAMBDA')
                axes[0].set_ylabel('SIGMA')
                axes[0].set_title("R values")
                sns.heatmap(np.log(HypMAEs), cmap='jet', ax=axes[1])
                axes[1].set_xlabel('LAMBDA')
                axes[1].set_ylabel('SIGMA')
                axes[1].set_title("log(MAE) values")
                plt.show()
                rescoord = np.where(HypR2SK == np.nanmax(HypR2SK))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.RCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.RCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                print("Best R2 was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(HypR2SK[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                print("MAE at this point is", HypMAEs[SIG_MAX, LAM_MAX])
                print("")
                rescoord = np.where(HypMAEs == np.nanmin(HypMAEs))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.MCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.MCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                print("Best MAE was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(HypMAEs[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                print("R2 at this point is", HypR2SK[SIG_MAX, LAM_MAX])

            if useGUI == False:
                SSS1 = "Full Grid done after {} seconds.".format(AllTs)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#                sns.heatmap(np.log(HypR2SK), cmap='jet', ax=axes[0])
                sns.heatmap(HypR2SK, cmap='jet', ax=axes[0])
                axes[0].set_xlabel('LAMBDA')
                axes[0].set_ylabel('SIGMA')
                axes[0].set_title("R values")
                sns.heatmap(np.log(HypMAEs), cmap='jet', ax=axes[1])
                axes[1].set_xlabel('LAMBDA')
                axes[1].set_ylabel('SIGMA')
                axes[1].set_title("log(MAE) values")
                rescoord = np.where(HypR2SK == np.nanmax(HypR2SK))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.RCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.RCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                SSS2 = "Best R2 was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(HypR2SK[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX)
                SSS3 = "MAE at this point is {}.\n".format(HypMAEs[SIG_MAX, LAM_MAX])
                rescoord = np.where(HypMAEs == np.nanmin(HypMAEs))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.MCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.MCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                SSS4 = "Best MAE was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(HypMAEs[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX)
                SSS5 = "R2 at this point is {}.\n".format(HypR2SK[SIG_MAX, LAM_MAX])
                OutFNam = "CVStats_{}.out".format(len(self.RCurLambdas))
                OutPNam = "CVStats_{}.svg".format(len(self.RCurLambdas))
                OID = open(OutFNam, "w")
                OID.write(SSS1)
                OID.write(SSS2)
                OID.write(SSS3)
                OID.write(SSS4)
                OID.write(SSS5)
                OID.close()
                fig.savefig(OutPNam, format="svg", dpi=300)
                plt.close(fig)

        if optimizer == "mean-field":
            # Do one 2D-run for the "mean-field" local minimum-sigmas.
            # Get the memory option. This determines if Sigma Matrices and Vali/Train Tensors have been pre-calculated or need to be evaluated on-the-fly.
            memmode = DataTensObj.memmode
            AllT    = time.time()
            if memmode == "low":
                if MLObjList==[]:
                    print("In low-memory mode, it is necessary to add the MLObject List to cv_para!")
                    raise InputError
                print("Low-memory on-the-fly mode was selected.")
                print("Initialising Loops over Hyper-Parameters - with on-the-fly calculation of distance matrices.")
                PreMAEs = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                PreMSEs = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                PreR2SK = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                # Pre-allocate KMatrices for training-only and (Vali x Train) sets.
                AuxK = np.zeros((len(DataTensObj.CVTrainIDs), len(DataTensObj.CVTrainIDs)))
                AuxP = np.zeros((len(DataTensObj.CVValiIDs), len(DataTensObj.CVTrainIDs)))

                # For each nth slice of the sigma vector, generate on the fly each distance matrix, get the sigma range and embed into the auxiliary KMAT.
                for nth_slice in range(DataTensObj.CVGridPts):
                    print(" ...{} slice of the Sigma grid...".format(nth_slice))
                    AuxK = np.zeros((len(DataTensObj.CVTrainIDs), len(DataTensObj.CVTrainIDs)))
                    AuxP = np.zeros((len(DataTensObj.CVValiIDs), len(DataTensObj.CVTrainIDs)))
                    CurDescCnt = 0
                    # Go through the descriptors
                    for ii in DataTensObj.DescL:
                        if DataTensObj.DescType[ii] != "unused":
                            LocDescL = ii
                            LocDataF = DataTensObj.DescType[ii][0]                                         # Get the Data Format
                            LocDataO = DataTensObj.DescType[ii][1]                                         # Get the Distance operator

                            # Scalar - DistMat
                            if LocDataF == "Scalar":
                                if LocDataO == "DistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    Aux, LSig = otf_distmat(MLObjList, DataTensObj, LocDescL, nth_slice)
                                    DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                    AuxK += Aux
                                    Aux = []
                                    AuxP += calc_pred_distmat(MLObjList, LocDescL, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                    CurDescCnt += 1

                            # EigCoulVec - EucDistMat
                            if LocDataF == "EigCoulVec":
                                if LocDataO == "EucDistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    Aux, LSig = otf_eucdistmat(MLObjList, DataTensObj, LocDescL, nth_slice)
                                    DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                    AuxK += Aux
                                    Aux = []
                                    AuxP += calc_pred_eucdistmat(MLObjList, LocDescL, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                    CurDescCnt += 1

                            # ArrOfScal - DistMat
                            if LocDataF == "ArrOfScal":
                                if LocDataO == "DistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    LocArrLen = len(MLObjList[0].Desc[LocDescL])
                                    for jj in range(LocArrLen):
                                        Aux, LSig = otf_distmat_layer(MLObjList, DataTensObj, LocDescL, jj, nth_slice)
                                        DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                        AuxK += Aux
                                        Aux = []
                                        AuxP += calc_pred_distmat_layer(MLObjList, LocDescL, jj, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                        CurDescCnt += 1

                            # ArrOfArr - EucDistMat
                            if LocDataF == "ArrOfArr":
                                if LocDataO == "EucDistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    LocArrLen = len(MLObjList[0].Desc[LocDescL])
                                    for jj in range(LocArrLen):
                                        Aux, LSig = otf_eucdistmat_layer(MLObjList, DataTensObj, LocDescL, jj, nth_slice)
                                        DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                        AuxK += Aux
                                        Aux = []
                                        AuxP += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                        CurDescCnt += 1

                    KMAT           = np.exp(-AuxK)
                    AuxK           = []
                    KPred          = np.exp(-AuxP)
                    AuxP           = []
                    LocTrainLabels = DataTensObj.TrainLabels[self.Target]
                    LocValiLabels  = DataTensObj.ValiLabels[self.Target]
                    for ii in range(len(DataTensObj.LAMBDA_GRID)):
                        LAMBDA = DataTensObj.LAMBDA_GRID[ii]
                        kr = KernelRidge(kernel="precomputed", alpha=LAMBDA)
                        kr.fit(KMAT, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_kr = kr.predict(KPred)
                        Y_Bar = np.mean(LocValiLabels)
                        PreR2SK[nth_slice, ii] = self.correval(LocValiLabels, y_kr)                          # R value
                        PreMAEs[nth_slice, ii] = sum(abs(y_kr - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                        PreMSEs[nth_slice, ii] = np.mean((y_kr - LocValiLabels)**2)                     # Mean Squared Error

            if (memmode == "high") or (memmode == "medium"):
                print("High-memory or medium-memory mode was selected.")
                AllT = time.time()
                print("Initialising Loops over Hyper-Parameters...")
                PreMAEs = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                PreMSEs = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                PreR2SK = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                for nth_slice in range(len(DataTensObj.SIGMA_MAT[0, :])):
                    # Form Kernel matrices for training and validation
                    LocSigmas      = DataTensObj.SIGMA_MAT[:, nth_slice]
                    KMAT           = kernel_gen(LocSigmas, DataTensObj.TrainTens)
                    KPred          = kernel_gen(LocSigmas, DataTensObj.ValiTens)
                    LocTrainLabels = DataTensObj.TrainLabels[self.Target]
                    LocValiLabels  = DataTensObj.ValiLabels[self.Target]
                    # Remember, the larger LAMBDA becomes, the less sensitive the method becomes to the actual descriptors.
                    for jj in range(len(DataTensObj.LAMBDA_GRID)):
                        LAMBDA = DataTensObj.LAMBDA_GRID[jj]
                        kr = KernelRidge(kernel="precomputed", alpha=LAMBDA)
                        kr.fit(KMAT, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_kr = kr.predict(KPred)
                        Y_Bar = np.mean(LocValiLabels)
                        PreR2SK[nth_slice, jj] = self.correval(LocValiLabels, y_kr)                          # R value
                        PreMAEs[nth_slice, jj] = sum(abs(y_kr - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                        PreMSEs[nth_slice, jj] = np.mean((y_kr - LocValiLabels)**2)                     # Mean Squared Error

            AllTs = time.time() - AllT
            if useGUI == True:
                print("Full Grid done after ", AllTs, "seconds.")
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#                sns.heatmap(np.log(PreR2SK), cmap='jet', ax=axes[0])
                sns.heatmap(PreR2SK, cmap='jet', ax=axes[0])
                axes[0].set_xlabel('LAMBDA')
                axes[0].set_ylabel('SIGMA')
                axes[0].set_title("R values")
                sns.heatmap(np.log(PreMAEs), cmap='jet', ax=axes[1])
                axes[1].set_xlabel('LAMBDA')
                axes[1].set_ylabel('SIGMA')
                axes[1].set_title("log(MAE) values")
                plt.show()
                rescoord = np.where(PreR2SK == np.nanmax(PreR2SK))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.RCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.RCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                print("Best R2 was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(PreR2SK[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                print("MAE at this point is", PreMAEs[SIG_MAX, LAM_MAX])
                oldR2 = dc(PreR2SK[SIG_MAX, LAM_MAX])
                print("")
                rescoord = np.where(PreMAEs == np.nanmin(PreMAEs))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.MCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.MCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                print("Best MAE was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(PreMAEs[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                print("R2 at this point is", PreR2SK[SIG_MAX, LAM_MAX])
                oldMAE = dc(PreMAEs[SIG_MAX, LAM_MAX])

                print("#############################################################")
                print(" Initial guess done - run individual 1D post-optimizations...")
                print("#############################################################")

                # Grab the "static" sigma-values and lambda as the values from pre-optimization
                LocPreOptSigsR = dc(self.RCurSigmas[-1])
                LocPreOptSigsM = dc(self.MCurSigmas[-1])
                LocPreOptLamR  = dc(self.RCurLambdas[-1])
                LocPreOptLamM  = dc(self.MCurLambdas[-1])

                # Initialize storage vector for "individual best performing ones".
                SigVecStoreR = np.zeros((len(LocPreOptSigsR)))
                SigVecStoreM = np.zeros((len(LocPreOptSigsM)))

                # Initialize performance storages for individual ones.
                PerfStoreR   = np.zeros((len(LocPreOptSigsR)))
                PerfStoreM   = np.zeros((len(LocPreOptSigsM)))

                # Select a sigma to 1D-optimize. Grab it's original start- and end-points and generate its 1D Sigma grid.
                for ii in range(len(LocPreOptSigsR)):
                    LocSigStart = dc(DataTensObj.SIGMA_MAT[ii,  0])
                    LocSigEnd   = dc(DataTensObj.SIGMA_MAT[ii, -1])
                    LocSigGrid  = np.linspace(LocSigStart, LocSigEnd, num=DataTensObj.CVGridPts)

                    # initialize local 1D HypMAEs / HypR2s
                    LocHypMAEsR = np.zeros((DataTensObj.CVGridPts))
                    LocHypR2sR  = np.zeros((DataTensObj.CVGridPts))
                    LocHypMAEsM = np.zeros((DataTensObj.CVGridPts))
                    LocHypR2sM  = np.zeros((DataTensObj.CVGridPts))

                    # run over the selected 1D sigma while keeping all others static
                    for jj in range(len(LocSigGrid)):
                        LocSigR          = dc(LocPreOptSigsR)  # take the static solution
                        LocSigR[ii]      = dc(LocSigGrid[jj])  # replace the currently to-opt value with the one on the grid
                        LocSigM          = dc(LocPreOptSigsM)  # take the static solution
                        LocSigM[ii]      = dc(LocSigGrid[jj])  # replace the one value with the one on the grid
                        # Generate the Kernel matrices.
                        KMAT_R           = kernel_gen(LocSigR, DataTensObj.TrainTens)
                        KPred_R          = kernel_gen(LocSigR, DataTensObj.ValiTens)
                        KMAT_M           = kernel_gen(LocSigM, DataTensObj.TrainTens)
                        KPred_M          = kernel_gen(LocSigM, DataTensObj.ValiTens)
                        
                        krR = KernelRidge(kernel="precomputed", alpha=LocPreOptLamR)
                        krR.fit(KMAT_R, LocTrainLabels)
                        krM = KernelRidge(kernel="precomputed", alpha=LocPreOptLamM)
                        krM.fit(KMAT_M, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_krR = krR.predict(KPred_R)
                        y_krM = krM.predict(KPred_M)
                        LocHypR2sR[jj]  = self.correval(LocValiLabels, y_krR)                          # R value
                        LocHypMAEsM[jj] = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error

                    # Evaluate the best ones and add to the StoreVecs. Do not create sub-branches from the behavior R2-opt or MAE-opt.
                    Loc1DPosR2R = np.argmax(LocHypR2sR)
                    Loc1DPosM2M = np.argmin(LocHypMAEsM)
                    SigVecStoreR[ii] = dc(DataTensObj.SIGMA_MAT[ii, Loc1DPosR2R])
                    SigVecStoreM[ii] = dc(DataTensObj.SIGMA_MAT[ii, Loc1DPosM2M])
                    print("...Individual Opt #{} resulted in R2/MAE of ({: 1.3f} / {: 1.3f}) vs. previous ({: 1.3f} / {: 1.3f}).".format(ii, max(LocHypR2sR), min(LocHypMAEsM), oldR2, oldMAE))
                    PerfStoreR[ii]      = dc(max(LocHypR2sR))
                    PerfStoreM[ii]      = dc(min(LocHypMAEsM))

                # Overwrite the last added CurVecs. Putting this at the end of the upper for loop would make this a stepwise "live-updating" routine.
                # (A live-updating version would be sequence-biased though...)
                
                # Check Performances and Accept / Reject best performing one.
                print("#############################################################")
                print(" Refinement done - checking post-processing results...")
                print("#############################################################")

                # Form the new globally changed ones
                KMAT_R  = kernel_gen(SigVecStoreR, DataTensObj.TrainTens)
                KMAT_M  = kernel_gen(SigVecStoreM, DataTensObj.TrainTens)
                KPred_R = kernel_gen(SigVecStoreR, DataTensObj.ValiTens)
                KPred_M = kernel_gen(SigVecStoreM, DataTensObj.ValiTens)
                krR = KernelRidge(kernel="precomputed", alpha=LocPreOptLamR)
                krR.fit(KMAT_R, LocTrainLabels)
                krM = KernelRidge(kernel="precomputed", alpha=LocPreOptLamM)
                krM.fit(KMAT_M, LocTrainLabels)
                y_krR = krR.predict(KPred_R)
                y_krM = krM.predict(KPred_M)
                FinHypR2R  = self.correval(LocValiLabels, y_krR)                          # R value
                FinHypMAER = sum(abs(y_krR - LocValiLabels))/len(LocValiLabels)
                FinHypR2M  = self.correval(LocValiLabels, y_krM)
                FinHypMAEM = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                print("")
                print("Post-processed R2-Model's R2 was {} - with MAE of {}".format(FinHypR2R, FinHypMAER))
                print("Post-processed MAE-Model's MAE was {} - with R2 of {}".format(FinHypMAEM, FinHypR2M))
                print("")

                CheckValsR = [max(PerfStoreR), FinHypR2R, oldR2]
                CheckValsM = [min(PerfStoreM), FinHypMAEM, oldMAE]

                # Keep the best models
                RCase = np.argmax(CheckValsR)
                MCase = np.argmin(CheckValsM)

                if RCase == 0:
                    print("...storing an individually change model for R2!")
                    IDAltered              = np.argmax(PerfStoreR)
                    FinStoreVec            = dc(LocPreOptSigsR)
                    FinStoreVec[IDAltered] = dc(SigVecStoreR[IDAltered])
                    self.RCurSigmas[-1]    = dc(FinStoreVec)                # overwrite Sigmas
                if RCase == 1:
                    print("...storing the globally changed model for R2!")
                    self.RCurSigmas[-1] = dc(SigVecStoreR)                  # overwrite Sigmas
                if RCase == 2:
                    print("...keeping the old model for R2!")
                    pass                                                    # no overwrite required

                if MCase == 0:
                    print("...storing an individually change model for MAE!")
                    IDAltered              = np.argmax(PerfStoreM)
                    FinStoreVec            = dc(LocPreOptSigsM)
                    FinStoreVec[IDAltered] = dc(SigVecStoreM[IDAltered])
                    self.MCurSigmas[-1]    = dc(FinStoreVec)                # overwrite Sigmas
                if RCase == 1:
                    print("...storing the globally changed model for MAE!")
                    self.MCurSigmas[-1] = dc(SigVecStoreM)                  # overwrite Sigmas
                if RCase == 2:
                    print("...keeping the old model for MAE!")
                    pass                                                    # no overwrite required

            if useGUI == False:
                SSS=[]
                SSS.append("Full Grid done after {} seconds.\n".format(AllTs))
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#                sns.heatmap(np.log(PreR2SK), cmap='jet', ax=axes[0])
                sns.heatmap(PreR2SK, cmap='jet', ax=axes[0])
                axes[0].set_xlabel('LAMBDA')
                axes[0].set_ylabel('SIGMA')
                axes[0].set_title("R values")
                sns.heatmap(np.log(PreMAEs), cmap='jet', ax=axes[1])
                axes[1].set_xlabel('LAMBDA')
                axes[1].set_ylabel('SIGMA')
                axes[1].set_title("log(MAE) values")
                rescoord = np.where(PreR2SK == np.nanmax(PreR2SK))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.RCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.RCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                SSS.append("Best R2 was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})\n".format(PreR2SK[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                SSS.append("MAE at this point is {}\n\n".format(PreMAEs[SIG_MAX, LAM_MAX]))
                oldR2 = dc(PreR2SK[SIG_MAX, LAM_MAX])
                rescoord = np.where(PreMAEs == np.nanmin(PreMAEs))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                self.MCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.MCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                SSS.append("Best MAE was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})\n".format(PreMAEs[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                SSS.append("R2 at this point is {}\n\n".format(PreR2SK[SIG_MAX, LAM_MAX]))
                oldMAE = dc(PreMAEs[SIG_MAX, LAM_MAX])

                SSS.append("#############################################################\n")
                SSS.append(" Initial guess done - run individual 1D post-optimizations...\n")
                SSS.append("#############################################################\n\n")

                # Grab the "static" sigma-values and lambda as the values from pre-optimization
                LocPreOptSigsR = dc(self.RCurSigmas[-1])
                LocPreOptSigsM = dc(self.MCurSigmas[-1])
                LocPreOptLamR  = dc(self.RCurLambdas[-1])
                LocPreOptLamM  = dc(self.MCurLambdas[-1])

                # Initialize storage vector for "individual best performing ones".
                SigVecStoreR = np.zeros((len(LocPreOptSigsR)))
                SigVecStoreM = np.zeros((len(LocPreOptSigsM)))

                # Initialize performance storages for individual ones.
                PerfStoreR   = np.zeros((len(LocPreOptSigsR)))
                PerfStoreM   = np.zeros((len(LocPreOptSigsM)))

                # Select a sigma to 1D-optimize. Grab it's original start- and end-points and generate its 1D Sigma grid.
                for ii in range(len(LocPreOptSigsR)):
                    LocSigStart = dc(DataTensObj.SIGMA_MAT[ii,  0])
                    LocSigEnd   = dc(DataTensObj.SIGMA_MAT[ii, -1])
                    LocSigGrid  = np.linspace(LocSigStart, LocSigEnd, num=DataTensObj.CVGridPts)

                    # initialize local 1D HypMAEs / HypR2s
                    LocHypMAEsR = np.zeros((DataTensObj.CVGridPts))
                    LocHypR2sR  = np.zeros((DataTensObj.CVGridPts))
                    LocHypMAEsM = np.zeros((DataTensObj.CVGridPts))
                    LocHypR2sM  = np.zeros((DataTensObj.CVGridPts))

                    # run over the selected 1D sigma while keeping all others static
                    for jj in range(len(LocSigGrid)):
                        LocSigR          = dc(LocPreOptSigsR)  # take the static solution
                        LocSigR[ii]      = dc(LocSigGrid[jj])  # replace the currently to-opt value with the one on the grid
                        LocSigM          = dc(LocPreOptSigsM)  # take the static solution
                        LocSigM[ii]      = dc(LocSigGrid[jj])  # replace the one value with the one on the grid
                        # Generate the Kernel matrices.
                        KMAT_R           = kernel_gen(LocSigR, DataTensObj.TrainTens)
                        KPred_R          = kernel_gen(LocSigR, DataTensObj.ValiTens)
                        KMAT_M           = kernel_gen(LocSigM, DataTensObj.TrainTens)
                        KPred_M          = kernel_gen(LocSigM, DataTensObj.ValiTens)
                        
                        krR = KernelRidge(kernel="precomputed", alpha=LocPreOptLamR)
                        krR.fit(KMAT_R, LocTrainLabels)
                        krM = KernelRidge(kernel="precomputed", alpha=LocPreOptLamM)
                        krM.fit(KMAT_M, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_krR = krR.predict(KPred_R)
                        y_krM = krM.predict(KPred_M)
                        LocHypR2sR[jj]  = self.correval(LocValiLabels, y_krR)                          # R value
                        LocHypMAEsM[jj] = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error

                    # Evaluate the best ones and add to the StoreVecs. Do not create sub-branches from the behavior R2-opt or MAE-opt.
                    Loc1DPosR2R = np.argmax(LocHypR2sR)
                    Loc1DPosM2M = np.argmin(LocHypMAEsM)
                    SigVecStoreR[ii] = dc(DataTensObj.SIGMA_MAT[ii, Loc1DPosR2R])
                    SigVecStoreM[ii] = dc(DataTensObj.SIGMA_MAT[ii, Loc1DPosM2M])
                    SSS.append("...Individual Opt #{} resulted in R2/MAE of ({: 1.3f} / {: 1.3f}) vs. previous ({: 1.3f} / {: 1.3f}).\n".format(ii, max(LocHypR2sR), min(LocHypMAEsM), oldR2, oldMAE))
                    PerfStoreR[ii]      = dc(max(LocHypR2sR))
                    PerfStoreM[ii]      = dc(min(LocHypMAEsM))

                # Overwrite the last added CurVecs. Putting this at the end of the upper for loop would make this a stepwise "live-updating" routine.
                # (A live-updating version would be sequence-biased though...)
                
                # Check Performances and Accept / Reject best performing one.
                SSS.append("#############################################################\n")
                SSS.append(" Refinement done - checking post-processing results...\n")
                SSS.append("#############################################################\n\n")

                # Form the new globally changed ones
                KMAT_R  = kernel_gen(SigVecStoreR, DataTensObj.TrainTens)
                KMAT_M  = kernel_gen(SigVecStoreM, DataTensObj.TrainTens)
                KPred_R = kernel_gen(SigVecStoreR, DataTensObj.ValiTens)
                KPred_M = kernel_gen(SigVecStoreM, DataTensObj.ValiTens)
                krR = KernelRidge(kernel="precomputed", alpha=LocPreOptLamR)
                krR.fit(KMAT_R, LocTrainLabels)
                krM = KernelRidge(kernel="precomputed", alpha=LocPreOptLamM)
                krM.fit(KMAT_M, LocTrainLabels)
                y_krR = krR.predict(KPred_R)
                y_krM = krM.predict(KPred_M)
                FinHypR2R  = self.correval(LocValiLabels, y_krR)                          # R value
                FinHypMAER = sum(abs(y_krR - LocValiLabels))/len(LocValiLabels)
                FinHypR2M  = self.correval(LocValiLabels, y_krM)
                FinHypMAEM = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                SSS.append("Post-processed R2-Model's R2 was {} - with MAE of {}\n".format(FinHypR2R, FinHypMAER))
                SSS.append("Post-processed MAE-Model's MAE was {} - with R2 of {}\n\n".format(FinHypMAEM, FinHypR2M))

                CheckValsR = [max(PerfStoreR), FinHypR2R, oldR2]
                CheckValsM = [min(PerfStoreM), FinHypMAEM, oldMAE]

                # Keep the best models
                RCase = np.argmax(CheckValsR)
                MCase = np.argmin(CheckValsM)

                if RCase == 0:
                    SSS.append("...storing an individually change model for R2!\n")
                    IDAltered              = np.argmax(PerfStoreR)
                    FinStoreVec            = dc(LocPreOptSigsR)
                    FinStoreVec[IDAltered] = dc(SigVecStoreR[IDAltered])
                    self.RCurSigmas[-1]    = dc(FinStoreVec)                # overwrite Sigmas
                if RCase == 1:
                    SSS.append("...storing the globally changed model for R2!\n")
                    self.RCurSigmas[-1] = dc(SigVecStoreR)                  # overwrite Sigmas
                if RCase == 2:
                    SSS.append("...keeping the old model for R2!\n")
                    pass                                                    # no overwrite required

                if MCase == 0:
                    SSS.append("...storing an individually change model for MAE!\n")
                    IDAltered              = np.argmax(PerfStoreM)
                    FinStoreVec            = dc(LocPreOptSigsM)
                    FinStoreVec[IDAltered] = dc(SigVecStoreM[IDAltered])
                    self.MCurSigmas[-1]    = dc(FinStoreVec)                # overwrite Sigmas
                if RCase == 1:
                    SSS.append("...storing the globally changed model for MAE!\n")
                    self.MCurSigmas[-1] = dc(SigVecStoreM)                  # overwrite Sigmas
                if RCase == 2:
                    SSS.append("...keeping the old model for MAE!\n")
                    pass                                                    # no overwrite required
                # Write everything to the files
                OutFNam = "CVStats_{}.out".format(len(self.RCurLambdas))
                OutPNam = "CVStats_{}.svg".format(len(self.RCurLambdas))
                OID = open(OutFNam, "w")
                for ii in range(len(SSS)):
                    OID.write(SSS[ii])
                OID.close()
                fig.savefig(OutPNam, format="svg", dpi=300)
                plt.close(fig)



        if optimizer == "descent":
            # Do one 2D-run for the initial, local minimum-sigmas.
            # IMPORTANT : Store the optimal starting Sigma-Positions on the grid!
            # Get the memory option. This determines if Sigma Matrices and Vali/Train Tensors have been pre-calculated or need to be evaluated on-the-fly.
            memmode = DataTensObj.memmode
            AllT    = time.time()
            if memmode == "low":
                if MLObjList==[]:
                    print("In low-memory mode, it is necessary to add the MLObject List to cv_para!")
                    raise InputError
                print("Low-memory on-the-fly mode was selected.")
                print("Initialising Loops over Hyper-Parameters - with on-the-fly calculation of distance matrices.")
                PreMAEs = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                PreMSEs = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                PreR2SK = np.zeros((DataTensObj.CVGridPts, DataTensObj.CVGridPts))
                # Pre-allocate KMatrices for training-only and (Vali x Train) sets.
                AuxK = np.zeros((len(DataTensObj.CVTrainIDs), len(DataTensObj.CVTrainIDs)))
                AuxP = np.zeros((len(DataTensObj.CVValiIDs), len(DataTensObj.CVTrainIDs)))

                # For each nth slice of the sigma vector, generate on the fly each distance matrix, get the sigma range and embed into the auxiliary KMAT.
                for nth_slice in range(DataTensObj.CVGridPts):
                    print(" ...{} slice of the Sigma grid...".format(nth_slice))
                    AuxK = np.zeros((len(DataTensObj.CVTrainIDs), len(DataTensObj.CVTrainIDs)))
                    AuxP = np.zeros((len(DataTensObj.CVValiIDs), len(DataTensObj.CVTrainIDs)))
                    CurDescCnt = 0
                    # Go through the descriptors
                    for ii in DataTensObj.DescL:
                        if DataTensObj.DescType[ii] != "unused":
                            LocDescL = ii
                            LocDataF = DataTensObj.DescType[ii][0]                                         # Get the Data Format
                            LocDataO = DataTensObj.DescType[ii][1]                                         # Get the Distance operator

                            # Scalar - DistMat
                            if LocDataF == "Scalar":
                                if LocDataO == "DistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    Aux, LSig = otf_distmat(MLObjList, DataTensObj, LocDescL, nth_slice)
                                    DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                    AuxK += Aux
                                    Aux = []
                                    AuxP += calc_pred_distmat(MLObjList, LocDescL, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                    CurDescCnt += 1

                            # EigCoulVec - EucDistMat
                            if LocDataF == "EigCoulVec":
                                if LocDataO == "EucDistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    Aux, LSig = otf_eucdistmat(MLObjList, DataTensObj, LocDescL, nth_slice)
                                    DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                    AuxK += Aux
                                    Aux = []
                                    AuxP += calc_pred_eucdistmat(MLObjList, LocDescL, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                    CurDescCnt += 1

                            # ArrOfScal - DistMat
                            if LocDataF == "ArrOfScal":
                                if LocDataO == "DistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    LocArrLen = len(MLObjList[0].Desc[LocDescL])
                                    for jj in range(LocArrLen):
                                        Aux, LSig = otf_distmat_layer(MLObjList, DataTensObj, LocDescL, jj, nth_slice)
                                        DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                        AuxK += Aux
                                        Aux = []
                                        AuxP += calc_pred_distmat_layer(MLObjList, LocDescL, jj, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                        CurDescCnt += 1

                            # ArrOfArr - EucDistMat
                            if LocDataF == "ArrOfArr":
                                if LocDataO == "EucDistMat":
                                    #print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                                    LocArrLen = len(MLObjList[0].Desc[LocDescL])
                                    for jj in range(LocArrLen):
                                        Aux, LSig = otf_eucdistmat_layer(MLObjList, DataTensObj, LocDescL, jj, nth_slice)
                                        DataTensObj.SIGMA_MAT[CurDescCnt, nth_slice] = LSig
                                        AuxK += Aux
                                        Aux = []
                                        AuxP += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, DataTensObj.CVTrainIDs, DataTensObj.CVValiIDs, LSig)
                                        CurDescCnt += 1

                    KMAT           = np.exp(-AuxK)
                    AuxK           = []
                    KPred          = np.exp(-AuxP)
                    AuxP           = []
                    LocTrainLabels = DataTensObj.TrainLabels[self.Target]
                    LocValiLabels  = DataTensObj.ValiLabels[self.Target]
                    for ii in range(len(DataTensObj.LAMBDA_GRID)):
                        LAMBDA = DataTensObj.LAMBDA_GRID[ii]
                        kr = KernelRidge(kernel="precomputed", alpha=LAMBDA)
                        kr.fit(KMAT, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_kr = kr.predict(KPred)
                        Y_Bar = np.mean(LocValiLabels)
                        PreR2SK[nth_slice, ii] = self.correval(LocValiLabels, y_kr)                          # R value
                        PreMAEs[nth_slice, ii] = sum(abs(y_kr - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                        PreMSEs[nth_slice, ii] = np.mean((y_kr - LocValiLabels)**2)                     # Mean Squared Error

            if (memmode == "high") or (memmode == "medium"):
                print("High-memory or medium-memory mode was selected.")
                AllT = time.time()
                print("Initialising Loops over Hyper-Parameters...")
                PreMAEs = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                PreMSEs = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                PreR2SK = np.zeros((len(DataTensObj.SIGMA_MAT[0, :]), len(DataTensObj.LAMBDA_GRID)))
                for nth_slice in range(len(DataTensObj.SIGMA_MAT[0, :])):
                    # Form Kernel matrices for training and validation
                    LocSigmas      = DataTensObj.SIGMA_MAT[:, nth_slice]
                    KMAT           = kernel_gen(LocSigmas, DataTensObj.TrainTens)
                    KPred          = kernel_gen(LocSigmas, DataTensObj.ValiTens)
                    LocTrainLabels = DataTensObj.TrainLabels[self.Target]
                    LocValiLabels  = DataTensObj.ValiLabels[self.Target]
                    # Remember, the larger LAMBDA becomes, the less sensitive the method becomes to the actual descriptors.
                    for jj in range(len(DataTensObj.LAMBDA_GRID)):
                        LAMBDA = DataTensObj.LAMBDA_GRID[jj]
                        kr = KernelRidge(kernel="precomputed", alpha=LAMBDA)
                        kr.fit(KMAT, LocTrainLabels)
                        # Make predictions based on the validation set and store metrics on model performance
                        y_kr = kr.predict(KPred)
                        Y_Bar = np.mean(LocValiLabels)
                        PreR2SK[nth_slice, jj] = self.correval(LocValiLabels, y_kr)                     # R value
                        PreMAEs[nth_slice, jj] = sum(abs(y_kr - LocValiLabels))/len(LocValiLabels)      # Mean Absolute Error
                        PreMSEs[nth_slice, jj] = np.mean((y_kr - LocValiLabels)**2)                     # Mean Squared Error

            AllTs = time.time() - AllT
            if useGUI == True:
                print("Full Grid done after ", AllTs, "seconds.")
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#                sns.heatmap(np.log(PreR2SK), cmap='jet', ax=axes[0])
                sns.heatmap(PreR2SK, cmap='jet', ax=axes[0])
                axes[0].set_xlabel('LAMBDA')
                axes[0].set_ylabel('SIGMA')
                axes[0].set_title("R values")
                sns.heatmap(np.log(PreMAEs), cmap='jet', ax=axes[1])
                axes[1].set_xlabel('LAMBDA')
                axes[1].set_ylabel('SIGMA')
                axes[1].set_title("log(MAE) values")
                plt.show()
                rescoord = np.where(PreR2SK == np.nanmax(PreR2SK))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                SigMaxPosR = dc(SIG_MAX)
                LamMaxPosR = dc(LAM_MAX)
                self.RCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.RCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                print("Best R2 was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(PreR2SK[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                print("MAE at this point is", PreMAEs[SIG_MAX, LAM_MAX])
                oldR2 = dc(PreR2SK[SIG_MAX, LAM_MAX])
                print("")
                rescoord = np.where(PreMAEs == np.nanmin(PreMAEs))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                SigMaxPosM = dc(SIG_MAX)
                LamMaxPosM = dc(LAM_MAX)
                self.MCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.MCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                print("Best MAE was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})".format(PreMAEs[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                print("R2 at this point is", PreR2SK[SIG_MAX, LAM_MAX])
                oldMAE = dc(PreMAEs[SIG_MAX, LAM_MAX])

                print("#############################################################")
                print(" Initial guess done - run iterative 1D post-optimizations... ")
                print("#############################################################")

                # Grab the "static" sigma-values and lambda as the values from pre-optimization. Pack both sigma and lambda inside the same scan.

                # R-MODEL
                # R-MODEL
                # R-MODEL
                UpdatingPosR  = [SigMaxPosR]*len(DataTensObj.SIGMA_MAT) + [LamMaxPosR]   # place the initial positions of all dimensions on the grid (R2).
                UpdatingPerfR = dc(oldR2)                                                # place the latest peformance of the model here.
                RConverged = False
                CurLoopCnt = 0
                # WHILE STARTS HERE
                print("# Entering steepest-ascent optimization for the R2-Model...")
                while RConverged == False:
                    print("## Current Loop Number {}. Best R2-Model's R2 so far: {}".format(CurLoopCnt, UpdatingPerfR))
                    # Flush the current TrialPosVector list
                    TrialPosR = []
                    TrialPerf = []
                    # Individual directions start here.
                    for ii in range(len(UpdatingPosR)):
                        # Perform the "decrease" operation, if possible.
                        if UpdatingPosR[ii] > 0:
                            # Copy most recent accepted Vector
                            LocPosR = dc(UpdatingPosR)
                            # Decrease the trial position of the ii-th dimension
                            LocPosR[ii] -= 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosR[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosR[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_R           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_R          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krR = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krR.fit(KMAT_R, LocTrainLabels)
                            y_krR = krR.predict(KPred_R)
                            TrialR2R  = self.correval(LocValiLabels, y_krR)
                            TrialPosR.append(LocPosR)
                            TrialPerf.append(TrialR2R)

                        # Perform the "increase" operation if possible. (Possible, if currently smaller than GridPts-2, as Pts can only be accessed until Lambda[GridPts-1])
                        if UpdatingPosR[ii] < (DataTensObj.CVGridPts-2):
                            # Copy most recent accepted Vector
                            LocPosR = dc(UpdatingPosR)
                            # Increase the trial position of the ii-th dimension
                            LocPosR[ii] += 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosR[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosR[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_R           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_R          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krR = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krR.fit(KMAT_R, LocTrainLabels)
                            y_krR = krR.predict(KPred_R)
                            TrialR2R  = self.correval(LocValiLabels, y_krR)
                            TrialPosR.append(LocPosR)
                            TrialPerf.append(TrialR2R)

                    # Exiting the Trial-Generator-Loop now.
                    # Grab the Trial that had the best performance.
                    #print(TrialPerf)
#                    CurBestID   = np.argmax(TrialPerf)
                    CurBestID   = np.where(np.isinf(TrialPerf), -np.Inf, TrialPerf).argmax()
                    #print(CurBestID)
                    #print(CurBestPerf)
                    CurBestPos  = dc(TrialPosR[CurBestID])
                    CurBestPerf = dc(TrialPerf[CurBestID])
#                    print("DEBUGGING")
#                    print(TrialPerf)
#                    for ii in range(len(TrialPerf)):
#                        print("Value {} is of type {}".format(TrialPerf[ii], type(TrialPerf[ii])))
#                        if TrialPerf[ii] == np.nan:
#                            print("This is a == np.nan case")
#                        if np.isnan(TrialPerf[ii]):
#                            print("isnan worked")
#                    print("DEBUGGING")

                    # Check convergence condition
                    if CurBestPerf > UpdatingPerfR:                                                              # Is it better, than before?
#                        print("Is better than before! {} > {}".format(CurBestPerf, UpdatingPerfR))
                        if  ((CurBestPerf - UpdatingPerfR) < convR) and (ignoreConv == False):                                               # Yes - and fulfils convergence criterion.
                            # Accept the model and break out of the while loop.
                            FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                            FinLamPos            = dc(CurBestPos[-1])
                            FinSigs              = np.zeros((len(FinSigPos)))
                            for jj in range(len(FinSigPos)):
                                FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                            self.RCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                            self.RCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                            RConverged = True
                            print("## Steepest Descent CONVERGED!")
                            print("## Final R2 Performance : {}!".format(CurBestPerf))

                        else:                                                                                    # Yes - but not yet converged!
                            # Update the Reference Positions with the optimized ones and go for the next loop.
                            UpdatingPosR  = dc(CurBestPos)
                            UpdatingPerfR = dc(CurBestPerf)
                            CurLoopCnt += 1
                            if CurLoopCnt > MaxIter:                                                              # if more than acceptable number of steps, signal convergence.
                                FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                                FinLamPos            = dc(CurBestPos[-1])
                                FinSigs              = np.zeros((len(FinSigPos)))
                                for jj in range(len(FinSigPos)):
                                    FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                                self.RCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                                self.RCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                                RConverged = True
                                print("## Steepest Descent RAN OUT OF STEPS!")
                                print("## Final R2 Performance : {}!".format(CurBestPerf))

                    else:                                                                                        # No - none of the current trials were better than before.
                        # Signal convergence and accept current Updating Positions.
                        FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                        FinLamPos            = dc(CurBestPos[-1])
                        FinSigs              = np.zeros((len(FinSigPos)))
                        for jj in range(len(FinSigPos)):
                            FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                        self.RCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                        self.RCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                        RConverged = True
                        print("## Steepest Descent CONVERGED!")
                        print("## Final R2 Performance : {}!".format(CurBestPerf))



                # M-MODEL
                # M-MODEL
                # M-MODEL
                print("")
                UpdatingPosM  = [SigMaxPosM]*len(DataTensObj.SIGMA_MAT) + [LamMaxPosM]   # place the initial positions of all dimensions on the grid (MAE).
                UpdatingPerfM = dc(oldMAE)                                              # place the latest perfomance of the model here.
                MConverged = False
                CurLoopCnt = 0
                # WHILE STARTS HERE
                print("# Entering steepest-descent optimization for the MAE-Model...")
                while MConverged == False:
                    print("## Current Loop Number {}. Best MAE-Model's MAE so far: {}".format(CurLoopCnt, UpdatingPerfM))
                    # Flush the current TrialPosVector list
                    TrialPosM = []
                    TrialPerf = []
                    # Individual directions start here.
                    for ii in range(len(UpdatingPosM)):
                        # Perform the "decrease" operation, if possible.
                        if UpdatingPosM[ii] > 0:
                            # Copy most recent accepted Vector
                            LocPosM = dc(UpdatingPosM)
                            # Decrease the trial position of the ii-th dimension
                            LocPosM[ii] -= 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosM[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosM[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_M           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_M          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krM = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krM.fit(KMAT_M, LocTrainLabels)
                            y_krM = krM.predict(KPred_M)
                            TrialMAEM  = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)
                            TrialPosM.append(LocPosM)
                            TrialPerf.append(TrialMAEM)

                        # Perform the "increase" operation if possible. (Possible, if currently smaller than GridPts-2, as Pts can only be accessed until Lambda[GridPts-1])
                        if UpdatingPosM[ii] < (DataTensObj.CVGridPts-2):
                            # Copy most recent accepted Vector
                            LocPosM = dc(UpdatingPosM)
                            # Increase the trial position of the ii-th dimension
                            LocPosM[ii] += 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosM[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosM[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_M           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_M          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krM = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krM.fit(KMAT_M, LocTrainLabels)
                            y_krM = krM.predict(KPred_M)
                            TrialMAEM  = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)
                            TrialPosM.append(LocPosM)
                            TrialPerf.append(TrialMAEM)

                    # Exiting the Trial-Generator-Loop now.
                    # Grab the Trial that had the best performance.
                    CurBestID   = np.argmin(TrialPerf)
                    CurBestPerf = min(TrialPerf)
                    CurBestPos  = dc(TrialPosM[CurBestID])

                    # Check convergence condition
                    if CurBestPerf < UpdatingPerfM:                                                              # Is it better, than before?
#                        print("Is better than before! {} < {}".format(CurBestPerf, UpdatingPerfM))
                        if  (abs(CurBestPerf - UpdatingPerfM) < convM) and (ignoreConv == False):                # Yes - and fulfils convergence criterion.
                            # Accept the model and break out of the while loop.
                            FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                            FinLamPos            = dc(CurBestPos[-1])
                            FinSigs              = np.zeros((len(FinSigPos)))
                            for jj in range(len(FinSigPos)):
                                FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                            self.MCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                            self.MCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                            MConverged = True
                            print("## Steepest Descent CONVERGED!")
                            print("## Final MAE Performance : {}!".format(CurBestPerf))

                        else:                                                                                    # Yes - but not yet converged!
                            # Update the Reference Positions with the optimized ones and go for the next loop.
                            UpdatingPosM  = dc(CurBestPos)
                            UpdatingPerfM = dc(CurBestPerf)
                            CurLoopCnt   += 1
                            if CurLoopCnt > MaxIter:                                                              # if more than acceptable number of steps, signal convergence.
                                FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                                FinLamPos            = dc(CurBestPos[-1])
                                FinSigs              = np.zeros((len(FinSigPos)))
                                for jj in range(len(FinSigPos)):
                                    FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                                self.MCurSigmas[-1]  = dc(FinSigs)                                               # Grab all values up to length of Sigma_Mat inside the "self" object.
                                self.MCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                                MConverged = True
                                print("## Steepest Descent RAN OUT OF STEPS!")
                                print("## Final MAE Performance : {}!".format(CurBestPerf))

                    else:                                                                                        # No - none of the current trials were better than before.
                        # Signal convergence and accept current Updating Positions.
                        FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                        FinLamPos            = dc(CurBestPos[-1])
                        FinSigs              = np.zeros((len(FinSigPos)))
                        for jj in range(len(FinSigPos)):
                            FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                        self.MCurSigmas[-1]  = dc(FinSigs)                                                       # Grab all values up to length of Sigma_Mat inside the "self" object.
                        self.MCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                        MConverged = True
                        print("## Steepest Descent CONVERGED!")
                        print("## Final MAE Performance : {}!".format(CurBestPerf))

                print("")
                print("#############################################################")
                print("                FINISHED STEEPEST DESCENT                    ")
                print("#############################################################")

            if useGUI == False:
                SSS=[]
                SSS.append("Full Grid done after {} seconds.\n".format(AllTs))
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#                sns.heatmap(np.log(PreR2SK), cmap='jet', ax=axes[0])
                sns.heatmap(PreR2SK, cmap='jet', ax=axes[0])
                axes[0].set_xlabel('LAMBDA')
                axes[0].set_ylabel('SIGMA')
                axes[0].set_title("R2 values")
                sns.heatmap(np.log(PreMAEs), cmap='jet', ax=axes[1])
                axes[1].set_xlabel('LAMBDA')
                axes[1].set_ylabel('SIGMA')
                axes[1].set_title("log(MAE) values")
                rescoord = np.where(PreR2SK == np.nanmax(PreR2SK))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                SigMaxPosR = dc(SIG_MAX)
                LamMaxPosR = dc(LAM_MAX)
                self.RCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.RCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                SSS.append("Best R2 was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})\n".format(PreR2SK[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                SSS.append("MAE at this point is {}.\n\n".format(PreMAEs[SIG_MAX, LAM_MAX]))
                oldR2 = dc(PreR2SK[SIG_MAX, LAM_MAX])
                rescoord = np.where(PreMAEs == np.nanmin(PreMAEs))
                SIG_MAX = rescoord[0][0]
                LAM_MAX = rescoord[1][0]
                SigMaxPosM = dc(SIG_MAX)
                LamMaxPosM = dc(LAM_MAX)
                self.MCurSigmas.append(DataTensObj.SIGMA_MAT[:, SIG_MAX])
                self.MCurLambdas.append(DataTensObj.LAMBDA_GRID[LAM_MAX])
                SSS.append("Best MAE was {: 2.8f} @ LAMBDA-Point({}) / SIGMA-Point({})\n".format(PreMAEs[SIG_MAX, LAM_MAX], LAM_MAX, SIG_MAX))
                SSS.append("R2 at this point is {}.\n\n".format(PreR2SK[SIG_MAX, LAM_MAX]))
                oldMAE = dc(PreMAEs[SIG_MAX, LAM_MAX])

                SSS.append("#############################################################\n")
                SSS.append(" Initial guess done - run iterative 1D post-optimizations... \n")
                SSS.append("#############################################################\n\n")

                # Grab the "static" sigma-values and lambda as the values from pre-optimization. Pack both sigma and lambda inside the same scan.

                # R-MODEL
                # R-MODEL
                # R-MODEL
                UpdatingPosR  = [SigMaxPosR]*len(DataTensObj.SIGMA_MAT) + [LamMaxPosR]   # place the initial positions of all dimensions on the grid (R2).
                UpdatingPerfR = dc(oldR2)                                                # place the latest peformance of the model here.
                RConverged = False
                CurLoopCnt = 0
                # WHILE STARTS HERE
                SSS.append("# Entering steepest-ascent optimization for the R2-Model...\n")
                while RConverged == False:
                    SSS.append("## Current Loop Number {}. Best R2-Model's R2 so far: {}\n".format(CurLoopCnt, UpdatingPerfR))
                    # Flush the current TrialPosVector list
                    TrialPosR = []
                    TrialPerf = []
                    # Individual directions start here.
                    for ii in range(len(UpdatingPosR)):
                        # Perform the "decrease" operation, if possible.
                        if UpdatingPosR[ii] > 0:
                            # Copy most recent accepted Vector
                            LocPosR = dc(UpdatingPosR)
                            # Decrease the trial position of the ii-th dimension
                            LocPosR[ii] -= 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosR[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosR[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_R           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_R          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krR = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krR.fit(KMAT_R, LocTrainLabels)
                            y_krR = krR.predict(KPred_R)
                            TrialR2R  = self.correval(LocValiLabels, y_krR)
                            TrialPosR.append(LocPosR)
                            TrialPerf.append(TrialR2R)

                        # Perform the "increase" operation if possible. (Possible, if currently smaller than GridPts-2, as Pts can only be accessed until Lambda[GridPts-1])
                        if UpdatingPosR[ii] < (DataTensObj.CVGridPts-2):
                            # Copy most recent accepted Vector
                            LocPosR = dc(UpdatingPosR)
                            # Increase the trial position of the ii-th dimension
                            LocPosR[ii] += 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosR[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosR[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_R           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_R          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krR = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krR.fit(KMAT_R, LocTrainLabels)
                            y_krR = krR.predict(KPred_R)
                            TrialR2R  = self.correval(LocValiLabels, y_krR)
                            TrialPosR.append(LocPosR)
                            TrialPerf.append(TrialR2R)

                    # Exiting the Trial-Generator-Loop now.
                    # Grab the Trial that had the best performance.
                    CurBestID   = np.argmax(TrialPerf)
                    CurBestPerf = max(TrialPerf)
                    CurBestPos  = dc(TrialPosR[CurBestID])

                    # Check convergence condition
                    if CurBestPerf > UpdatingPerfR:                                                              # Is it better, than before?
#                        print("Is better than before! {} > {}".format(CurBestPerf, UpdatingPerfR))
                        if  ((CurBestPerf - UpdatingPerfR) < convR) and (ignoreConv == False):                                               # Yes - and fulfils convergence criterion.
                            # Accept the model and break out of the while loop.
                            FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                            FinLamPos            = dc(CurBestPos[-1])
                            FinSigs              = np.zeros((len(FinSigPos)))
                            for jj in range(len(FinSigPos)):
                                FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                            self.RCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                            self.RCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                            RConverged = True
                            SSS.append("## Steepest Descent CONVERGED!\n")
                            SSS.append("## Final R2 Performance : {}!\n\n".format(CurBestPerf))

                        else:                                                                                    # Yes - but not yet converged!
                            # Update the Reference Positions with the optimized ones and go for the next loop.
                            UpdatingPosR  = dc(CurBestPos)
                            UpdatingPerfR = dc(CurBestPerf)
                            CurLoopCnt += 1
                            if CurLoopCnt > MaxIter:                                                              # if more than acceptable number of steps, signal convergence.
                                FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                                FinLamPos            = dc(CurBestPos[-1])
                                FinSigs              = np.zeros((len(FinSigPos)))
                                for jj in range(len(FinSigPos)):
                                    FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                                self.RCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                                self.RCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                                RConverged = True
                                SSS.append("## Steepest Descent RAN OUT OF STEPS!\n")
                                SSS.append("## Final R2 Performance : {}!\n\n".format(CurBestPerf))

                    else:                                                                                        # No - none of the current trials were better than before.
                        # Signal convergence and accept current Updating Positions.
                        FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                        FinLamPos            = dc(CurBestPos[-1])
                        FinSigs              = np.zeros((len(FinSigPos)))
                        for jj in range(len(FinSigPos)):
                            FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                        self.RCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                        self.RCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                        RConverged = True
                        SSS.append("## Steepest Descent CONVERGED!\n")
                        SSS.append("## Final R2 Performance : {}!\n\n".format(CurBestPerf))



                # M-MODEL
                # M-MODEL
                # M-MODEL
                UpdatingPosM  = [SigMaxPosM]*len(DataTensObj.SIGMA_MAT) + [LamMaxPosM]   # place the initial positions of all dimensions on the grid (MAE).
                UpdatingPerfM = dc(oldMAE)                                              # place the latest perfomance of the model here.
                MConverged = False
                CurLoopCnt = 0
                # WHILE STARTS HERE
                SSS.append("# Entering steepest-descent optimization for the MAE-Model...\n")
                while MConverged == False:
                    SSS.append("## Current Loop Number {}. Best MAE-Model's MAE so far: {}\n".format(CurLoopCnt, UpdatingPerfM))
                    # Flush the current TrialPosVector list
                    TrialPosM = []
                    TrialPerf = []
                    # Individual directions start here.
                    for ii in range(len(UpdatingPosM)):
                        # Perform the "decrease" operation, if possible.
                        if UpdatingPosM[ii] > 0:
                            # Copy most recent accepted Vector
                            LocPosM = dc(UpdatingPosM)
                            # Decrease the trial position of the ii-th dimension
                            LocPosM[ii] -= 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosM[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosM[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_M           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_M          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krM = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krM.fit(KMAT_M, LocTrainLabels)
                            y_krM = krM.predict(KPred_M)
                            TrialMAEM  = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)
                            TrialPosM.append(LocPosM)
                            TrialPerf.append(TrialMAEM)

                        # Perform the "increase" operation if possible. (Possible, if currently smaller than GridPts-2, as Pts can only be accessed until Lambda[GridPts-1])
                        if UpdatingPosM[ii] < (DataTensObj.CVGridPts-2):
                            # Copy most recent accepted Vector
                            LocPosM = dc(UpdatingPosM)
                            # Increase the trial position of the ii-th dimension
                            LocPosM[ii] += 1
                            # Generate the resulting Trial Sigma vetor / Trial Lambda from these trial Positions
                            TrialSigma  = np.zeros((len(DataTensObj.SIGMA_MAT)))
                            for jj in range(len(DataTensObj.SIGMA_MAT)):
                                TrialSigma[jj] = dc(DataTensObj.SIGMA_MAT[jj, LocPosM[jj]])                        # From the original grid, grab the jj'ths descriptor at its current trial position.
                            TrialLambda = DataTensObj.LAMBDA_GRID[LocPosM[-1]]
                            # Store Performance of the ensuing model alongside the full N(Descriptors)+1(Lambda) TrialPos Vector.
                            KMAT_M           = kernel_gen(TrialSigma, DataTensObj.TrainTens)
                            KPred_M          = kernel_gen(TrialSigma, DataTensObj.ValiTens)
                            krM = KernelRidge(kernel="precomputed", alpha=TrialLambda)
                            krM.fit(KMAT_M, LocTrainLabels)
                            y_krM = krM.predict(KPred_M)
                            TrialMAEM  = sum(abs(y_krM - LocValiLabels))/len(LocValiLabels)
                            TrialPosM.append(LocPosM)
                            TrialPerf.append(TrialMAEM)

                    # Exiting the Trial-Generator-Loop now.
                    # Grab the Trial that had the best performance.
                    CurBestID   = np.argmin(TrialPerf)
                    CurBestPerf = min(TrialPerf)
                    CurBestPos  = dc(TrialPosM[CurBestID])

                    # Check convergence condition
                    if CurBestPerf < UpdatingPerfM:                                                              # Is it better, than before?
#                        print("Is better than before! {} < {}".format(CurBestPerf, UpdatingPerfM))
                        if  (abs(CurBestPerf - UpdatingPerfM) < convM) and (ignoreConv == False):                # Yes - and fulfils convergence criterion.
                            # Accept the model and break out of the while loop.
                            FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                            FinLamPos            = dc(CurBestPos[-1])
                            FinSigs              = np.zeros((len(FinSigPos)))
                            for jj in range(len(FinSigPos)):
                                FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                            self.MCurSigmas[-1]  = dc(FinSigs)                                                   # Grab all values up to length of Sigma_Mat inside the "self" object.
                            self.MCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                            MConverged = True
                            SSS.append("## Steepest Descent CONVERGED!\n")
                            SSS.append("## Final MAE Performance : {}!\n\n".format(CurBestPerf))

                        else:                                                                                    # Yes - but not yet converged!
                            # Update the Reference Positions with the optimized ones and go for the next loop.
                            UpdatingPosM  = dc(CurBestPos)
                            UpdatingPerfM = dc(CurBestPerf)
                            CurLoopCnt   += 1
                            if CurLoopCnt > MaxIter:                                                              # if more than acceptable number of steps, signal convergence.
                                FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                                FinLamPos            = dc(CurBestPos[-1])
                                FinSigs              = np.zeros((len(FinSigPos)))
                                for jj in range(len(FinSigPos)):
                                    FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                                self.MCurSigmas[-1]  = dc(FinSigs)                                               # Grab all values up to length of Sigma_Mat inside the "self" object.
                                self.MCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                                MConverged = True
                                SSS.append("## Steepest Descent RAN OUT OF STEPS!\n")
                                SSS.append("## Final MAE Performance : {}!\n\n".format(CurBestPerf))

                    else:                                                                                        # No - none of the current trials were better than before.
                        # Signal convergence and accept current Updating Positions.
                        FinSigPos            = dc(CurBestPos[:len(DataTensObj.SIGMA_MAT)])
                        FinLamPos            = dc(CurBestPos[-1])
                        FinSigs              = np.zeros((len(FinSigPos)))
                        for jj in range(len(FinSigPos)):
                            FinSigs[jj] = dc(DataTensObj.SIGMA_MAT[jj, FinSigPos[jj]])
                        self.MCurSigmas[-1]  = dc(FinSigs)                                                       # Grab all values up to length of Sigma_Mat inside the "self" object.
                        self.MCurLambdas[-1] = dc(DataTensObj.LAMBDA_GRID[FinLamPos])
                        MConverged = True
                        SSS.append("## Steepest Descent CONVERGED!\n")
                        SSS.append("## Final MAE Performance : {}!\n\n".format(CurBestPerf))

                SSS.append("#############################################################\n")
                SSS.append("                FINISHED STEEPEST DESCENT                    \n")
                SSS.append("#############################################################\n")

                # Write everything to the files
                OutFNam = "CVStats_{}.out".format(len(self.RCurLambdas))
                OutPNam = "CVStats_{}.svg".format(len(self.RCurLambdas))
                OID = open(OutFNam, "w")
                for ii in range(len(SSS)):
                    OID.write(SSS[ii])
                OID.close()
                fig.savefig(OutPNam, format="svg", dpi=300)
                plt.close(fig)

        return None

    # After all K folds of CV have been performed, use the R/M CurSigmas and Lambdas to have a final training of the two different training styles with the
    # averaged CV'ed hyperparameters. Choose the better performing model.
    def final_cv(self, DataTensObj, MLObjList=[], useGUI=True):
        """
        A method to finalize the hyperparameter optimization and compare the test set's "true" values versus "predicted" values.
        A learning curve may be constructed, when additionally controlling the amount of active training data, as explained
        in the Data Tensor class. Will store the finalized model to the instance itself.
        (returns None)

        --- Parameters ---
        ------------------
        DataTensObj : data tensor object instance
            Contains the training / prediction data in the required format - including the information which parts are
            (active / inactive) training data for the cross-validation and which parts are testing data.

        MLObjList : list of MLEntry object instances
            A list of the MLEntries.  Providing it is required for medium and low-memory training options, as only then,
            the on-the-fly methods can be applied.
        """
        # Put together the current mean values of the cross-validations
        RCV_Sigmas = np.zeros((len(self.RCurSigmas[0])))
        MCV_Sigmas = np.zeros((len(self.MCurSigmas[0])))
        for ii in range(len(self.RCurSigmas)):
            for jj in range(len(self.RCurSigmas[0])):
                RCV_Sigmas[jj] += self.RCurSigmas[ii][jj]/float(len(self.RCurSigmas))
                MCV_Sigmas[jj] += self.MCurSigmas[ii][jj]/float(len(self.MCurSigmas))
        RCV_Lambda = np.mean(self.RCurLambdas)
        MCV_Lambda = np.mean(self.MCurLambdas)
        self.FinSigmas_M = dc(MCV_Sigmas)
        self.FinLambda_M = dc(MCV_Lambda)
        self.FinSigmas_R = dc(RCV_Sigmas)
        self.FinLambda_R = dc(RCV_Lambda)
        # Get the current Active Training IDs and TestIDs, respectively.
        LocTRIDs, LocTEIDs, Aux1, Aux2 = [], [], [], []
        for ii in range(DataTensObj.NEntries):
            LocStat = DataTensObj.Status[ii]
            if LocStat == "TrainAct":
                LocTRIDs.append(ii)
                Aux1.append(DataTensObj.ID[ii])
                Aux2.append(DataTensObj.FingerP[ii])
            if LocStat == "Test":
                LocTEIDs.append(ii)
        self.FinTrainNames   = dc(Aux1)
        self.FinTrainFPrints = dc(Aux2)
        self.FinDBSpecs      = dc(DataTensObj.DBSpecs)

        # Differentiate between memory variants.
        if DataTensObj.memmode == "high":
            # Use the FinalTens routine to generate the Training and Testing Tensors
            DataTensObj.final_tens(LocTRIDs, LocTEIDs)
            # Generate the Training / Testing KMat's with the respective Sigmas.
            KMAT_R         = kernel_gen(RCV_Sigmas, DataTensObj.FinTrainTens)
            KMAT_M         = kernel_gen(MCV_Sigmas, DataTensObj.FinTrainTens)
            DataTensObj.FinTrainTens = []
            KPred_R        = kernel_gen(RCV_Sigmas, DataTensObj.FinTestTens)
            KPred_M        = kernel_gen(MCV_Sigmas, DataTensObj.FinTestTens)
            DataTensObj.FinTestTens = []
            LocFinTrainLabels = DataTensObj.FinTrainLabels[self.Target]
            LocFinTestLabels  = DataTensObj.FinTestLabels[self.Target]

        if DataTensObj.memmode == "medium":
            if MLObjList==[]:
                print("In medium-memory mode, it is necessary to add the MLObject List to final_tens!")
            # Use the FinalTens routine to generate the Training and Testing Tensors
            DataTensObj.final_tens(LocTRIDs, LocTEIDs, MLObjList)
            # Generate the Training / Testing KMat's with the respective Sigmas.
            KMAT_R         = kernel_gen(RCV_Sigmas, DataTensObj.FinTrainTens)
            KMAT_M         = kernel_gen(MCV_Sigmas, DataTensObj.FinTrainTens)
            DataTensObj.FinTrainTens = []
            KPred_R        = kernel_gen(RCV_Sigmas, DataTensObj.FinTestTens)
            KPred_M        = kernel_gen(MCV_Sigmas, DataTensObj.FinTestTens)
            DataTensObj.FinTestTens = []
            LocFinTrainLabels = DataTensObj.FinTrainLabels[self.Target]
            LocFinTestLabels  = DataTensObj.FinTestLabels[self.Target]

        if DataTensObj.memmode == "low":
            if MLObjList==[]:
                print("In low-memory mode, it is necessary to add the MLObject List to final_tens!")
            print("Low-memory on-the-fly mode was selected.")
            # Pre-allocate KMatrices for training-only and (Vali x Train) sets.
            AuxKM = np.zeros((len(LocTRIDs), len(LocTRIDs)))
            AuxPM = np.zeros((len(LocTEIDs), len(LocTRIDs)))
            AuxKR = np.zeros((len(LocTRIDs), len(LocTRIDs)))
            AuxPR = np.zeros((len(LocTEIDs), len(LocTRIDs)))
            # Go through the descriptors
            curLayer      = 0
            for ii in DataTensObj.DescL:
                if DataTensObj.DescType[ii] != "unused":
                    LocDescL = ii
                    LocDataF = DataTensObj.DescType[ii][0]                                         # Get the Data Format
                    LocDataO = DataTensObj.DescType[ii][1]                                         # Get the Distance operator

                    # Scalar - DistMat
                    if LocDataF == "Scalar":
                        if LocDataO == "DistMat":
                            print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                            MSig = self.FinSigmas_M[curLayer]
                            RSig = self.FinSigmas_R[curLayer]
                            AuxKM += calc_pred_distmat(MLObjList, LocDescL, LocTRIDs, LocTRIDs, MSig)
                            AuxPM += calc_pred_distmat(MLObjList, LocDescL, LocTRIDs, LocTEIDs, MSig)
                            AuxKR += calc_pred_distmat(MLObjList, LocDescL, LocTRIDs, LocTRIDs, RSig)
                            AuxPR += calc_pred_distmat(MLObjList, LocDescL, LocTRIDs, LocTEIDs, RSig)
                            curLayer += 1

                    # EigCoulVec - EucDistMat
                    if LocDataF == "EigCoulVec":
                        if LocDataO == "EucDistMat":
                            print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                            MSig = self.FinSigmas_M[curLayer]
                            RSig = self.FinSigmas_R[curLayer]
                            AuxKM += calc_pred_eucdistmat(MLObjList, LocDescL, LocTRIDs, LocTRIDs, MSig)
                            AuxPM += calc_pred_eucdistmat(MLObjList, LocDescL, LocTRIDs, LocTEIDs, MSig)
                            AuxKR += calc_pred_eucdistmat(MLObjList, LocDescL, LocTRIDs, LocTRIDs, RSig)
                            AuxPR += calc_pred_eucdistmat(MLObjList, LocDescL, LocTRIDs, LocTEIDs, RSig)
                            curLayer += 1

                    # ArrOfScal - DistMat
                    if LocDataF == "ArrOfScal":
                        if LocDataO == "DistMat":
                            print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            for jj in range(LocArrLen):
                                MSig = self.FinSigmas_M[curLayer]
                                RSig = self.FinSigmas_R[curLayer]
                                AuxKM += calc_pred_distmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTRIDs, MSig)
                                AuxPM += calc_pred_distmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTEIDs, MSig)
                                AuxKR += calc_pred_distmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTRIDs, RSig)
                                AuxPR += calc_pred_distmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTEIDs, RSig)
                                curLayer += 1

                    # ArrOfArr - EucDistMat
                    if LocDataF == "ArrOfArr":
                        if LocDataO == "EucDistMat":
                            print("On-the-Fly embedding of descriptor {}...".format(LocDescL))
                            LocArrLen = len(MLObjList[0].Desc[LocDescL])
                            for jj in range(LocArrLen):
                                MSig = self.FinSigmas_M[curLayer]
                                RSig = self.FinSigmas_R[curLayer]
                                AuxKM += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTRIDs, MSig)
                                AuxPM += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTEIDs, MSig)
                                AuxKR += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTRIDs, RSig)
                                AuxPR += calc_pred_eucdistmat_layer(MLObjList, LocDescL, jj, LocTRIDs, LocTEIDs, RSig)
                                curLayer += 1

            KMAT_R           = np.exp(-AuxKR)
            KMAT_M           = np.exp(-AuxKM)
            KPred_R          = np.exp(-AuxPR)
            KPred_M          = np.exp(-AuxPM)
            AuxKR, AuxPR     = [], []
            AuxKM, AuxPM     = [], []
            All_Train_L = {}
            All_Test_L  = {}
            for ii in range(len(DataTensObj.LabelL)):
                LocLabelL = DataTensObj.LabelL[ii]
                LocList   = DataTensObj.Labels[LocLabelL]
                LocAux    = []
                for jj in LocTRIDs:
                    LocAux.append(LocList[jj])
                All_Train_L[LocLabelL] = dc(LocAux)
                LocAux    = []
                for jj in LocTEIDs:
                    LocAux.append(LocList[jj])
                All_Test_L[LocLabelL]  = dc(LocAux)
            LocFinTrainLabels = All_Train_L[self.Target]
            LocFinTestLabels  = All_Test_L[self.Target]

        # Store the Labels used in final testing.
        self.FinTestLabelVals = dc(LocFinTestLabels)

        # Train the models.
        R_Model = KernelRidge(kernel="precomputed", alpha=RCV_Lambda)
        M_Model = KernelRidge(kernel="precomputed", alpha=MCV_Lambda)
        R_Model.fit(KMAT_R, LocFinTrainLabels)
        M_Model.fit(KMAT_M, LocFinTrainLabels)
        # Evaluate their accuracies
        TestPreds_M = M_Model.predict(KPred_M)
        self.FinPredMVals = dc(TestPreds_M)
        R2_M = self.correval(LocFinTestLabels, TestPreds_M)
        MAE_M = sum(abs(TestPreds_M.flatten() - LocFinTestLabels))/len(LocFinTestLabels)
        Errors_M = np.zeros((len(TestPreds_M)))
        for ii in range(len(TestPreds_M)):
            Errors_M[ii] = TestPreds_M[ii] - LocFinTestLabels[ii]
        STD_M = np.std(Errors_M)
        TrainPreds_M = M_Model.predict(KMAT_M)
        R2_MT = self.correval(LocFinTrainLabels, TrainPreds_M)
        MAE_MT = sum(abs(TrainPreds_M.flatten() - LocFinTrainLabels))/len(LocFinTrainLabels)


        # Storing the final performance of the MAE trained model.
        self.FinR2_M     = r2_score(LocFinTestLabels, TestPreds_M)   # Stores the R2 score of the testing with the MAE trained model.
        Aux  = np.asarray(LocFinTestLabels).reshape(-1, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            AuxVal = pearsonr(Aux, TestPreds_M)
            self.FinpR_M = AuxVal[0]                                 # Stores the pearson R score of the testing with the MAE model.
        self.FinMAE_M    = MAE_M                                     # Stores the MAE score of the testing with the MAE trained model.

        self.FinR2_MT    = r2_score(LocFinTrainLabels, TrainPreds_M) # Stores the R2 score of the Training-Data-Prediction with the MAE model.
        Aux  = np.asarray(LocFinTrainLabels).reshape(-1, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            AuxVal = pearsonr(Aux, TrainPreds_M)
            self.FinpR_MT = AuxVal[0]                                # Stores the pearson R score of the testing with the MAE model.
        self.FinMAE_MT   = MAE_MT                                    # Stores the MAE score of the Training-Data-Prediction with the MAE model.

        TestPreds_R = R_Model.predict(KPred_R)
        self.FinPredRVals = dc(TestPreds_R)
        R2_R = self.correval(LocFinTestLabels, TestPreds_R)
        MAE_R = sum(abs(TestPreds_R.flatten() - LocFinTestLabels))/len(LocFinTestLabels)
        Errors_R = np.zeros((len(TestPreds_R)))
        for ii in range(len(TestPreds_R)):
            Errors_R[ii] = TestPreds_R[ii] - LocFinTestLabels[ii]
        STD_R = np.std(Errors_R)
        TrainPreds_R = R_Model.predict(KMAT_R)
        R2_RT = self.correval(LocFinTrainLabels, TrainPreds_R)
        MAE_RT = sum(abs(TrainPreds_R.flatten() - LocFinTrainLabels))/len(LocFinTrainLabels)

        # Storing the final performance of the R trained model.
        self.FinR2_R     = r2_score(LocFinTestLabels, TestPreds_R)   # Stores the R2 score of the testing with the R trained model.
        Aux  = np.asarray(LocFinTestLabels,).reshape(-1, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            AuxVal = pearsonr(Aux, TestPreds_R)
            self.FinpR_R = AuxVal[0]                                 # Stores the pearson R score of the testing with the R model.
        self.FinMAE_R    = MAE_R                                     # Stores the MAE score of the testing with the R trained model.

        self.FinR2_RT    = r2_score(LocFinTrainLabels, TrainPreds_R) # Stores the R2 score of the testing with the R trained model.
        Aux  = np.asarray(LocFinTrainLabels,).reshape(-1, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            AuxVal = pearsonr(Aux, TrainPreds_R)
            self.FinpR_RT = AuxVal[0]                                # Stores the pearson R score of the testing with the R model.
        self.FinMAE_RT   = MAE_RT                                    # Stores the MAE score of the testing with the R trained model.

        if useGUI == True:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].set_title("Trained against R as Loss")
            axes[0].set_aspect(aspect="equal")
            axes[0].plot(LocFinTestLabels, TestPreds_R, "o", mec='#ffffff', mew=0.25, ls="None", ms=3, label=r"R = {:2.3f}".format(R2_R))
            axes[0].set_xlabel('True {} values'.format(self.Target))
            axes[0].set_ylabel('Predicted {} values'.format(self.Target))
            edgeMin = min([min(LocFinTestLabels), min(TestPreds_R)])
            edgeMax = max([max(LocFinTestLabels), max(TestPreds_R)])
            LocDiff = (edgeMax - edgeMin)*0.01
            bounds =[edgeMin-LocDiff, edgeMax+LocDiff]
            lims = [bounds[0]-1, bounds[1]+1]
            axes[0].set_xlim(bounds)
            axes[0].set_ylim(bounds)
            winm = [lims[0]-MAE_R, lims[1]-MAE_R]
            winu = [lims[0]+MAE_R, lims[1]+MAE_R]
            axes[0].plot(lims, lims, 'b-', lw=1.0)
            axes[0].plot(lims, winm, 'r--', lw=0.75)
            axes[0].plot(lims, winu, 'r--', label="MAE window ({:2.3f})".format(MAE_R), lw=0.75)
            Winm = [lims[0]-2*STD_R, lims[1]-2*STD_R]
            Winu = [lims[0]+2*STD_R, lims[1]+2*STD_R]
            axes[0].plot(lims, Winm, 'g:', lw=0.75)
            axes[0].plot(lims, Winu, 'g:', label=r"2$\sigma$ Window ({:2.3f})".format(2*STD_R), lw=0.75)
            leg = axes[0].legend(handlelength=0, handletextpad=0, fancybox=True)
            leg.legendHandles[1].set_visible(False)
            axes[0].legend(loc=2)
            axes[0].grid(True)

            axes[1].set_title("Trained against MAE as Loss")
            axes[1].set_aspect(aspect="equal")
            axes[1].plot(LocFinTestLabels, TestPreds_M, "o", mec='#ffffff', mew=0.25, ls="None", ms=3, label=r"R = {:2.3f}".format(R2_M))
            axes[1].set_xlabel('True {} values'.format(self.Target))
            axes[1].set_ylabel('Predicted {} values'.format(self.Target))
            edgeMin = min([min(LocFinTestLabels), min(TestPreds_R)])
            edgeMax = max([max(LocFinTestLabels), max(TestPreds_R)])
            LocDiff = (edgeMax - edgeMin)*0.01
            bounds =[edgeMin-LocDiff, edgeMax+LocDiff]
            lims = [bounds[0]-1, bounds[1]+1]
            axes[1].set_xlim(bounds)
            axes[1].set_ylim(bounds)
            winm = [lims[0]-MAE_M, lims[1]-MAE_M]
            winu = [lims[0]+MAE_M, lims[1]+MAE_M]
            axes[1].plot(lims, lims, 'b-', lw=1.0)
            axes[1].plot(lims, winm, 'r--', lw=0.75)
            axes[1].plot(lims, winu, 'r--', label="MAE window ({:2.3f})".format(MAE_M), lw=0.75)
            Winm = [lims[0]-2*STD_M, lims[1]-2*STD_M]
            Winu = [lims[0]+2*STD_M, lims[1]+2*STD_M]
            axes[1].plot(lims, Winm, 'g:', lw=0.75)
            axes[1].plot(lims, Winu, 'g:', label=r"2$\sigma$ Window ({:2.3f})".format(2*STD_M), lw=0.75)
            leg = axes[1].legend(handlelength=0, handletextpad=0, fancybox=True)
            leg.legendHandles[1].set_visible(False)
            axes[1].legend(loc=2)
            axes[1].grid(True)
            plt.show()

            self.FinModel_M  = dc(M_Model)
            self.FinModel_R  = dc(R_Model)
            self.FinDescL = dc(DataTensObj.DescL)

            print("R Trained Model Statistics   :  Train-R2    Train-pR    Train-MAE    Test-R2    Test-pR    Test-MAE")
            print("                                {}          {}          {}           {}         {}         {}".format(self.FinR2_RT, self.FinpR_RT, self.FinMAE_RT, self.FinR2_R, self.FinpR_R, self.FinMAE_R))
            print("")                                                                                         
            print("MAE Trained Model Statistics :  Train-R2    Train-pR    Train-MAE    Test-R2    Test-pR    Test-MAE")
            print("                                {}          {}          {}           {}         {}         {}".format(self.FinR2_MT, self.FinpR_MT, self.FinMAE_MT, self.FinR2_M, self.FinpR_M, self.FinMAE_M))

        if useGUI == False:
            SSS=[]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].set_title("Trained against R as Loss")
            axes[0].set_aspect(aspect="equal")
            axes[0].plot(LocFinTestLabels, TestPreds_R, "o", mec='#ffffff', mew=0.25, ls="None", ms=3, label=r"R = {:2.3f}".format(R2_R))
            axes[0].set_xlabel('True {} values'.format(self.Target))
            axes[0].set_ylabel('Predicted {} values'.format(self.Target))
            edgeMin = min([min(LocFinTestLabels), min(TestPreds_R)])
            edgeMax = max([max(LocFinTestLabels), max(TestPreds_R)])
            LocDiff = (edgeMax - edgeMin)*0.01
            bounds =[edgeMin-LocDiff, edgeMax+LocDiff]
            lims = [bounds[0]-1, bounds[1]+1]
            axes[0].set_xlim(bounds)
            axes[0].set_ylim(bounds)
            winm = [lims[0]-MAE_R, lims[1]-MAE_R]
            winu = [lims[0]+MAE_R, lims[1]+MAE_R]
            axes[0].plot(lims, lims, 'b-', lw=1.0)
            axes[0].plot(lims, winm, 'r--', lw=0.75)
            axes[0].plot(lims, winu, 'r--', label="MAE window ({:2.3f})".format(MAE_R), lw=0.75)
            Winm = [lims[0]-2*STD_R, lims[1]-2*STD_R]
            Winu = [lims[0]+2*STD_R, lims[1]+2*STD_R]
            axes[0].plot(lims, Winm, 'g:', lw=0.75)
            axes[0].plot(lims, Winu, 'g:', label=r"2$\sigma$ Window ({:2.3f})".format(2*STD_R), lw=0.75)
            leg = axes[0].legend(handlelength=0, handletextpad=0, fancybox=True)
            leg.legendHandles[1].set_visible(False)
            axes[0].legend(loc=2)
            axes[0].grid(True)

            axes[1].set_title("Trained against MAE as Loss")
            axes[1].set_aspect(aspect="equal")
            axes[1].plot(LocFinTestLabels, TestPreds_M, "o", mec='#ffffff', mew=0.25, ls="None", ms=3, label=r"R = {:2.3f}".format(R2_M))
            axes[1].set_xlabel('True {} values'.format(self.Target))
            axes[1].set_ylabel('Predicted {} values'.format(self.Target))
            edgeMin = min([min(LocFinTestLabels), min(TestPreds_R)])
            edgeMax = max([max(LocFinTestLabels), max(TestPreds_R)])
            LocDiff = (edgeMax - edgeMin)*0.01
            bounds =[edgeMin-LocDiff, edgeMax+LocDiff]
            lims = [bounds[0]-1, bounds[1]+1]
            axes[1].set_xlim(bounds)
            axes[1].set_ylim(bounds)
            winm = [lims[0]-MAE_M, lims[1]-MAE_M]
            winu = [lims[0]+MAE_M, lims[1]+MAE_M]
            axes[1].plot(lims, lims, 'b-', lw=1.0)
            axes[1].plot(lims, winm, 'r--', lw=0.75)
            axes[1].plot(lims, winu, 'r--', label="MAE window ({:2.3f})".format(MAE_M), lw=0.75)
            Winm = [lims[0]-2*STD_M, lims[1]-2*STD_M]
            Winu = [lims[0]+2*STD_M, lims[1]+2*STD_M]
            axes[1].plot(lims, Winm, 'g:', lw=0.75)
            axes[1].plot(lims, Winu, 'g:', label=r"2$\sigma$ Window ({:2.3f})".format(2*STD_M), lw=0.75)
            leg = axes[1].legend(handlelength=0, handletextpad=0, fancybox=True)
            leg.legendHandles[1].set_visible(False)
            axes[1].legend(loc=2)
            axes[1].grid(True)

            self.FinModel_M  = dc(M_Model)
            self.FinModel_R  = dc(R_Model)
            self.FinDescL = dc(DataTensObj.DescL)

            SSS.append("R Trained Model Statistics   :  Train-R2    Train-pR    Train-MAE    Test-R2    Test-pR    Test-MAE\n")
            SSS.append("                                {}          {}          {}           {}         {}         {}\n\n".format(self.FinR2_RT, self.FinpR_RT, self.FinMAE_RT, self.FinR2_R, self.FinpR_R, self.FinMAE_R))
            SSS.append("MAE Trained Model Statistics :  Train-R2    Train-pR    Train-MAE    Test-R2    Test-pR    Test-MAE\n")
            SSS.append("                                {}          {}          {}           {}         {}         {}\n".format(self.FinR2_MT, self.FinpR_MT, self.FinMAE_MT, self.FinR2_M, self.FinpR_M, self.FinMAE_M))

            # Write everything to the files
            OutFNam = "CVFinal.out".format(len(self.RCurLambdas))
            OutPNam = "CVFinal.svg".format(len(self.RCurLambdas))
            OID = open(OutFNam, "w")
            for ii in range(len(SSS)):
                OID.write(SSS[ii])
            OID.close()
            fig.savefig(OutPNam, format="svg", dpi=300)
            plt.close(fig)

        return None

    # Saves the model to the ModelPath after Final Training - Needs the DataTensObject to see which entries were Train-Active.
    # Options are to choose which Loss to compare (has to be either "R2" or "MAE") and/or which Method to use for comparison (which overrides automatically if not "Auto")
    def save_model(self, iName, Loss="R2", Method="auto"):
        """
        A method to save the finalized model, along other vital information about the database, as to run predictions later.
        Will store the model inside the provided ModelPath.
        (returns None)

        --- Parameters ---
        ------------------
        iName : str
            The name of the file to be saved to.

        Loss : str
            The training metric for which to store the better-performing model. If Method is "auto", it will select the better of the two models.

        Method : str
            An override switch to just select one of the metrics directly.

        --- Raises ---
        --------------
        InputError :
            Will raise an InputError in case Loss and Method were both set to auto.

            Will also raise an InputError if final_cv has not been performed yet.
        """
        # Throw Error if both are Auto
        if (Loss == "auto") and (Method == "auto"):
            raise InputError("Loss and Method may not both be set to 'auto'. No comparison possible like this.")
        # Throw Error if no Final Testing was performed yet.
        if self.FinModel_M == []:
            raise InputError("No Final Training performed yet. Finish the Cross-Validation and FinalCV first before trying to save a model.")
        Choice = []
        ModNam = MLModel.ModelPath+"{}.model".format(iName)
        try:
            fid = open(MLModel.ModelPath+"{}.hyperpars".format(iName), "w")
        except FileNotFoundError:
            os.mkdir(MLModel.ModelPath)
            fid = open(MLModel.ModelPath+"{}.hyperpars".format(iName), "w")
        # Override options
        if Method == "MAE":
            Choice = "MAE"
            print("Override: Storing the MAE model.")
        if Method == "R2":
            Choice = "R2"
            print("Override: Storing the R2 model.")
        # Auto selection of training Method depending on the selected Loss
        if (Method == "auto"):
            if (Loss == "R2"):
                if self.FinR2_M > self.FinR2_R:
                    Choice = "MAE"
                    print("Storing the MAE model, since R2 is higher there ({} vs {}).".format(self.FinR2_M, self.FinR2_R))
                else:
                    Choice = "R2"
                    print("Storing the R2 model, since R2 is higher there ({} vs {}).".format(self.FinR2_R, self.FinR2_M))
            if (Loss == "MAE"):
                if self.FinMAE_M > self.FinMAE_R:
                    Choice = "R2"
                    print("Storing the R2 model, since MAE is lower there ({} vs {}).".format(self.FinMAE_M, self.FinMAE_R))
                else:
                    Choice = "MAE"
                    print("Storing the MAE model, since MAE is lower there ({} vs {}).".format(self.FinMAE_R, self.FinMAE_M))
        if (Choice == "MAE"):
            pickle.dump(self.FinModel_M, open(ModNam, "wb"))
            fid.write("{}   {}\n".format(len(self.FinSigmas_M), self.FinLambda_M))
            for ii in range(len(self.FinSigmas_M)):
                fid.write("{}\n".format(self.FinSigmas_M[ii]))
        # Override to save R2 model
        if (Choice == "R2"):
            pickle.dump(self.FinModel_R, open(ModNam, "wb"))
            fid.write("{}   {}\n".format(len(self.FinSigmas_R), self.FinLambda_R))
            for ii in range(len(self.FinSigmas_R)):
                fid.write("{}\n".format(self.FinSigmas_R[ii]))
        fid.close()
        # Store the training IDs in their exact order of the KMatrix as well, since they are necessary for consistent predictions.
        gid = open(MLModel.ModelPath+"{}.TrainIDs".format(iName), "w")
        for ii in range(len(self.FinTrainNames)):
            Aux = ""
            for jj in range(len(self.FinTrainFPrints[ii])):
                Aux += "{},".format(self.FinTrainFPrints[ii][jj])
            SSS = Aux.rstrip(Aux[-1])
            gid.write("{}   {}\n".format(self.FinTrainNames[ii], SSS))
        gid.close()
        # Store the DescL information in its exact ordering! Otherwise the Sigmas might not match afterwards!
        hid = open(MLModel.ModelPath+"{}.DescL".format(iName), "w")
        for ii in range(len(self.FinDescL)):
            hid.write("{}\n".format(self.FinDescL[ii]))
        hid.close()
        # Store the DBSpecs when initializing the MLEntries
        jid = open(MLModel.ModelPath+"{}.DBSpecs".format(iName), "w")
        jid.write("{}\n".format(self.FinDBSpecs[0][0]))
        jid.write("{}\n".format(self.FinDBSpecs[1][0]))
        jid.write("{}\n".format(self.FinDBSpecs[2][0]))
        jid.write("{}   {}   {}   {}   {}\n".format(MLEntry.Fold_L1, MLEntry.Fold_L2, MLEntry.MOWin, MLEntry.MergePath, MLEntry.DataPath))
        Aux = ["{} ".format(ii) for ii in self.FinDBSpecs[3][0]]
        SSS = ""
        for ii in range(len(Aux)):
            SSS += Aux[ii]
        SSS += "\n"
        jid.write(SSS)
        jid.close()
        return None

    # Get_Model is a "heavy lifting" method that will, by itself, instantiate two other Objects (a List of MLEntry-Objects and an Descriptor Object) and update itself accordingly.
    # Goal is, to just call this once and it will re-instantiate the necessary "training" data, set up the Descriptor.
    # It should NOT instantiate the DataTens object yet, since this should be loaded together with the (later) prediction data.    
    def get_model(self, iName):
        """
        A method that loads a previously saved model, initializes a descriptor instance and collects the reference training data
        required to build the prediction kernel matrix.
        (returns LocObjList, LocDescInst)

        --- Parameters ---
        ------------------
        iName : str
            The name of the model to load. Will assume that the model is found in the ModelPath attribute.

        --- Returns ---
        ---------------
        LocObjList : list of MLEntry object instances
            A list that contains the training entries (in their exact order) as they were used during training the model.

        LocDescInst : descriptor object instance
            A descriptor instance that has been set up in the exact same way as during original training.
        """
        LocObjList  = []
        LocDescInst = []
        # Load the DBSpecs
        fid  = open(MLModel.ModelPath+"{}.DBSpecs".format(iName), "r")
        FLoc = fid.readlines()
        fid.close()
        LPack  = FLoc[0].split()[0]
        Llow   = FLoc[1].split()[0]
        Lhigh  = FLoc[2].split()[0]
        LFold_L1, LFold_L2, LMOWin, LMergePath, LDataPath = int(FLoc[3].split()[0]), int(FLoc[3].split()[1]), int(FLoc[3].split()[2]), FLoc[3].split()[3], FLoc[3].split()[4]
        LDataL = []
        LType  = "Training"
        DataLLen = len(FLoc[4].split())
        for ii in range(DataLLen):
            LDataL.append(FLoc[4].split()[ii])
        FLoc = []
        # Using the DBSpecs, instantiate the Training MLEntries
        # First, set the Global Values of the MLEntry Object.
        MLEntry.Fold_L1   = LFold_L1
        MLEntry.Fold_L2   = LFold_L2
        MLEntry.MOWin     = LMOWin
        MLEntry.MergePath = LMergePath
        MLEntry.DataPath  = LDataPath
        gid  = open(MLModel.ModelPath+"{}.TrainIDs".format(iName), "r")
        GLoc = gid.readlines()
        gid.close()
        # Read the Training IDs and Fingerprints from Training Library. Initialize the MLEntry objects.
        LID      = []
        LFingerP = []
        for line in range(len(GLoc)):
            LID.append(GLoc[line].split()[0])
            Aux     = GLoc[line].split()[1]
            FingLen = len(Aux.split(","))
            LFing = []
            for jj in range(FingLen):
                LFing.append(int(Aux.split(",")[jj]))
            LFingerP.append(LFing)
            LObj = MLEntry(LID[-1], LType, LFingerP[-1], LPack, Llow, Lhigh, LDataL, self.Target)
            LocObjList.append(dc(LObj))        
        # Load the Descriptor List. Does not have any global Variables that need to be carried over.
        LDescL=[]
        hid  = open(MLModel.ModelPath+"{}.DescL".format(iName), "r")
        HLoc = hid.readlines()
        hid.close()
        for ii in range(len(HLoc)):
            LDescL.append(HLoc[ii].split()[0])
        LocDescInst = Descriptor(LDescL)    
        # Now, finally, load the MLModel and set the Sigmas and Lambda
        # Load the Sigmas and Lambda.
        jid  = open(MLModel.ModelPath+"{}.hyperpars".format(iName), "r")
        JLoc = jid.readlines()
        jid.close()
        NSigs = int(JLoc[0].split()[0])
        self.LoadLamba = float(JLoc[0].split()[1])
        Aux = []
        line = 1
        for _ in range(NSigs):
            Aux.append(float(JLoc[line].split()[0]))
            line += 1
        self.LoadSigmas = dc(Aux)
        self.LoadModel  = pickle.load(open(MLModel.ModelPath+"{}.model".format(iName), "rb"))
        return LocObjList, LocDescInst

    # This function needs to pass on the sigmas to the DataTensObj for generating the PredTens on-the-fly.
    def predict_from_loaded(self, DataTensObj, MLObjList):
        """
        A method that predicts the data contained inside the Data Tensor Object. Uses the MLObjList to calculate the kernel matrix
        on-the-fly (i.e. more memory efficient). Stores the output inside the MLModel instance itself.
        (returns None)

        --- Parameters ---
        ------------------
        DataTensObj : data tensor instance
            A data tensor instance containing both training and prediction data initilized.

        MLObjList : list of MLEntry object instances
            A list of MLEntry instances that contain the raw data of both the training and prediction
            data. Allows on-the-fly calculation of the prediction kernel matrix.
        """
        # Get the To-Predict and Training set.
        LocTRIDs, LocPRIDs, Aux1, Aux2 = [], [], [], []
        for ii in range(DataTensObj.NEntries):
            LocStat = DataTensObj.Status[ii]
            if LocStat == "Training":
                LocTRIDs.append(ii)
            if LocStat == "Prediction":
                LocPRIDs.append(ii)
        # Use the get_pred_tens routine to generate the Prediction KMat.
        # This should on-the-fly generate the KMAT, without use of kernel_gen.
        KMAT = DataTensObj.get_pred_tens(MLObjList, LocTRIDs, LocPRIDs, self.LoadSigmas)
        # Generate the Training / Testing KMat's with the respective Sigmas.
        self.Predictions = self.LoadModel.predict(KMAT)
        return None
