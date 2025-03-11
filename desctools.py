# # # # # # # # # # # # #
# The Descriptor class gives a flexible object that can construct a set of requested Descriptors in a standardized manner from the native data format.
# It's only method is "Describe", which will attach the requested descriptors to the MLEntry object that it is used on.
#
# There is a general programming philosophy behind how the descriptors are attached to the MLEntries:
# For each newly defined descriptor, the feature is attached as a dictionary entry to the MLEntry alongside what form the feature has,
# and what needs to be done to compare two molecules.
#
# For example, if the feature is just the number of total electrons, the descriptor will use the raw data of the MLEntry Object to
# calculate the number of electrons. This is a scalar quantity - and each entry will have just one of these values. Hence, to compare
# to all molecules, we need to ultimately form a "direct" distance matrix.
#
# On the other hand, if we consider the Coulomb Matrix representation, we can generate an array of eigenvalues of the Coulomb Matrix, such
# that for this descriptor, every MLEntry Object will carry one Array of eigenvalues instead of a scalar. To compare two molecules to each other
# using the Coulomb Matrix eigenvalues, one then may form the Euclidean Norm between the eigenvalues in a specialized manner (acoounting for different lengths of arrays
# and filling up with zeros).
#
# All of these operations are later performed by the DataTens Class - which does not need to know what a specific descriptor is called; but just needs to know
# what it "should do with it". This way, the Descriptor class is free from the maths of how to generate the distance matrices and all and just has the definitions of
# each descriptor.
#
# FW 07/13/23

# packages required for "legacy" descriptors MaxConj.
import os
from copy import deepcopy as dc

import mendeleev as md
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdqueries

from .utils import dist, eig_coul, coulomb_mat, coulomb_mat_sing_sign, coulomb_mat_duo_abs
from .utils import coulomb_mat_duo_sign

# Pre-Generate the ZEffs via md once, then only use it as dictionary.
ZEffs = []
for ii in range(1, 100):
    At = md.element(ii)
    ZEffs.append(At.zeff())

class Descriptor:
    """
    A class that decorates MLEntries with descriptor values.

    Based on what descriptors were requested by the user, an object
    instance will add descriptor information to a provided MLEntry object.
    Here, a descriptor's value/list/etc. is accompanied by a meta-information
    on the descriptor's data format and an operation that is to be performed
    on the descriptor data, in case a distance between two MLEntries is to
    be calculated.

    --- Attributes ---
    ------------------
    DescL : a list of strings
        contains the requested descriptors.
        (this means also, that new descriptors only need to be added to this module)

    --- Methods ---
    ---------------
    describe(self, MLObject)
        An instance-specific method that will add attributes to the provided MLObject.
        These attributes contain the descriptor values as well as their meta-information
        on how to deal with them.
        (returns None)
    """

    def __init__(self, iDescL):
        """
        --- Parameters ---
        ------------------
        iDescL : a list of strings
            This list contains the descriptor names that the user wishes to add to all
            MLEntries.
        """
        self.DescL = iDescL

    # Take the MLObject and - depending on the IDescL - calculate different Descriptors.
    # Store the Descriptors inside the MLObject's Desc Library.
    def describe(self, MLObject):
        """
        A method that will add descriptor values to the provided MLObject.

        Depending on the user-defined list of requested descriptors, the respective
        descriptor value is calculated from raw data attached to the MLObject. The
        calculations are performed by functions inside the utils module.

        --- Parameters ---
        ------------------
        MLObject: an object instance of the MLEntry class

        --- Returns ---
        ---------------
        None        
        """

        if "SEmpOrbEnDiffs" in self.DescL:                                                        # a.k.a. "PM6Vecs", "PM6_OrbEns", "PM6EnDiff", "DistEnDiff"
            MLObject.Desc['SEmpOrbEnDiffs']     = dc(MLObject.Data['OccVirDif'])                  # This descriptor stores the Occ-Vir energy differences for all
                                                                                                  # allowed Jumps inside the OrbWin. It is an array of scalars per molecule.
                                                                                                  # Note that the array follows the succession of the Transition Tuples (Data['OccVirTup'])
            MLObject.DescType['SEmpOrbEnDiffs'] = ["ArrOfScal", "DistMat"]                        # Similarity function shall form distance matrix using normal distances.
            MLObject.DescL.append('SEmpOrbEnDiffs')
          
        if "SEmpOccs" in self.DescL:                                                              # a.k.a. "OccEn", "DistOccEn"
            MLObject.Desc['SEmpOccs'] = dc(MLObject.Data['OccMO'])                                # This descriptor stores Occupied Orbital energies within the OrbWin.
            MLObject.DescType['SEmpOccs'] = ["ArrOfScal", "DistMat"]                              # Disentangled statewise as an array of scalars.
                                                                                                  # Similarity function shall form distance matrix using normal distances.
            MLObject.DescL.append('SEmpOccs')

        if "SEmpVirs" in self.DescL:                                                              # a.k.a. "VirEn", "DistVirEn"
            MLObject.Desc['SEmpVirs'] = dc(MLObject.Data['VirMO'])                                # This descriptor stores Virtual Orbital energies within the OrbWin.
            MLObject.DescType['SEmpVirs'] = ["ArrOfScal", "DistMat"]                              # Disentangled statewise as an array of scalars.
                                                                                                  # Similarity function shall form distance matrix using normal distances.
            MLObject.DescL.append('SEmpVirs')

        if "SEmpTups" in self.DescL:
            MLObject.Desc['SEmpTups'] = dc(MLObject.Data['OccVirTup'])                            # This descriptor stores the specific IDs of HOMO+n and LUMO-m for the
            MLObject.DescType['SEmpTups'] = "unused"                                              # transitions as defined in SEmpOrbEnDiffs and other transition relevant
                                                                                                  # descriptors. It is not used directly though.
            MLObject.DescL.append('SEmpTups')

        if "SEmpEigCoul" in self.DescL:                                                           # a.k.a "EigCouls", "DistCMat"
            MLObject.Desc['SEmpEigCoul'] = dc(eig_coul(coulomb_mat(MLObject.GeomH)))              # This descriptor stores the eigenvalues of the Coulomb-Matrix.
            MLObject.DescType['SEmpEigCoul'] = ["EigCoulVec", "EucDistMat"]                       # Similarity function shall form a distance matrix using the euclidean norm.
            MLObject.DescL.append('SEmpEigCoul')

        if "SEmpOccEigCoul" in self.DescL:                                                        # a.k.a. "EigOccDiffs" (for some reason...?)
            MLObject.Desc['SEmpOccEigCoul'] = dc(eig_coul(coulomb_mat_sing_sign(MLObject.GeomH,   # This descriptor stores the eigenvalues of the Coulomb-Matrix-like Mulliken Charges.
                                                          MLObject.Data['GeoMullQ'])))            # Similarity function shall form a distance matrix using the euclidean norm.
            MLObject.DescType['SEmpOccEigCoul'] = ["EigCoulVec", "EucDistMat"]
            MLObject.DescL.append('SEmpOccEigCoul')

        if "SEmpNEl" in self.DescL:                                                               # a.k.a. "DistNEl"
            M_NEl = 0                                                                             # This descriptor stores the total number of electrons in the system.
            for ii in range(len(MLObject.GeomH)):                                                 # i.e. This is one scalar per molecule.
                M_NEl        +=  MLObject.GeomH[ii, 3]                                            # Similarity function shall form distance matrix using normal distances.
            MLObject.Desc['SEmpNEl'] = dc(M_NEl)
            MLObject.DescType['SEmpNEl'] = ["Scalar", "DistMat"]
            MLObject.DescL.append('SEmpNEl')
        if "SEmpOccPCMEigCoul" in self.DescL:                                                     # a.k.a. "OccPCMEigC", "DistOccPCMat"
            Aux = []                                                                              # This descriptor stores the P-Orbital Character of each site as the eigenvalues of a
            for ii in range(len(MLObject.Data['OccPChar'])):                                      # Coulomb-Matrix-like object. This is stored state-wise; i.e. an array of arrays.
                LocAux = eig_coul(coulomb_mat_duo_abs(MLObject.GeomH,                             # Similarity function shall form a distance matrix using the euclidean norm.
                                  MLObject.Data['OccPChar'][ii]))
                Aux.append(LocAux)
            MLObject.Desc['SEmpOccPCMEigCoul'] = dc(Aux)
            MLObject.DescType['SEmpOccPCMEigCoul'] = ["ArrOfArr", "EucDistMat"]
            MLObject.DescL.append('SEmpOccPCMEigCoul')

        if "SEmpVirPCMEigCoul" in self.DescL:                                                     # a.k.a. "VirPCMEigC", "DistVirPCMat"
            Aux = []                                                                              # This descriptor stores the P-Orbital Character of each site as the eigenvalues of a
            for ii in range(len(MLObject.Data['VirPChar'])):                                      # Coulomb-Matrix-like object. This is stored state-wise; i.e. an array of arrays.
                LocAux = eig_coul(coulomb_mat_duo_abs(MLObject.GeomH,                             # Similarity function shall form a distance matrix using the euclidean norm.
                                  MLObject.Data['VirPChar'][ii]))
                Aux.append(LocAux)
            MLObject.Desc['SEmpVirPCMEigCoul'] = dc(Aux)
            MLObject.DescType['SEmpVirPCMEigCoul'] = ["ArrOfArr", "EucDistMat"]
            MLObject.DescL.append('SEmpVirPCMEigCoul')

        if "SEmpTransPCMEigCoul" in self.DescL:                                                   # a.k.a. "TransPCMEigC", "DistTransPCMat"
            Aux = []                                                                              # This descriptor stores rough overlap integrals based on P-character (hence "Transition")
            for ii in range(len(MLObject.Data["OccVirPTransChar"])):                              # as eigenvalues of a Coulomb-Matrix-like object. This is stored state-wise; i.e.
                LocAux = eig_coul(coulomb_mat_duo_sign(MLObject.GeomH,                            # an array of arrays.
                                  MLObject.Data['OccVirPTransChar'][ii]))                         # Similarity function shall form a distance matrix using the euclidean norm.
                Aux.append(LocAux)
            MLObject.Desc['SEmpTransPCMEigCoul'] = Aux
            MLObject.DescType['SEmpTransPCMEigCoul'] = ["ArrOfArr", "EucDistMat"]
            MLObject.DescL.append('SEmpTransPCMEigCoul')

        if "SEmpOccVirPTransSum" in self.DescL:                                                   # a.k.a. "PM6TransSums", "TransSums", "DistTSum"
            MLObject.Desc['SEmpOccVirPTransSum'] = MLObject.Data["OccVirPTransSum"]               # This descriptor stores state-to-state-wise sums of scalars.
            MLObject.DescType['SEmpOccVirPTransSum'] = ["ArrOfScal", "DistMat"]                   # Therefore has the shape of an array of scalars.
                                                                                                  # Similarity function shall form distance matrix using normal distances.
            MLObject.DescL.append('SEmpOccVirPTransSum')

        if "SEmpHOLUPDiff" in self.DescL:                                                         # a.k.a. "SpinGuess", "DistSpinGuess"
            LocAux = eig_coul(coulomb_mat_duo_abs(MLObject.GeomH, MLObject.Data['HOLUPDiff']))
            MLObject.Desc['SEmpHOLUPDiff'] = dc(LocAux)                                           # This descriptor stores the HOMO**2-LUMO**2 density as eigenvalues of a
            MLObject.DescType['SEmpHOLUPDiff'] = ["EigCoulVec", "EucDistMat"]                     # Coulomb-Matrix-like object. One such array for each molecule.
                                                                                                  # Similarity function shall form a distance matrix using the euclidean norm.
            MLObject.DescL.append('SEmpHOLUPDiff')
  
        ################################## Descriptor Section over - following the Label Section ######################################
        # For the time being, just assume that there exist files with the Label names in the format of
        #
        #  ID    value
        #
        # This way, the model really does not know (or care) what properties are tried to be fitted - or where they came from exactly.
        ###############################################################################################################################
  
        # Only assign Labels, if the Object is in fact a Training Entry!
        if MLObject.Type == "Training":
            for ii in range(len(MLObject.LabelL)):
                LocLabelL = MLObject.LabelL[ii]
                # Assume that Label-Files lie inside the MergeDirectory.
                FID  = open(MLObject.MergePath+LocLabelL, "r")
                line = FID.readline()
                while MLObject.ID not in line:
                    line = FID.readline()
                    if not line:
                        raise IndexError("Descriptor.describe could not find MLEntry {} in the label-file {}.".format(MLObject.ID, LocLabelL))
                MLObject.Label[LocLabelL] = float(line.split()[1])
                FID.close()

        # Free up the Raw Data space of Memory after describe has assigned the descriptor values
        MLObject.Data = {}
        return None
