# # # # # # # # # # # # #
# The Generator Module
#
# This module is responsible for setting up the database architecture in its required form.
# Also, it will serve as the Generator for calculations at a later point of program development.
#
# FW 07/13/23

import os
import time
import shutil
from copy import deepcopy as dc

import numpy as np
from pyquaternion import Quaternion

from .utils import dist, darr, timeconv
from .common import PSEDICT, PSEIDICT, PREDIST, GENNROT

# Reads in a an xyz file as both a Geom matrix as well as the pure XYZ file in string form (required for some visualization packages...)
def rdgeom(path):
    """
    A function to read an .xyz file.

    --- Parameters ---
    ------------------
    path : str
        The absolute path to the .xyz file in string.
        
    --- Returns ---
    ---------------
    Geom : matrix of floats
        A matrix containing the geometry of the xyz file.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number

    FLoc : list of str
        The xyz in plain text. Useful for rdkit's Chem visualization.
    """
    fid = open(path)
    FLoc  = fid.read().splitlines()
    fid.close()
    NAt   = int(FLoc[0].split()[0])
    Geom  = np.zeros((NAt, 4))
    cnt = 0
    for ii in range(2, len(FLoc)):
        Geom[cnt, 0] = float(FLoc[ii].split()[1])       # X Coord
        Geom[cnt, 1] = float(FLoc[ii].split()[2])       # Y Coord
        Geom[cnt, 2] = float(FLoc[ii].split()[3])       # Z Coord
        Geom[cnt, 3] = PSEDICT[FLoc[ii].split()[0]]     # Atom Type
        cnt += 1
    return Geom, FLoc

# This function calculates the basic nuclear repulsion beteen fragments for an initial structure guess.
# Note that it exaggerates very small distances.
def get_repuls(GeomA, GeomB):
    """
    A function to calculate a simple, modified nuclear repulsion between two molecular
    fragments. Used to find a local minimum while moving/rotating two parts against
    each other.

    Uses atomic numbers as nuclear charges directly and exaggerates repulsion by a
    fixed factor of 10, for parts of the fragments that become closer than 0.6
    Angstroms. This choice was made, as the normal repulsion would sometimes
    find minima with hydrogen atoms that were unreasonably close to each other.

    --- Parameters ---
    ------------------
    GeomA : matrix of floats
        A molecular geometry in the usual formatting.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number, [4] placeholder (currently unused)

    GeomB : matrix of floats
        The second molecular fragment with which to determine repulsion.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number, [4] placeholder (currently unused)

    --- Returns ---
    ---------------
    Repuls : float
        The modified nuclear repulsion of the current geometries.

    Joint : matrix of floats
        The joint geometry of both molecular fragments. Ordered with GeomA before GeomB.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number, [4] placeholder (uccrently unused)

    """
    AAtm = len(GeomA)
    BAtm = len(GeomB)
    Repuls = 0.0
    Joint = np.zeros((AAtm+BAtm, 5))
    for ii in range(AAtm):
        Joint[ii,0] = GeomA[ii,0]
        Joint[ii,1] = GeomA[ii,1]
        Joint[ii,2] = GeomA[ii,2]
        Joint[ii,3] = GeomA[ii,3]
        Joint[ii,4] = GeomA[ii,4]
    for jj in range(BAtm):
        Joint[ii+1+jj,0] = GeomB[jj,0]
        Joint[ii+1+jj,1] = GeomB[jj,1]
        Joint[ii+1+jj,2] = GeomB[jj,2]
        Joint[ii+1+jj,3] = GeomB[jj,3]
        Joint[ii+1+jj,4] = GeomB[jj,4]
    for ii in range(len(Joint)):
        Zii = Joint[ii, 3]
        for jj in range(ii+1, len(Joint)):
            try:
                Zjj = Joint[jj,3]
                Rij = dist(Joint[ii,:], Joint[jj,:])
                Repuls += (Zii*Zjj)/Rij
                if Rij < 0.6:                                   # exaggerate the repulsion for very low distances - since H atoms are only Z=1,
                    Repuls += 10*(Zii*Zjj)/Rij                  # they sometimes come too close to each other for PM6 to handle.
            except IndexError:
                Repuls += 0
    return Repuls, Joint

# Rotation around a bond using quaternions.
def quater_rot(Vec, Pos, Geom, Angle):
    """
    A function performing a rotation of a molecular fragment around a vector using a
    quaternion.

    --- Parameters ---
    ------------------
    Vec : array of 3 floats [x, y, z]
        The vector around which shall be rotated. This is usually a bonding direction.

    Pos : array of 3 floats [x, y, z]
        The final position, where the rotated molecule should be placed at.

    Geom : matrix of floats
        The molecule (or molecular fragment) that is to be rotated and positioned.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number, [4] substituent tag

    Angle : float
        The angle of rotation for the quaternion.

    --- Returns ---
    ---------------
    RotGeom : matrix of floats
        The rotated molecule fragment's final positioning.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number, [4] substituent tag
    """
    NAtm = len(Geom)
    RotGeom = np.zeros((NAtm, 5))
    # Define Quaternion rotation
    RotAx = Quaternion(axis=[Vec[0], Vec[1], Vec[2]], angle=2*Angle)
    for ii in range(NAtm):
        AuxM = np.zeros((3))
        RotM = np.zeros((3))
        AuxM[0] = Geom[ii][0] - Pos[0]
        AuxM[1] = Geom[ii][1] - Pos[1]
        AuxM[2] = Geom[ii][2] - Pos[2]
        RotM = RotAx.rotate(AuxM)
        RotGeom[ii][0] = RotM[0] + Pos[0]
        RotGeom[ii][1] = RotM[1] + Pos[1]
        RotGeom[ii][2] = RotM[2] + Pos[2]
        RotGeom[ii][3] = Geom[ii][3]
        RotGeom[ii][4] = Geom[ii][4]
    return RotGeom


# This function puts together two different molecules at a user predefined bond.
def junction(CoreSubPos, CurFrag, CurrentCore, iCORE, iFLAGS):  # Careful - uses CORE and FLAGS as global variables!
    """
    A function putting replacing a general substituent site inside
    the core structure with a substituent.

    This function will take the current core structure, and substitute
    the "X" and "Y" labeled atoms with the new substituent molecular
    fragment. This is performed by rotating the fragment such that
    it will point opposite of the original X->Y bond with Rodrigues'
    rotation theorem. Then it is "placed down" with an initial guess
    distance (taken from common.py's PREDIST dictionary) and rotated
    a few times around the newly formed bond to determine an initial
    best guess w.r.t. to nuclear repulsion.

    --- Parameters ---
    ------------------
    CoreSubPos : str
        Currently considered n-th substituent position with respect to the
        !Xn / !Yn commentaries as given in the CoreStructure.xyz file.

    CurFrag : dict
        Dictionary containing data on the current substituent fragment.
        Keys are:
            "ID" : str
                The internal name of the substituent as taken from the
                substituent library file
            "Num" : str
                The internal numerical ID that will be reflected also
                in the fingerprint.
            "Path" : str
                The path that leads to the 'Opted.xyz' and 'Parent.xyz'
                files, which determine how to attach the substituent.
            "Geom" : matrix of floats
                A matrix containing the atoms as given in the 'Opted.xyz'
                file. This is the _clean_ structure.
                [0] x-position, [1] y-position, [2] z-position, [3] atomic number
            "XYZ" : list of str
                The .xyz file as in plain text - for visualization with rdkit
            "Tags" : matrix of floats
                A matrix containing the atoms as given in the 'Parent.xyz'
                file. This reflects where the new bond is to be formed.
                [0] x-position, [1] y-position, [2] z-position, [3] atomic number

    CurrentCore : matrix of floats
        The current state of the substitued core. In principle, the junction
        function is called several times over, until all substituent positions have
        been changed, so this matrix contains the updating structure.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number

    iCORE : matrix of floats
        The original core structure in matrix format.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number

    iFLAGS : list of str
        A list of the commentaries' '!Xn' / '!Yn' inside the CoreStructure.xyz.
        Has either the string 'Xn', 'Yn' or '0'.

    --- Raises ---
    --------------
    KeyError
        Raises an error, if the function tries to access an initial guessing distance
        from the common.PREDIST dictionary that has not been defined yet.


    --- Returns ---
    ---------------
    Junctioned : matrix of floats
        The 'minimum repulsion' structure, where the substituent was attached to the
        current core.
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number
    """
    Origin     = np.zeros(3)
    # Step 1 - Get the connecting atom's vectors for both Core and Fragment
    XString = "X"+str(int(CoreSubPos))
    YString = "Y"+str(int(CoreSubPos))
    CoreXID = np.where(np.asarray(iFLAGS) == XString)[0][0]
    CoreYID = np.where(np.asarray(iFLAGS) == YString)[0][0]
    FragXID = np.where(0  == CurFrag["Tags"][:, 3])[0][0]
    FragYID = np.where(39 == CurFrag["Tags"][:, 3])[0][0]    
    CurXAt = CurFrag["Tags"][FragXID]
    CurYAt = CurFrag["Tags"][FragYID]
    XDArr  = darr(CurXAt, CurFrag["Geom"])
    YDArr  = darr(CurYAt, CurFrag["Geom"])
    FragMapXID = np.argsort(XDArr)[0]
    FragMapYID = np.argsort(YDArr)[0]
    # Get a general MAPPING for later generation of RotFrag atom information...
    MAP_TtoG = np.zeros(len(CurFrag["Tags"]))
    MAP_GtoT = np.zeros(len(CurFrag["Geom"]))
    for ii in range(len(CurFrag["Tags"])):
        CurAtT = CurFrag["Tags"][ii]
        CurTID = ii
        CurGID = np.argsort(darr(CurAtT, CurFrag["Geom"]))[0]
        MAP_TtoG[CurTID] = CurGID
        MAP_GtoT[CurGID] = CurTID

    # Step 2 - Get normed X-Y Vectors - from the tagged ones is fine.
    CoreXYVec = np.zeros(3)
    FragXYVec = np.zeros(3)
    for ii in range(3):
        CoreXYVec[ii] = iCORE[CoreXID, ii] - iCORE[CoreYID, ii]
        FragXYVec[ii] = CurFrag["Tags"][FragXID, ii] - CurFrag["Tags"][FragYID, ii]
    CoreVecNorm = dist(Origin, CoreXYVec)
    FragVecNorm = dist(Origin, FragXYVec)
    CoreXYVec /= CoreVecNorm
    FragXYVec /= FragVecNorm
 
    # Step 3 - Rotation of Fragment (untagged) to point towards the Core structure.
    # Apply Rodrigues' Rotation Theorem to find the rotation matrix.
    # Get normalized Vectors perpendicular to both XY vectors.
    # Rotate the fragment around this vector.
    PerpVec  = np.cross([FragXYVec[0], FragXYVec[1], FragXYVec[2]], [-CoreXYVec[0], -CoreXYVec[1], -CoreXYVec[2]])
    AuxS     = dist(PerpVec,Origin)
    AuxC     = np.dot([FragXYVec[0], FragXYVec[1], FragXYVec[2]], [-CoreXYVec[0], -CoreXYVec[1], -CoreXYVec[2]])
    # Check if AuxC is exactly -1.0 ; which is the case, when the vectors point in opposite directions already.
    if AuxC != -1.0:
        # Construct Auxiliary Matrix K, Identity Matrix I
        IMat = np.zeros((3,3))
        IMat[0,0], IMat[1,1], IMat[2,2] = 1, 1, 1
        KMat = np.zeros((3,3))
        KMat[0,0] =  0
        KMat[0,1] = -PerpVec[2]
        KMat[0,2] =  PerpVec[1]
        KMat[1,0] =  PerpVec[2]
        KMat[1,1] =  0
        KMat[1,2] = -PerpVec[0]
        KMat[2,0] = -PerpVec[1]
        KMat[2,1] =  PerpVec[0]
        KMat[2,2] =  0
        # Construct Rotation Matrix to rotate TagVec onto -UnTagVec
        RMat = np.zeros((3,3))
        RMat = IMat+KMat+np.dot(KMat, KMat)*(1 / (1+AuxC))
        # Rotate the TAGGED fragment Geometry accordingly.
        RotFrag = np.zeros((len(CurFrag["Tags"]), 5))
        for ii in range(len(CurFrag["Tags"])):
            AuxVec = np.matmul(RMat, [CurFrag["Tags"][ii,0], CurFrag["Tags"][ii,1], CurFrag["Tags"][ii,2]])
            RotFrag[ii][0] = AuxVec[0]
            RotFrag[ii][1] = AuxVec[1]
            RotFrag[ii][2] = AuxVec[2]
            RotFrag[ii][3] = CurFrag["Tags"][ii, 3]
            RotFrag[ii][4] = CoreSubPos
    # If vectors point exactly opposite, just overwrite the RotFrag directly from the CurFrag.
    else:
        RotFrag = np.zeros((len(CurFrag["Tags"]), 5))
        for ii in range(len(CurFrag["Tags"])):
            RotFrag[ii][0] = CurFrag["Tags"][ii, 0]
            RotFrag[ii][1] = CurFrag["Tags"][ii, 1]
            RotFrag[ii][2] = CurFrag["Tags"][ii, 2]
            RotFrag[ii][3] = CurFrag["Tags"][ii, 3]
            RotFrag[ii][4] = CoreSubPos

    # Position the TAGGED Fragment at the Core structure with an initial bond distance guess.
    # Identify what bond it's going to be. HERE - mapping is needed, since tagged one doesn't know what atom X is goinhg to become.
    AtmTypes = np.sort([int(iCORE[CoreXID, 3]), int(CurFrag["Geom"][FragMapXID, 3])])
    BondType = PSEIDICT[AtmTypes[0]] + PSEIDICT[AtmTypes[1]]
    try:
        XTargetPos = iCORE[CoreXID, :3] - PREDIST[BondType] * CoreXYVec
    except KeyError:
        raise KeyError("An unknown bond type was tried to connect in Generator.Junction ({}). Please add the bond with an initial guess distance to the common.py PreDist dictionary.".format(BondType))
    XShift     = RotFrag[FragXID, :3] - XTargetPos                         # RotFrag IS the tagged geometry, thus, shift the unmapped ID.
    for ii in range(len(RotFrag)):
        RotFrag[ii][0] -= XShift[0]
        RotFrag[ii][1] -= XShift[1]
        RotFrag[ii][2] -= XShift[2]
    FragGeom = dc(RotFrag)
    CoreGeom = dc(CurrentCore)
    # Re-Substitute the X of the FragGeom for the correct atom
    FragGeom[FragXID, 3] = CurFrag["Geom"][FragMapXID, 3]
    # Remove the Y atoms in both Subunits. Careful - in CoreGeom, check for the Y atom via distance-argument-comparison to original CORE.
    FragGeom = np.delete(FragGeom, FragYID, 0)
    YDArr = darr(iCORE[CoreYID, :3], CoreGeom)
    CoreGeom = np.delete(CoreGeom, np.argsort(YDArr)[0], 0)
    # Calculate the total nuclear repulsion as a function of the rotation around the bond in 1 degree increments
    XRot      = np.linspace(-np.pi/2, np.pi/2, GENNROT)
    NucRepuls = np.zeros((len(XRot)))
    Coupled   = []
    for ii in range(len(XRot)):
        Rotated            = quater_rot(-CoreXYVec, XTargetPos, FragGeom, XRot[ii])
        NucRepuls[ii], Aux = get_repuls(CoreGeom, Rotated)
        Coupled.append(Aux)
    # Finally, use the minimum repulsion result as Guess input for further steps.
    ID         = np.argsort(NucRepuls)[0]
    Junctioned = dc(Coupled[ID])
    return Junctioned

# Wrapper function for running Junctions in recursive parallel mode
# def FullJuncOnce(LocFing, iCORE, iFRAGMENTS, iNSub, iFLAGS)
def full_junc_once(TUP):
    """
    A wrapper function for running the function junction
    in recursive, parallel fashion.

    --- Parameters ---
    ------------------
    TUP : tuple of inputs
        Contains [0] - the current full fingerprint of the entry
                       (list of floats)

                 [1] - the unaltered core structure
                       (matrix of floats)

                 [2] - list of dictionaries of all available substituents.
                       (described in more detail for the Generator class)

                 [3] - the number of overall possible substituent positions.
                       (int)

                 [4] - the core structure's substitution pattern in terms of
                       the commentaries 'Xn', 'Yn' or '0'
                       (list of strings)

    --- Returns ---
    ---------------
    LocJunc : matrix of floats
        The final, fully substituted structure belonging to the LocFing fingerprint.

    LocFing : list of floats
        A copy of the used fingerprint for this full junction.
        (originally intended for debugging purposes)

    """
    LocFing    = TUP[0]
    iCORE      = TUP[1]
    iFRAGMENTS = TUP[2]
    iNSub      = TUP[3]
    iFLAGS     = TUP[4]
    state = 1                                                    # "state" starts at 1, since X and Y are indexed with 1
    LocJunc = dc(iCORE)
    while state < iNSub+1:
        FRAGID  = int(LocFing[state-1])-1                           # Fingerprints have no 0, but FRAGMENTS will count from 0, obv.
        OutJunc = junction(state, iFRAGMENTS[FRAGID], LocJunc, iCORE, iFLAGS)    
        LocJunc = dc(OutJunc)
        state  += 1
    return LocJunc, LocFing


class DBGenerator:
    """
    An object that manages generation of inputs and folder hierarchy for
    your project.

    The DBGenerator class will generate the initial fingerprints based off your provided
    input files (CoreStructure + substituent library -- or -- Conformations trajectory).
    Further, it can select a random sample of the full chem./conf. space and generate
    the calculation inputs for the external quantum-chemistry package you selected. The
    generator will be useful, whenever you decide to add new structures to your database.

    --- Attributes ---
    ------------------
    ProjName : str
        Name of the project.

    FragStrucLib : str
        Path to the substituent library file.

    CoreStrucFID : str
        Path to the Core structure (chemical screen) or conformer trajectory (conformational screen).

    StrucLib : str
        Path to the file that will contain all possible names and fingerprints of the entries of your database.

    MergePath : str
        Path to the folder that will contain the raw data in ArchOnML's native format. Also, the program
        will expect to find the files with label values in this folder.

    GuessPath : str
        Path to the folder in which the original guess structures will be stored.

    DBPath : str
        Path to the Database folder in which all quantum chemical calculations are.
        This path does not contain the project name (yet).

    Fold_L1 : int
        An integer describing how the database is structured. Used to (re-)create the
        paths to the individual calculations based off the fingerprint.
        "First level" of the path string.

    Fold_L2 : int
        An integer describing how the database is structured. Used to (re-)create the
        paths to the individual calculations based off the fingerprint.
        "Second level" of the path string.

    ConfScanFlag : bool
        A boolean that determines whether this is a chemical space screen or conformational
        space screen. True if conformational space.

    QCDBPath : str
        Path to the project-specific database folder (i.e. a subfolder of the DBPath)

    CORE : matrix of floats
        Contains the structure of the CoreStructure (in case of a chemical space screen).
        [0] x-position, [1] y-position, [2] z-position, [3] atomic number

    FLAGS : list of str
        Contains the "commentary flags" inside the CoreStructure.xyz file. For each atom
        of the core structure, a flag reading either 'Xn', 'Yn' or '0' is placed in here.
        'X' and 'Y' refer to the start and end of a substituent bond, and 'n' refers to the
        n-th substituent position. A '0' means that this atom is not to be modified.

    NSub : int
        The number of overall substituent positions.

    FRAGMENTS : list of dict
        For each substituent, a dictionary is created. These contain:
            'ID' : str
                The name of the substituent as given in the substituent library.
            'Num' : str
                The number that was assigned to this substituent. These have to be unique.
                The number will be used in encoding the fingerprint (chemical space screen).
            'Path' : str
                The path to the folder that contains the 'Opted.xyz' and 'Parent.xyz' files
                for this substituent.
            'Geom' : matrix of floats
                Contains the structure of the substitutent as taken from 'Opted.xyz'.
                [0] x-position, [1] y-position, [2] z-position, [3] atomic number
            'XYZ' : list of str
                The xyz-file in plain text. Useful for visualization with rdkit.
            'Tags' : matrix of floats
                Contains the structure of the substituent as taken from 'Parent.xyz'.
                This means, that the start and end of the to-be-replaced bond are encrypted
                as 'XX' (i.e. atomic number 0) and 'Y' (i.e. atomic number 39)
                [0] x-position, [1] y-position, [2] z-position, [3] atomic number

    NFrag : int
        Total number of possible substituent fragments that could be attached.

    FINGERPRINTS : list of lists of floats
        For each possible entry of the database, one list of floats is generated that contains
        what structure is to be used for this entry.
        In case of a chemical space screen, each list has a length of NSub, where the [ii]-th
        position refers to your ii-th substituent position (or R1, R2, R3 as drawn in Lewis
        structures). This position then carries a number, that refers to which substituent was
        placed here. The number refers to the FRAGMENTS[jj]['Num'] directly.
        In case of a conformational space screen, each list has a length of 3, where [0] and [1]
        just carry the information in which folder to find this structure (without any meaning).
        The last position of the list then points to the n-th structure inside the ConformerStructures.xyz


    The following dictionaries are used for mapping purposes.
    Since the user may reduce the number FINGERPRINTS with their own (symmetry) rules, the internal
    pointers may end up to need additional mapping from the original sorting to the new one.
    For this purpose, the GuessLib should be used as the static list of "final" possible Fingerprints.

    FingToID : dict
        A mapping dictionary that will map a fingerprint to a ID (that is, a name like 'PROJ121').
    FingToII : dict
        A mapping dictionary that will map a fingerprint to the internal numerical pointer of the
        Fingerprint list.
    IIToID : dict
        A mapping dictionary that will map the internal numerical pointer II to a name like
        'PROJ121'
    IIToFing : dict
        A mapping dictionary that will map the internal numerical pointer II to a fingerprint.
    SamDict : dict
        A mapping dictionary for the SampleLib, i.e. the static file that contains all entries
        that had been sampled at any point in the past.

    --- Methods ---
    ---------------
    gen_fingers(self)
        A method that recursively generates all possible fingerprints from the number of substituent
        positions (NSub) and the number of available substituent fragments (NFrag).
        This is the _initial_ list of Fingerprints, which should be reduced by the user w.r.t. to
        symmetry, eventually.
        In case of a conformational space screen, it will instead use the provided Fold_L1 and L2
        to recursively assign folder positions to each available conformer of the Conformers.xyz file.
        (returns None)

    gen_db_folders(self)
        A method that will generate all paths for the QC calculations, guess structure storage and
        merged raw-data files.
        (returns None)

    sample(iSampLib, iLocSetLib, iNSamp, iFixedSeed=False)
        A method that picks out iNSamp random samples from all available Fingerprints/IDs and then
        writes the selection to the iLocSetLib file.
        Further, it will add the selected samples to the static iSampLib file, which will keep track
        of all the entries that have been sampled previously as well.
        By setting the iFixedSeed boolean to True, a fixed seed will be used for randomized selection.
        (returns None)

    gen_calcs(iGenLib, iQCPack, iCalType, iCalFlav, iCalPathLib, **kwargs)
        A wrapper method that sets up the generator w.r.t. the selected external quantum-chemistry package
        iQCPack for all the samples contained in the iGenLib. The format of the iGenLib is the same one
        as is produced by the sample function. It will place the calculation inputs w.r.t. the QCDBPath
        and the individual entry's stem folder location.
        The wrapper loads a module from the calcgen subpackage for the actual generation and passes on the selected
        calculation type and flavour, as well as additional options in the keyword arguments.
        Finally, it writes all relative locations of the entries' stem folders to the iCalPahLib file.
        (returns None)
    """
    def __init__(self, iProjName, iFragStrucLib, iCoreStrucFID, iStrucLib, iMergePath, iGuessPath, iDBPath, iFold_L1, iFold_L2):
        """
        --- Parameters ---
        ------------------
        iProjName : str
            Name of the project that this Generator instance shall be working on.

        iFragStrucLib : str
            Path to the substituent fragment library file. If [] is provided, then the generator will assume that a
            conformer scan should be performed.

        iCoreStrucFID : str
            Path to the core structure file.

        iStrucLib : str
            Path to the Guess structure library file.

        iMergePath : str
            Path to the folder containing the merged raw-data in native format.

        iGuessPath : str
            Path to the folder containing the Guess structures.

        iDBPath :str
            Path to the folder containing the QC calculations.

        iFold_L1 : int
            An integer describing how the database is structured. Used to (re-)create the
            paths to the individual calculations based off the fingerprint.
            "First level" of the path string.

        iFold_L2 : int
            An integer describing how the database is structured. Used to (re-)create the
            paths to the individual calculations based off the fingerprint.
            "Second level" of the path string.
        """
        self.ProjName     = iProjName
        self.FragStrucLib = iFragStrucLib
        self.CoreStrucFID = iCoreStrucFID
        self.StrucLib     = iStrucLib
        self.MergePath    = iMergePath
        self.GuessPath    = iGuessPath
        self.DBPath       = iDBPath
        self.Fold_L1      = iFold_L1
        self.Fold_L2      = iFold_L2
        self.ConfScanFlag = False           # Flag that determines generator's behaviour.
                                            # If False, produce the Folder architecture according to a virtual screening of chemical space.
                                            # If True, produce the Folder architecture according to a virtual screening of conformer space.

        if iFragStrucLib == []:             # Set Conformer Scan Flag to True, if no substituents are provided.
            self.ConfScanFlag = True

        # Dump to DBSpecs.
        try:
            os.mkdir(self.DBPath)
        except FileExistsError:
            print("Skipped creation of {}, since folder already exists.".format(self.DBPath))
        self.QCDBPath     = self.DBPath+"{}/".format(self.ProjName)       # SubPath for the QC-Database.
        # Dump the DBSpecs. These should be read by the MLEntry class.
        FID = open("./DBSpecs.cfg", "w")
        FID.write("# This file, as well as its name should not be changed.\n")
        FID.write("{}\n".format(self.Fold_L1))
        FID.write("{}\n".format(self.Fold_L2))
        FID.write("{}\n".format(self.MergePath))
        FID.write("{}\n".format(self.QCDBPath))
        if self.ConfScanFlag == False:
            FID.write("{}\n".format("Chem"))
        if self.ConfScanFlag == True:
            FID.write("{}\n".format("Conf"))
        FID.close()

# Function for identifying the possible Fingerprints from the Fragment library and CoreStructure
    def gen_fingers(self):
        """
        A method that recursively generates all possible fingerprints from the number of substituent
        positions (NSub) and the number of available substituent fragments (NFrag).

        This is the _initial_ list of Fingerprints, which should be reduced by the user w.r.t. to
        symmetry, eventually.
        In case of a conformational space screen, it will instead use the provided Fold_L1 and L2
        to recursively assign folder positions to each available conformer of the Conformers.xyz file.
        (returns None)
        """
        # Below, behaviour for scanning a chemical space.
        if self.ConfScanFlag == False:
            fid     = open(self.CoreStrucFID, "r")
            FLoc    = fid.read().splitlines()
            fid.close()
  
            NAt   = int(FLoc[0].split()[0])
            self.CORE  = np.zeros((NAt, 5))
            self.FLAGS = []
            GeoEndSec = NAt+2
            cnt = 0
            for ii in range(2, GeoEndSec):
                self.CORE[cnt, 0] = float(FLoc[ii].split()[1])       # X Coord
                self.CORE[cnt, 1] = float(FLoc[ii].split()[2])       # Y Coord
                self.CORE[cnt, 2] = float(FLoc[ii].split()[3])       # Z Coord
                self.CORE[cnt, 3] = PSEDICT[FLoc[ii].split()[0]]     # Atom Type
                try:
                    self.FLAGS.append(FLoc[ii].split("!")[1])        # Fragment Flag
                except IndexError:
                    self.FLAGS.append("0")                           # Fragment Flag
                cnt += 1
            # Identify, how many substituent connectors there are.
            self.NSub = 0
            for ii in self.FLAGS:
                if "X" in str(ii):
                    self.NSub += 1
  
            # Next, read the Fragments, their IDs and numbers.
            with open(self.FragStrucLib) as fid:
                FLoc = fid.read().splitlines()
            self.FRAGMENTS = []
            for line in FLoc:
                LocID   = line.split()[0]
                LocNum  = line.split()[1]
                LocPath = line.split()[2]
                LocGeom, LocXYZ = rdgeom(LocPath+"Opted.xyz")
                LocTags, _      = rdgeom(LocPath+"Parent.xyz")
                self.FRAGMENTS.append({"ID"  : LocID,
                                       "Num"  : LocNum,
                                       "Path" : LocPath,
                                       "Geom" : LocGeom,
                                       "XYZ"  : LocXYZ,
                                       "Tags" : LocTags})
            self.NFrag = len(self.FRAGMENTS)
  
            ### Preparation of Dataframe
            # Recursive generation of all possible Fingerprints. This will NOT care about molecular symmetry, so use with care. 
            # The number of possible permutations explodes pretty quickly, so better check beforehand that it is not too much to handle...
            # Symmetry and further generation rules may be implemented in an extra step after generating all fingerprints to eliminate redundancy.
            # Note - a singular produced .xyz file may require about 3 KB - so, 1 million generated structures will require about 3 GB.  
            # Generate all possible Fingerprints using all Substituent Positions and Fragments
            # For full generality, this needs to be written as a recursive algorithm, as number of substituent positions may vary.
            LocFing = np.ones(self.NSub)
            self.FINGERPRINTS = []
            # Start with zeroth substituent position, i.e. zeroth state.
            state = 0
            while state == 0:
                for ii in range(self.NFrag):
                    LocFing[state] = self.FRAGMENTS[ii]["Num"]
                    self.FINGERPRINTS.append(dc(LocFing))
                state += 1
            # Now go flexibly through all other positions, always using the currently available fingerprints to "grow" the new ones.
            while state < self.NSub:
                CurDB   = dc(self.FINGERPRINTS)
                for ii in range(len(CurDB)):
                    LocFing = dc(CurDB[ii])                              # Select one of the currently formed states...
                    CurFing = dc(LocFing[state])                         # ...and remember its value at the current state.
                    for jj in range(self.NFrag):
                        if str(int(CurFing)) != self.FRAGMENTS[jj]["Num"]:    # only change and save, if not a Fingerprint, already.
                            LocFing[state] = self.FRAGMENTS[jj]["Num"]        # alter the state...
                            self.FINGERPRINTS.append(dc(LocFing))             # ...and save to the Fingerprints.
                state += 1
            print("Identified {} possible substituent fingerprints for this Core structure and Fragment library.".format(len(self.FINGERPRINTS)))

        # Below, behaviour for scanning a conformer space.
        if self.ConfScanFlag == True:
            # In case of Conformer Scan, fingerprints will consist of just 3 entries.

            # First  entry contains Fold_L1 position.
            # Second entry contains Fold_L2 position.
            # Third  entry contains Nth structure count, referring to the original input file of all conformers.

            # Step 1 - Check the input conformer space and count how many structures there are.
            FID  = open(self.CoreStrucFID, "r")
            FLoc = FID.readlines()
            FID.close()

            NAt    = int(FLoc[0].split()[0])      # Get the number of atoms.
                                                  # NOTE - currently assumes that all individual structures have the same number of atoms!

            Lines  = len(FLoc)                    # Total number of lines.
            NStruc = int(Lines / (NAt + 2))       # Number of Structures. The "plus 2" refers to the typical two "title lines" for each structure.
            print("Found {} structures in Conformer file...".format(NStruc))

            # Distribute Fingerprints recursively.
            Fold1Cnt = 1                          # Set starting upper ("slower") Folder Level. Will be counting from 1 to Fold_L1.
            Fold2Cnt = 1                          # Set starting lower ("faster") Folder Level. Will be counting from 1 to Fold_L2.
            self.FINGERPRINTS=[]
            for ii in range(NStruc):
                # Check "fast running" Folder number
                # If Fold2Cnt has come "full circle" once, increment upper level once and reset lower level.
                if Fold2Cnt > self.Fold_L2:
                    Fold1Cnt += 1
                    Fold2Cnt  = 1
                # ...then check slower running Folder number.
                # If Fold1Cnt has come "full circle" once, reset both positions.
                if Fold1Cnt > self.Fold_L1:
                    Fold1Cnt = 1
                    Fold2Cnt = 1
                self.FINGERPRINTS.append([float(Fold1Cnt), float(Fold2Cnt), float(ii+1)])
                Fold2Cnt += 1
            print("Assigned {} fingerprints for the current conformer scan.".format(len(self.FINGERPRINTS)))
            print("There will be (on average) {} in each of the lowest level folders.".format(NStruc/(self.Fold_L1*self.Fold_L2)))

        return self.FINGERPRINTS

    # This function will set up all empty hierarchy folders w.r.t. the fingerprints. Does NOT create the ID-specific calculation folders.
    def gen_db_folders(self):
        """
        A method that will generate all paths for the QC calculations, guess structure storage and
        merged raw-data files.
        (returns None)
        """
        # Below, behaviour for scanning a chemical space.
        if self.ConfScanFlag == False:
            # Set up the Folder hierarchies for the Guess Structures, the MergeFiles and the QC-Database.
            # First, get all unique folder and subfolder names.
            L1Fs = []
            L2Fs = []
            for ii in range(len(self.FINGERPRINTS)):
                Aux = ""
                for jj in range(self.Fold_L1):
                    Aux += "{}_".format(int(self.FINGERPRINTS[ii][jj]))
                curL1 = Aux.rstrip(Aux[-1])
                Aux = ""
                for jj in range(self.Fold_L2):
                    Aux += "{}_".format(int(self.FINGERPRINTS[ii][jj]))
                curL2 = Aux.rstrip(Aux[-1])
                # Check if curL1 is in L1Fs
                addit = True
                for jj in range(len(L1Fs)):
                    if curL1 == L1Fs[jj]:
                        addit = False
                if addit:
                    L1Fs.append(curL1)
                # Check if curL2 is in L2Fs
                addit = True
                for jj in range(len(L2Fs)):
                    if curL2 == L2Fs[jj]:
                        addit = False
                if addit:
                    L2Fs.append(curL2)

            # Transform into relative paths / subpaths
            L0Paths = [self.MergePath, self.GuessPath, self.QCDBPath]
            L1Paths = []
            L2Paths = []
  
            # L1 Paths
            for ii in range(len(L1Fs)):
                curL1 = L1Fs[ii]
                L1Paths.append(curL1+"/")
      
            # L2 Paths
            for ii in range(len(L1Fs)):
                curL1 = L1Fs[ii]
                for jj in range(len(L2Fs)):
                    curL2 = L2Fs[jj]
                    L1atL2 = L2Fs[jj].split("_")[0]
                    if L1atL2 == curL1:
                        L2Paths.append(curL1+"/"+curL2+"/")

        # Below, behaviour for scanning a conformational space.
        if self.ConfScanFlag == True:
            # First, get all unique folder and subfolder names.
            L1Fs = []
            L2Fs = []
            for ii in range(len(self.FINGERPRINTS)):
                curL1 = "{}".format(int(self.FINGERPRINTS[ii][0]))
                curL2 = "{}_{}".format(int(self.FINGERPRINTS[ii][0]), int(self.FINGERPRINTS[ii][1]))
                addit = True
                # Check if curL1 is in L1Fs
                for jj in range(len(L1Fs)):
                    if curL1 == L1Fs[jj]:
                        addit = False
                if addit:
                    L1Fs.append(curL1)
                # Check if curL2 is in L2Fs
                addit = True
                for jj in range(len(L2Fs)):
                    if curL2 == L2Fs[jj]:
                        addit = False
                if addit:
                    L2Fs.append(curL2)

            # Transform into relative paths / subpaths
            L0Paths = [self.MergePath, self.GuessPath, self.QCDBPath]
            L1Paths = []
            L2Paths = []
  
            # L1 Paths
            for ii in range(len(L1Fs)):
                curL1 = L1Fs[ii]
                L1Paths.append(curL1+"/")
      
            # L2 Paths
            for ii in range(len(L1Fs)):
                curL1 = L1Fs[ii]
                for jj in range(len(L2Fs)):
                    curL2 = L2Fs[jj]
                    L1atL2 = L2Fs[jj].split("_")[0]
                    if L1atL2 == curL1:
                        L2Paths.append(curL1+"/"+curL2+"/")
  
        # Make Directories
        # make the zeroth level directories
        for ii in range(len(L0Paths)):
            curL0Path = L0Paths[ii]
            os.mkdir(curL0Path)
            # First level directories
            for jj in range(len(L1Paths)):
                curL1Path = curL0Path+L1Paths[jj]
                os.mkdir(curL1Path)
            # Second level directories
            for jj in range(len(L2Paths)):
                curL2Path = curL0Path+L2Paths[jj]
                os.mkdir(curL2Path)
        return None

    # Method for sampling a number of random structures from the library.
    def sample(self, iSampLib, iLocSetLib, iNSamp, iFixedSeed=False):
        """
        A method that picks out iNSamp random samples from all available Fingerprints/IDs and then
        writes the current selection to the iLocSetLib file - and adds it to a static collection
        file iSampLib, that keeps track of all samples so far.

        By setting the iFixedSeed boolean to True, a fixed seed will be used for randomized selection.

        --- Parameters ---
        ------------------
        iSampLib : str
            Path to the static sampling library. This name should not be changed within one project.

        iLocSetLib : str
            Path to the current sample library. This is going to serve as an input for other commands.

        iNSamp : int
            The number of entries to be collected inside one sample.

        iFixedSeed : bool
            A control to fix randomness for your machine. Useful for debugging.

        (returns None)
        """
        # Generate a local Dictionaries of IDs versus Fingerprints and integers for fast lookup.
        # Map Fingerprints to IDs
        self.FingToID = {}
        # Map Fingerprint to II (this is for sorting the DB)
        self.FingToII = {}
        # Map II to ID
        self.IIToID   = {}
        # Map II to Fingerprint
        self.IIToFing = {}
        AuxFString, AuxID, AuxII= [], [], []
        for ii in range(len(self.FINGERPRINTS)):
            AuxID = "{}{}".format(self.ProjName, ii+1)
            AuxII = ii
            Aux = ""
            for jj in range(len(self.FINGERPRINTS[0])):
                Aux += "{},".format(int(self.FINGERPRINTS[ii][jj]))
            AuxFString = Aux.rstrip(Aux[-1])

            self.FingToID[AuxFString] = AuxID
            self.FingToII[AuxFString] = AuxII
            self.IIToID[AuxII]        = AuxID
            self.IIToFing[AuxII]      = AuxFString
  
        # open the static sample library, if available.
        Sampled = []
        self.SamDict = {}
        NewEnts = []
        SkipChk = False
        try:
            FID  = open(iSampLib, "r")
            FLoc = FID.readlines()
            FID.close()
            # Grab the Fingerprints of sampled structures and store as full strings.
            for ii in range(len(FLoc)):
                FString = FLoc[ii].split()[1]
                Sampled.append(FString)
                self.SamDict[FString] = dc(self.FingToID[FString])
        except FileNotFoundError:
            print("There seems to be no Sampled structures yet in this project. Starting the static samples library...")

        # Grab the requested number of random samples.
        # Initialize Randomness according to fixed seed policy.
        print("Reminder: This function contains a random number generator. Make sure to transfer the outcome consistently across machines for full reproducibility.")
        if iFixedSeed:
            print("...using a fixed seed!")
            rng = np.random.RandomState(seed=1337)
        if not iFixedSeed:
            print("...using system time for seeding!")
            rng = np.random.RandomState(seed=int(time.time()))                              # If not using a fixed seed, use system time for seeding.

        # Check if iNSamp should be deprecated - i.e. if the new sample would "overshoot" NEntries.
        NNew = len(Sampled) + iNSamp
        if NNew > len(self.FINGERPRINTS):
            NDep = NNew - len(self.FINGERPRINTS)
            iNSamp = dc(iNSamp - NDep)
            print("Requested number of samples would overshoot the remaining unsampled library. Reducing the requested sample size by {} (to {})".format(NDep, iNSamp))

        Loccnt = 0
        NEntries = len(self.FINGERPRINTS)
        while Loccnt < iNSamp:
            AddEntry = True
            LocPick  = rng.randint(0, NEntries)
            LocAux   = self.FINGERPRINTS[LocPick]
            Aux = ""
            for jj in range(len(LocAux)):
                Aux += "{},".format(int(LocAux[jj]))
            LocFing = Aux.rstrip(Aux[-1])

            # Try to grab the ID from the SamDict. If there is a KeyError, then the entry does not exist yet and should thus be added.
            try:
                Probe = self.SamDict[LocFing]
                AddEntry = False
            except KeyError:
                pass

            # Finally, if the AddEntry condition is still met...
            if AddEntry:
                Sampled.append(LocFing)
                self.SamDict[LocFing] = dc(self.FingToID[LocFing])
                NewEnts.append(LocFing)
                Loccnt += 1
  
        # Next, sort the entries just for tidyness sake. Then overwrite the old all-time Sampled Library with the new (re-sorted) one and write the local sample library.
        # Sorting and writing of "Sampled"
        SampIIs     = [self.FingToII[Sampled[ii]] for ii in range(len(Sampled))]
        SortSampIIs = np.sort(SampIIs)
        FID = open(iSampLib, "w")
        for ii in range(len(SortSampIIs)):
            SSS = "{}\t\t\t{}\n".format(self.IIToID[SortSampIIs[ii]], self.IIToFing[SortSampIIs[ii]])
            FID.write(SSS)
        FID.close()
        # Sorting and writing of "LocSetLib"
        LocSetIIs    = [self.FingToII[NewEnts[ii]] for ii in range(len(NewEnts))]
        SortLocSetIIs = np.sort(LocSetIIs)
        FID = open(iLocSetLib, "w")
        for ii in range(len(SortLocSetIIs)):
            SSS = "{}\t\t\t{}\n".format(self.IIToID[SortLocSetIIs[ii]], self.IIToFing[SortLocSetIIs[ii]])
            FID.write(SSS)
        FID.close()
        return None
 
    # This method shall generate the QC inputs. It loads the generator from the calcgen sub-module with respect to the specified QCPack.
    # Note, that it should transport **kwargs to the GenInp function as-is.
    def gen_calcs(self, iGenLib, iQCPack, iCalType, iCalFlav, iCalPathLib, **kwargs):
        """
        A wrapper method that sets up the generator for the actual input files and runs it.

        The wrapper loads a module from the calcgen subpackage for the actual generation and passes on the selected
        calculation type, flavour, target entry as well as additional options in the keyword arguments.
        Finally, it writes all relative locations of the entries' stem folders to the iCalPahLib file.
        (returns None)

        --- Parameters ---
        ------------------
        iGenLib : str
            The path to a sample library file (as in the LocSetLib above).

        iQCPack : str
            Name of the quantum chemistry package to use for generating inputs.

        iCalType : str
            Name of the calculation type to select for generating inputs.

        iCalFlav : str
            Name of the calculation flavor to select for generating inputs.

        iCalPathLib : str
            Path to the calculation library file. This file points to all generated database
            folders.

        --- Keyword Arguments ---
        -------------------------
        Depending on what calculation type was chosen, you can add arguments like the
        number of states, definitions for implicit solvents and such.
        For a list of possible arguments, consult the documentation.
        """

        # Select the calcgen sub-module
        if iQCPack == "g16":
            from .calcgen.gaussian import GenInp
        if iQCPack == "orca":
            from .calcgen.orca import GenInp

        # Read, what database entries to generate the calculations for.
        FID  = open(iGenLib, "r")
        FLoc = FID.readlines()
        FID.close()

        # Construct the DATABASE and Guess Paths for the specified Fingerprint with respect to the Fold_L1 and Fold_L2
        GenDBPaths  = []
        GenGuePaths = []
        GenIDs      = []
        FID = open(iCalPathLib, "w")
        for ii in range(len(FLoc)):
            LocID         = FLoc[ii].split()[0]
            LocFingString = FLoc[ii].split()[1]
            LocFing       = LocFingString.split(",")

            if self.ConfScanFlag == False:
                # Generate the Level 1 path depending on Fold_L1
                Aux = ""
                for jj in range(self.Fold_L1):
                    Aux += "{}_".format(LocFing[jj])
                LocPL1 = Aux.rstrip(Aux[-1])+"/"
                # Generate the Level 2 path depending on Fold_L2
                Aux = ""
                for jj in range(self.Fold_L2):
                    Aux += "{}_".format(LocFing[jj])
                LocPL2 = Aux.rstrip(Aux[-1])+"/"

            if self.ConfScanFlag == True:
                LocPL1 = "{}/".format(LocFing[0])
                LocPL2 = "{}_{}".format(LocFing[0], LocFing[1])+"/"

            # Points to the (maybe yet to create) stem-folder of the Entry.
            LocDBPath  = self.DBPath + self.ProjName + "/" + LocPL1 + LocPL2 + LocID + "/"
            # Points to the xyz-file of the Guess structure.
            LocGuePath = self.GuessPath + LocPL1 + LocPL2 + "Guess_" + LocFingString + ".xyz"
            # Write ID, Fingerprint and LocDBPath to File
            SSS = "{}   {}   {}\n".format(LocID, LocFingString, LocDBPath)
            FID.write(SSS)
   
            # Check, if folder needs to be created. If so, do so now and copy the Guess structure there.
            # May NOT overwrite the Guess.xyz in an already existing folder!
            try:
                os.mkdir(LocDBPath)
                shutil.copy(LocGuePath, LocDBPath + "Guess.xyz")
            except FileExistsError:
                # print("Skipped creation of {}, since folder already exists.".format(LocDBPath))
                pass
   
            # Finally, generate the requested Calculation using the generator function.
            GenInp(LocDBPath, iCalType, iCalFlav, **kwargs)
        FID.close()
        return None
