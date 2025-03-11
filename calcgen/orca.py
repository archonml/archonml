# # # # # # # # # # # # #
# This file contains the dictionaries of calculation types and some (so far used) calculation "flavors" for calculations set up with orca.
# It was tested and designed with the orca version 5.0.4.
#
# For an explanation on calculation "types" and "flavors", please refer to the documentation file in this folder.
#
# FW 07/31/23

import os

# This is the Input File generator function for orca version 5.0.4.
# NOTE - This function will create folders and files. Thus, it requires the respective I/O permissions in the target folders.
#        For flexible keyword-arguments like "number of states" and such, added an **kwargs list.
def GenInp(InpPath, CalType, CalFlav, **kwargs):
    curType = CType[CalType]                  # load the current calculation type from the dictionary of dictionaries.
    curFlav = FType[CalFlav]                  # load the current calculation flavor.

    # Load local vars of CType. note that gaussian calculations typically have only one input string.
    # This might be different in the future, if (for example) NBO calculations are included.
    # In this case, one would need to split the input string into "Pre-Geometry" and "Post-Geometry" lines.
    curFold = curType["FolName"]
    curInpN = curType["InpName"]
    curDumS = curType["DumString"]
    curGeom = curType["GeoFlag"]

    # Load the local Flavor.
    curMeth = curFlav["Method"]
    curBase = curFlav["Basis"]

    try:
        curJBas = curFlav["JBasis"]                 # Some Flavors may include an auxiliary J basis for RIJCOSX.
    except KeyError:
        pass

    # Load defaults for the keyword arguments. If not provided anywhere, these take over the job.
    curStat = 12

    # Load possible keyword arguments and overwrite the defaults.
    # For now, only include the number of states. Could also include CPCM models etc in the future...
    for kwarg in kwargs:
        if kwarg == "nstates":
            curStat = kwargs["nstates"]

        if kwarg == "epsilon":
            curEps = kwargs["epsilon"]

        if kwarg == "refrac":
            curRefr = kwargs["refrac"]

    # Now, start the calculation type dependent input generation.
    # Depending on the Calculation type, turn the Dummy String into the final Input String.

    # Call for a Pre-Optimization
    if CalType == "PreOpt":                                        # Internal TYPE A
        curInpS = curDumS.format(curMeth)                             # Only add the Pre Optimization Method, as basis sets are fixed for Semi-Empirics, usually.

    if CalType == "Opt":                                           # Internal TYPE B
        curInpS = curDumS.format(curMeth, curBase)                    # Add Method and Basis set.

#    if CalType == "DelSCF":                                        # Internal TYPE C
#        curInpS = curDumS.format(curMeth, curBase)                    # Add Method and Basis set.

    if CalType == "TDST":                                          # Internal TYPE D
        curInpS = curDumS.format(curMeth, curBase, curJBas, curStat)  # Add Method, Basis, JBasis and Nstates.

    # TYPE E is an unused legacy type.

    # TYPE F is an unused legacy type.

#    if CalType == "UHFBS":                                         # Internal TYPE G
#        curInpS = curDumS.format(curMeth, curBase)                    # Add Method and Basis set. Note that despite its name, it can do UDFT as well by specifying a different flavor.

    # TYPE H is an unused legacy type.

    # TYPE I is an unused legacy type.

    # TYPE J is an unused legacy type.

    # TYPE K is an unused legacy type.

    # TYPE L is an unused legacy type.

    # TYPE M is an unused legacy type.

    if CalType == "OrbEns":                                         # Internal TYPE N
        curInpS = curDumS.format(curMeth)                           # Only add the Pre Optimization Method, as basis sets are fixed for Semi-Empirics, usually.

    if CalType == "OrbEns_Solv_Conf":                               # Internal TYPE N
        curInpS = curDumS.format(curMeth, curEps, curRefr)          # Add PreOptimization method, as well as solvent EPSILON and REFRAC



    # TYPE O is an unused legacy type.

    # TYPE P was effectively a DelSCF with UPM6 flavor.

    if CalType == "TDSn":                                              # Internal TYPE Q
        curInpS = curDumS.format(curMeth, curBase, curStat)            # Add Method, Basis and Nstates.

    if CalType == "TDSn_Solv_Conf":                                           # Internal TYPE Q
        curInpS = curDumS.format(curMeth, curBase, curEps, curRefr, curStat)  # Add Method, Basis and Nstates.


    if CalType == "TDSnRIJ":                                           # Internal TYPE Q - RIJ version
        curInpS = curDumS.format(curMeth, curBase, curJBas, curStat)   # Add Method, Basis, JBasis and Nstates.

#    if CalType == "TDTn":                                             # Internal TYPE R
#        curInpS = curDumS.format(curMeth, curBase, curStat)           # Add Method, Basis and Nstates.
#        Not available in orca.

    # TYPE S is an unused legacy type.

    # Type T is an unused legacy type.

    # Type U is an unused legacy type.

    # Type V is an unused legacy type.

#    if CalType == "SOC":                                           # Internal TYPE W
#        curInpS = curDumS.format(curMeth, curBase, curStat)           # Add Method, Basis and Nstates.

    # Modify folder name with respect to the flavor and generate it.
    curFold = curFold.format(CalFlav)
    os.mkdir(InpPath+curFold)

    # Write the input string including the "charge multiplicity" line.
    FID = open(InpPath+"/"+curFold+"/"+curInpN, "w")
    FID.write(curInpS)

    # Add the Geometry in xyz format. For now, this is the only planned input style as scans in Z-Matrix don't seem useful for ML applications.
    if curGeom == "Guess":
        # use the Guess structure.
        GID  = open(InpPath+'Guess.xyz','r')
        GLoc = GID.readlines()
        GID.close()
        for line in range(2, len(GLoc)):  # orca does not like the commentary bangs. Remove.
            Str1 = GLoc[line].split()[0]
            Str2 = float(GLoc[line].split()[1])
            Str3 = float(GLoc[line].split()[2])
            Str4 = float(GLoc[line].split()[3])
            SSS = "{}   {:>10.7f}   {:>10.7f}   {:>10.7f}\n".format(Str1, Str2, Str3, Str4)
            FID.write(SSS)

    if curGeom == "PreOpted":
        # Use the PreOpted structure.
        GID  = open(InpPath+'PreOpted.xyz','r')
        GLoc = GID.readlines()
        GID.close()
        for line in range(2, len(GLoc)):
            FID.write(GLoc[line])

    if curGeom == "Opted":
        # Use the Opted structure.
        GID  = open(InpPath+'Opted.xyz','r')
        GLoc = GID.readlines()
        GID.close()
        for line in range(2, len(GLoc)):
            FID.write(GLoc[line])

    # Just finalize the file with "*\n" for now. There may be kwargs later that may change this behaviour (e.g. for NBO calculations).
    FID.write("*\n")
    FID.close()
    return None

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# This is a Dummy for a single Calculation Type Dictionary.
DummyCType  = {"FolName"   : "FolderName",
               "InpName"   : "InputFileName",
               "OutName"   : "OutputFileName",
               "DumString" : "Dummy Input String containing {} for methods or basis sets".format("Placeholders"),
               "GeoFlag"   : "Guess _OR_ PreOpted _OR_ Opted",
              }

# This is a Dummy for a calculation flavor.
DummyFlavor = {"Method" : "HF",
               "Basis"  : "def2-SVP"
              }

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# Definitions for Calculation Types
CType = {}


# TYPE A - Pre-Optimization
TypN = "PreOpt"
FNam = "A_Geo_PreOpt_{}"
INam = "Geo_PreOpt.inp"
ONam = "Geo_PreOpt.out"
DStr = '''! PM3 Opt

*xyz 0 1
'''
GFlg = "Guess"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}


# TYPE B - Optimization
TypN = "Opt"
FNam = "B_Geo_Opt_{}"
INam = "Geo_Opt.inp"
ONam = "Geo_Opt.out"
DStr = '''! {} {} OPT PAL6

*xyz 0 1
'''
GFlg = "PreOpted"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}


## TYPE C - Delta SCF (T1 energy via T ground state calculation)
#TypN = "DelSCF"
#FNam = "C_DelSCF_{}"
#INam = "DelSCF.com"
#ONam = "DelSCF.out"
#DStr = '''%chk=DelSCF.chk
#%mem=1024MB
#%NProcShared=6
##T {}/{} symmetry=none geom(nodistance,noangle,nodihedral)
##iop(6/7=2, 4/33=0, 2/9=1111, 2/11=2) 5D 7F
#
#Gaussian Triplet SP calculation at Singlet GS structure.
#
#0 3
#'''
#GFlg = "Opted"
#CType[TypN]  = {"FolName"   : FNam,
#                "InpName"   : INam,
#                "OutName"   : ONam,
#                "DumString" : DStr,
#                "GeoFlag"   : GFlg}


## TYPE D - TD-DFT with Singlets/Triplets ("TDST")
TypN = "TDST"
FNam = "D_TDST_{}"
INam = "TDST.inp"
ONam = "TDST.out"
DStr = '''! {} {} TightSCF RIJCOSX {} PAL6

%tddft
 nroots {}
 maxdim 5
 Tprint 0.0001
 triplets true
end

*xyz 0 1
'''
GFlg = "Opted"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}


## TYPE G - Unrestricted calculation with broken symmetry for natural populations. ("UHFBS")
#TypN = "UHFBS"
#FNam = "G_UHFBS_{}"
#INam = "UHFBS.com"
#ONam = "UHFBS.out"
#DStr = '''%chk=EXX.chk
#%mem=1024MB
#%NProcShared=6
##p {}/{} GFINPUT GFPRINT iop(6/7=3,9/40=5) 5D 7F guess=mix pop=NaturalOrbitals symmetry=none
##scf(conver=7, maxcycle=256)
#
#Gaussian Unrestricted BS calculation Routine using GS geometry.
#
#0 1
#'''
#GFlg = "Opted"
#CType[TypN]  = {"FolName"   : FNam,
#                "InpName"   : INam,
#                "OutName"   : ONam,
#                "DumString" : DStr,
#                "GeoFlag"   : GFlg}


# TYPE N - Semi-Emp Calculation of Orbital energies from PreOpted structure.
TypN = "OrbEns"
FNam = "N_OrbEns_{}"
INam = "OrbEns.inp"
ONam = "OrbEns.out"
DStr = '''! {} Printbasis PrintMOs Mulliken

*xyz 0 1
'''
GFlg = "PreOpted"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}

# TYPE N - Semi-Emp Calculation of Orbital energies from PreOpted structure - including implicit solvens
TypN = "OrbEns_Solv_Conf"
FNam = "N_OrbEns_{}"
INam = "OrbEns.inp"
ONam = "OrbEns.out"
DStr = '''! {} Printbasis PrintMOs Mulliken CPCM

%cpcm EPSILON {}
      REFRAC  {}
end

*xyz 0 1
'''
GFlg = "Guess"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}


# TYPE Q - Calculation of Singlet Excited States using TDDFT at the Opted Structure.
TypN = "TDSn"
FNam = "Q_TDSn_{}"
INam = "TDSn.inp"
ONam = "TDSn.out"
DStr = '''! {} {} TightSCF PAL6

%tddft
 nroots {}
 maxdim 5
 Tprint 0.0001
end

*xyz 0 1
'''
GFlg = "Opted"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}


# TYPE Q - Calculation of Singlet Excited States using TDDFT.
# Note, that this will use the "Guess" structure from a conformer scan trajectory.
TypN = "TDSn_Solv_Conf"
FNam = "Q_TDSn_{}"
INam = "TDSn.inp"
ONam = "TDSn.out"
DStr = '''! {} {} TightSCF PAL6 CPCM

%cpcm EPSILON {}
      REFRAC  {}
end

%tddft
 nroots {}
 maxdim 5
 Tprint 0.0001
end

*xyz 0 1
'''
GFlg = "Guess"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}



# TYPE Q - Calculation of Singlet Excited States using TDDFT at the Opted Structure.
TypN = "TDSnRIJ"
FNam = "Q_TDSn_{}"
INam = "TDSn.inp"
ONam = "TDSn.out"
DStr = '''! {} {} TightSCF RIJCOSX {} PAL6

%tddft
 nroots {}
 maxdim 5
 Tprint 0.0001
end

*xyz 0 1
'''
GFlg = "Opted"
CType[TypN]  = {"FolName"   : FNam,
                "InpName"   : INam,
                "OutName"   : ONam,
                "DumString" : DStr,
                "GeoFlag"   : GFlg}


## TYPE R - Calculation of Singlet Excited States using TDDFT at the Opted Structure.
# Not available in orca

## TYPE W - Calculation of Singlet Excited States using TDDFT at the Opted Structure.
## Note, that the recommended calculation flavor (from the pySOC devs) would be using the wB97XD functional and a TZVP basis set.
#TypN = "SOC"
#FNam = "W_SOC_{}"
#INam = "SOC.com"
#ONam = "SOC.out"
#DStr = '''%rwf=gaussian.rwf
## {}/{} td(50-50, nstates={}) 6D 10F nosymm GFInput
#
#Calculation for later pySOC calculation. Uses GS optimized Structure.
#
#0 1
#'''
#GFlg = "Opted"
#CType[TypN]  = {"FolName"   : FNam,
#                "InpName"   : INam,
#                "OutName"   : ONam,
#                "DumString" : DStr,
#                "GeoFlag"   : GFlg}



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# Definitions for Calculation Flavors
FType = {}

# PM6 Flavor - load the PM6 Semi-Empirical method. Does not require a Basis set definition.
FType["PM3"] = {"Method" : "PM3",
                "Basis"  : ""}


# CB3LG Flavor - CAM-B3LYP with Pople Gaussians
FType["CB3LG"] = {"Method" : "CAM-B3LYP",
                  "Basis"  : "6-31G*"}


# CB3LD Flavor - CAM-B3LYP with Karlsruhe set
FType["CB3LD"] = {"Method" : "CAM-B3LYP",
                  "Basis"  : "def2-SVP",
                  "JBasis" : "def2/J"}


## SOCRec Flavor - recommended Flavor to use with pySOC.
#FType["SOCRec"] = {"Method" : "wB97XD",
#                   "Basis"  : "TZVP"}
