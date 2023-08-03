# The entire data flow for my analysis

# 1. Read in the data

# The data are stored in a ROOT file AnalysisResults.root

# The central data are stored in an AliEmcalList called AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2Centralq

# The associated track data are stored in a THnSparse called fhnJH

# The mixed event data are stored in a THnSparse called fhnMixedEvent

# The set of jets that pass my track and cluster bias are stored in a THnSparse called fhnTrigger

# 2. Subtract mixed event data from central data over dEta, dPhi

# This should be the correlation function

# store the correlation function in a THnSparse called fhnJHCorr along with the associated track data

# there may be some fancy details like matching event plane angle and multiplicity, etc. I need to read http://cds.cern.ch/record/2721341/files/CERN-THESIS-2019-357.pdf

import os
import ROOT
from JetHadron import JetHadron
from RPF import RPF
from matplotlib.colors import LogNorm
from itertools import product
from fpdf import FPDF
from templateFit import templateFit


fhnJH_axis_labels = {
    key: val
    for key, val in enumerate(
        [
            "V0 centrality (%)",
            "Jet p_{T}",
            "Track p_{T}",
            "#Delta#eta",
            "#Delta#phi",
            "Leading Jet",
            "Event plane angle",
            "TPC Pion Signal Delta",
        ]
    )
}
fhnMixedEvent_axis_labels = {
    key: val
    for key, val in enumerate(
        [
            "V0 centrality (%)",
            "Jet p_{T}",
            "Track p_{T}",
            "#Delta#eta",
            "#Delta#phi",
            "Leading Jet",
            "Event plane angle",
            "Z vertex (cm)",
            "deltaR",
        ]
    )
}
fhnTrigger_axis_labels = {
    key: val
    for key, val in enumerate(["V0 centrality (%)", "Jet p_{T}", "Event plane angle"])
}
import pickle
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt


def get_bin_contents_as_array(th1, forFitting=True):
    bin_contents = []
    for i in range(1, th1.GetNbinsX() + 1):
        if (
            th1.GetBinCenter(i) < -np.pi / 2 or th1.GetBinCenter(i) > np.pi / 2
        ) and forFitting:
            continue
        bin_contents.append(th1.GetBinContent(i))
    return bin_contents


def get_bin_centers_as_array(th1, forFitting=True):
    bin_centers = []
    for i in range(1, th1.GetNbinsX() + 1):
        if (
            th1.GetBinCenter(i) < -np.pi / 2 or th1.GetBinCenter(i) > np.pi / 2
        ) and forFitting:
            continue
        bin_centers.append(th1.GetBinCenter(i))
    return bin_centers


def get_bin_errors_as_array(th1, forFitting=True):
    bin_errors = []
    for i in range(1, th1.GetNbinsX() + 1):

        if (
            th1.GetBinCenter(i) < -np.pi / 2 or th1.GetBinCenter(i) > np.pi / 2
        ) and forFitting:
            continue
        bin_errors.append(th1.GetBinError(i))
    return bin_errors


train_output_r_filenames = [
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_296690_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_296794_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_296894_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_296941_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297031_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297085_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297118_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297129_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297372_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297415_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297441_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297446_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297479_merged.root",
    "/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_297544_merged.root",
]


if __name__ == "__main__":
    # turn off ROOT's automatic garbage collection
    ROOT.TH1.AddDirectory(False)
    flist = [
#         "/mnt/d/18r/297118.root",
# "/mnt/d/18r/297129.root",
# "/mnt/d/18r/297415.root",
# "/mnt/d/18r/297441.root",
# "/mnt/d/18r/297446.root",
# "/mnt/d/18r/297479.root",
# "/mnt/d/18r/297544.root",
# "/mnt/d/18r/297372.root",
# "/mnt/d/18r/297085.root",
# "/mnt/d/18r/296690.root",
# "/mnt/d/18r/296794.root",
# "/mnt/d/18r/296894.root",
# "/mnt/d/18r/296941.root",
# "/mnt/d/18r/297031.root",
 # "/mnt/d/18q/295754.root",
 # "/mnt/d/18q/296065.root",
 # "/mnt/d/18q/296068.root",
 # "/mnt/d/18q/296133.root",
 # "/mnt/d/18q/296191.root",
 # "/mnt/d/18q/296377.root",
 # "/mnt/d/18q/296379.root",
 # "/mnt/d/18q/296423.root",
 # "/mnt/d/18q/296433.root",
 # "/mnt/d/18q/296472.root",
 "/mnt/d/18q/296510.root",
 "/mnt/d/18q/296550.root",
 "/mnt/d/18q/296551.root",
"/mnt/d/18q/295673.root",]#[
        #f"/home/steffanic/Projects/Thesis/TrainOutputq/AnalysisResults_alihaddcomp0{i}.root"
        #for i in range(1, 8)
    #] # + train_output_r_filenames
    fpplist = ["/mnt/d/pp/17p.root"]
    LOAD_PARTIAL_PICKLES = True
    DO_PP = True
    DO_CENTRAL = False
    DO_SEMI_CENTRAL = True
    DO_PLOTTING = True
    DO_FITTING = True
    DO_PICKLING = False

    if DO_PP:
        if not LOAD_PARTIAL_PICKLES:
            if exists("jhAnapp.pickle"):
                jhAnapp = pickle.load(open("jhAnapp.pickle", "rb"))
            else:
                jhAnapp = JetHadron(fpplist, "pp")
                # save a pickle file of the analysis object
                if DO_PICKLING:
                    with open("jhAnapp.pickle", "wb") as handle:
                        pickle.dump(jhAnapp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if exists("jhAnappPartial.pickle"):
                jhAnappPartial = pickle.load(open("jhAnappPartial.pickle", "rb"))
                jhAnappPartial.pick_up_where_you_left_off()
                jhAnapp = jhAnappPartial
            else:
                jhAnapp = JetHadron(fpplist, "pp")
                # save a pickle file of the analysis object
                if DO_PICKLING:
                    with open("jhAnapp.pickle", "wb") as handle:
                        pickle.dump(jhAnapp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if DO_PLOTTING:
            jhAnapp.plot_everything()

    
    if DO_SEMI_CENTRAL:
        if not LOAD_PARTIAL_PICKLES:
            if exists("jhAnaSemiCentral.pickle"):
                jhAnaSemiCentral = pickle.load(open("jhAnaSemiCentral.pickle", "rb"))
            else:
                jhAnaSemiCentral = JetHadron(flist, "semicentral")
                # save a pickle file of the analysis object
                if DO_PICKLING:
                    with open("jhAnaSemiCentral.pickle", "wb") as handle:
                        pickle.dump(
                            jhAnaSemiCentral, handle, protocol=pickle.HIGHEST_PROTOCOL
                        )

        else:
            if exists("jhAnaSemiCentralPartial.pickle"):
                jhAnaSemiCentralPartial = pickle.load(
                    open("jhAnaSemiCentralPartial.pickle", "rb")
                )
                jhAnaSemiCentralPartial.pick_up_where_you_left_off()
                jhAnaSemiCentral = jhAnaSemiCentralPartial
            else:
                jhAnaSemiCentral = JetHadron(flist, "semicentral")
                # save a pickle file of the analysis object
                if DO_PICKLING:
                    with open("jhAnaSemiCentral.pickle", "wb") as handle:
                        pickle.dump(
                            jhAnaSemiCentral, handle, protocol=pickle.HIGHEST_PROTOCOL
                        )

        if DO_FITTING:
            for i in range(len(jhAnaSemiCentral.pTtrigBinEdges) - 1):
                for j in range(len(jhAnaSemiCentral.pTassocBinEdges) - 1):
                    for k in range(4):
                        jhAnaSemiCentral.fit_PionTPCNSigma(i,j,k)
            jhAnaSemiCentral.fit_RPFs()

        if DO_PLOTTING:
            jhAnaSemiCentral.plot_everything()

        if DO_PP:
            for i in range(len(jhAnaSemiCentral.pTtrigBinEdges) - 1):
                for j in range(len(jhAnaSemiCentral.pTassocBinEdges) - 1):
                    jhAnaSemiCentral.plot_dPhi_against_pp_reference(jhAnapp, i, j)
        
    if DO_CENTRAL:
        if not LOAD_PARTIAL_PICKLES:
            if exists("jhAnaCentral.pickle"):
                jhAnaCentral = pickle.load(open("jhAnaCentral.pickle", "rb"))
            else:
                jhAnaCentral = JetHadron(flist, "central")
                # save a pickle file of the analysis object
                if DO_PICKLING:
                    with open("jhAnaCentral.pickle", "wb") as handle:
                        pickle.dump(jhAnaCentral, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            if exists("jhAnaCentralPartial.pickle"):
                jhAnaCentralPartial = pickle.load(
                    open("jhAnaCentralPartial.pickle", "rb")
                )
                jhAnaCentralPartial.pick_up_where_you_left_off()
                jhAnaCentral = jhAnaCentralPartial
            else:
                jhAnaCentral = JetHadron(flist, "central")
                # save a pickle file of the analysis object
                if DO_PICKLING:
                    with open("jhAnaCentral.pickle", "wb") as handle:
                        pickle.dump(jhAnaCentral, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if DO_FITTING:
            jhAnaCentral.fit_RPFs()

        
        if DO_PLOTTING:
            jhAnaCentral.plot_everything()

        if DO_PP:
            for i in range(len(jhAnaCentral.pTtrigBinEdges) - 1):
                for j in range(len(jhAnaCentral.pTassocBinEdges) - 1):
                    jhAnaCentral.plot_dPhi_against_pp_reference(jhAnapp, i, j)

