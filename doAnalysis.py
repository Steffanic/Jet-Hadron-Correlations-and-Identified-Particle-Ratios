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

 
fhnJH_axis_labels = {key:val for key, val in enumerate(['V0 centrality (%)', 'Jet p_{T}', 'Track p_{T}', '#Delta#eta', '#Delta#phi', 'Leading Jet', 'Event plane angle','TPC Pion Signal Delta'])}
fhnMixedEvent_axis_labels = {key:val for key, val in enumerate(['V0 centrality (%)', 'Jet p_{T}', 'Track p_{T}', '#Delta#eta', '#Delta#phi', 'Leading Jet', 'Event plane angle', 'Z vertex (cm)', 'deltaR'])}
fhnTrigger_axis_labels = {key:val for key, val in enumerate(['V0 centrality (%)', 'Jet p_{T}', 'Event plane angle'])}
import pickle
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt

def get_bin_contents_as_array(th1, forFitting=True):
    bin_contents = []
    for i in range(1, th1.GetNbinsX()+1):
        if (th1.GetBinCenter(i) < -np.pi/2 or th1.GetBinCenter(i) > np.pi/2) and forFitting:
            continue
        bin_contents.append(th1.GetBinContent(i))
    return bin_contents

def get_bin_centers_as_array(th1, forFitting=True):
    bin_centers = []
    for i in range(1, th1.GetNbinsX()+1):
        if (th1.GetBinCenter(i) < -np.pi/2 or th1.GetBinCenter(i) > np.pi/2) and forFitting:
            continue
        bin_centers.append(th1.GetBinCenter(i))
    return bin_centers

def get_bin_errors_as_array(th1, forFitting=True):
    bin_errors = []
    for i in range(1, th1.GetNbinsX()+1):
        
        if (th1.GetBinCenter(i) < -np.pi/2 or th1.GetBinCenter(i) > np.pi/2) and forFitting:
            continue
        bin_errors.append(th1.GetBinError(i))
    return bin_errors

if __name__=="__main__":
    f = ROOT.TFile('/home/steffanic/Projects/Thesis/TrainOutputr/AnalysisResults_merged_ab.root')
    fpp = ROOT.TFile('/home/steffanic/Projects/Thesis/TrainOutputpp/AnalysisResults.root')
    DO_PP = True
    DO_CENTRAL = True
    DO_SEMI_CENTRAL = True
    DO_PLOTTING=False


    if DO_PP:
        if exists('jhAnapp.pickle'):
            jhAnapp = pickle.load(open('jhAnapp.pickle', 'rb'))
        else:
            jhAnapp = JetHadron(fpp, 'pp')
            #save a pickle file of the analysis object
            with open('jhAnapp.pickle', 'wb') as handle:
                pickle.dump(jhAnapp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        if DO_PLOTTING:
            jhAnapp.plot_everything()

    if DO_CENTRAL:
        if exists('jhAnaCentral.pickle'):
            jhAnaCentral = pickle.load(open('jhAnaCentral.pickle', 'rb'))
        else:
            jhAnaCentral = JetHadron(f, 'central')
            #save a pickle file of the analysis object
            with open('jhAnaCentral.pickle', 'wb') as handle:
                pickle.dump(jhAnaCentral, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if not jhAnaCentral.RPFObjs.all():
            jhAnaCentral.fit_RPFs()
        
        
        '''
        for i in range(len(jhAnaCentral.pTtrigBinEdges)-1):
            for j in range(len(jhAnaCentral.pTassocBinEdges)-1):
                print(f"Fitting {i}, {j}")
                ydata = np.array(jhAnaCentral.get_bin_contents_as_array(jhAnaCentral.dPionASsignals[i,j], forFitting=False))
                yerr = np.array(jhAnaCentral.get_bin_errors_as_array(jhAnaCentral.dPionASsignals[i,j], forFitting=False))
                xdata = np.array(jhAnaCentral.get_bin_centers_as_array(jhAnaCentral.dPionASsignals[i,j], forFitting=False))
                negative_supression_mask = (np.array(ydata)>0)
                ydata = ydata[negative_supression_mask]
                yerr = yerr[negative_supression_mask]
                xdata = xdata[negative_supression_mask]
                tF = templateFit(jhAnaCentral.pTassocBinEdges[j])
                tF.fit_sum_of_gaussians(xdata, ydata, yerr, title="$\\Delta_{\\pi}$" + f" for {jhAnaCentral.pTtrigBinEdges[i]}-{jhAnaCentral.pTtrigBinEdges[i+1]}, {jhAnaCentral.pTassocBinEdges[j]}-{jhAnaCentral.pTassocBinEdges[j+1]}")
        '''
        if DO_PLOTTING:
            jhAnaCentral.plot_everything()
        for i in range(len(jhAnaCentral.pTtrigBinEdges)-1):
            for j in range(len(jhAnaCentral.pTassocBinEdges)-1):
                for k in range(4):
                    jhAnaCentral.plot_dPhi_against_pp_reference(jhAnapp, i,j)

        '''
        pdf = FPDF("L", "in", "Letter")
        pdf.set_margins(0, 0, 0)
        base_path = '/home/steffanic/Projects/Thesis/backend_output/'
        for image in [base_path + 'central/' + img for img in sorted(os.listdir(base_path + 'central/')) if img.endswith('.png') and img.startswith("RPF")]:
            pdf.add_page()
            pdf.image(image, x=0, y=0, w=11, h=3.66)
        pdf.output(base_path + 'central.pdf')
        '''
        
    
    if DO_SEMI_CENTRAL:
        if exists('jhAnaSemiCentral.pickle'):
            jhAnaSemiCentral = pickle.load(open('jhAnaSemiCentral.pickle', 'rb'))
        else:
            jhAnaSemiCentral = JetHadron(f, 'semicentral')
            #save a pickle file of the analysis object
            with open('jhAnaSemiCentral.pickle', 'wb') as handle:
                pickle.dump(jhAnaSemiCentral, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if not jhAnaSemiCentral.RPFObjs.all():
            jhAnaSemiCentral.fit_RPFs()

        if DO_PLOTTING:
            jhAnaSemiCentral.plot_everything()
        for i in range(len(jhAnaSemiCentral.pTtrigBinEdges)-1):
            for j in range(len(jhAnaSemiCentral.pTassocBinEdges)-1):
                for k in range(4):
                    jhAnaSemiCentral.plot_dPhi_against_pp_reference(jhAnapp, i,j)
        '''
        pdf = FPDF("L", "in", "Letter")
        pdf.set_margins(0, 0, 0)
        base_path = '/home/steffanic/Projects/Thesis/backend_output/'
        for image in [base_path + 'semicentral/' + img for img in sorted(os.listdir(base_path + 'semicentral/')) if img.endswith('.png') and img.startswith("RPF")]:
            pdf.add_page()
            pdf.image(image, x=0, y=0, w=11, h=3.66)
        pdf.output(base_path + 'semicentral.pdf')
        '''

        

