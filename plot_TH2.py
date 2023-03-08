import numpy as np 
import matplotlib.pyplot as plt
import ROOT
import matplotlib
import numpy as np
import pandas as pd


def TH2toArray(hist):
    xbins = hist.GetXaxis().GetNbins()
    ybins = hist.GetYaxis().GetNbins()
    xedges = np.zeros(xbins)
    yedges = np.zeros(ybins)
    for i in range(xbins):
        xedges[i] = hist.GetXaxis().GetBinLowEdge(i)
    for i in range(ybins):
        yedges[i] = hist.GetYaxis().GetBinLowEdge(i)

    z = np.zeros((xbins,ybins))
    for i in range(xbins):
        for j in range(ybins):
            z[i,j] = hist.GetBinContent(i,j)
    return xedges, yedges, z

def plot_TH2(hist, title, xlabel, ylabel, zlabel, cmap='viridis'):
    xedges, yedges, z = TH2toArray(hist)
    X,Y = np.meshgrid(xedges, yedges)
    plt.figure(figsize=(6,6))
    plt.pcolormesh(X, Y, z.T, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=zlabel)
    plt.savefig(f'{title}.png')

def get_ThNSparse_axis_titles(sparse):
    '''
    Returns list of all axis titles of a THnSparse
    '''
    axis_titles = []
    for i in range(sparse.GetNdimensions()):
        axis_titles.append(sparse.GetAxis(i).GetTitle())
    return axis_titles

if __name__ == '__main__':
    f = ROOT.TFile('/home/steffanic/Projects/Thesis/LEGOTrainTest/AnalysisResults.root')
    print(f.ls())
    central = f.Get('AliAnalysisTaskJetH_tracks_caloClusters_dEdxtrackBias5R2Centralq')
    print(central.ls())
    fhnJH = central.FindObject("fhnJH")
    fhnMixedEvents = central.FindObject("fhnMixedEvents")
    fhnTrigger = central.FindObject("fhnTrigger")
    axis_titlesJH = get_ThNSparse_axis_titles(fhnJH)
    axis_titlesMixedEvents = get_ThNSparse_axis_titles(fhnMixedEvents)
    axis_titlesTrigger = get_ThNSparse_axis_titles(fhnTrigger)
    print(axis_titlesJH)
    print(axis_titlesMixedEvents)
    print(axis_titlesTrigger)
    # Get just leading jets 
    fhnJH.GetAxis(2).SetRangeUser(3,25)
    #plot Track p_T vs everything else
    for i in range(3, fhnJH.GetNdimensions()):
        hist = fhnJH.Projection(i,2)
        plot_TH2(hist, f'{axis_titlesJH[2]} vs. {axis_titlesJH[i]}', 'track p_T', axis_titlesJH[i], 'counts')
   