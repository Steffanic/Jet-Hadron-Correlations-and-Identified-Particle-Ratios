import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


import matplotlib.cm as cm
from matplotlib.pyplot import Axes

def TH2toArray(hist):
    xbins = hist.GetXaxis().GetNbins()
    ybins = hist.GetYaxis().GetNbins()
    x_centers = np.zeros(xbins)
    y_centers = np.zeros(ybins)
    for i in range(1,xbins+1):
        x_centers[i-1] = hist.GetXaxis().GetBinCenter(i)
    for i in range(1,ybins+1):
        y_centers[i-1] = hist.GetYaxis().GetBinCenter(i)

    z = np.zeros((xbins,ybins))
    zerr = np.zeros((xbins,ybins))
    for i in range(1,xbins+1):
        for j in range(1,ybins+1):
            z[i-1,j-1] = hist.GetBinContent(i,j)
            zerr[i-1,j-1] = hist.GetBinError(i,j)
    return x_centers, y_centers, z, zerr

def plot_TH2(hist, title, xlabel, ylabel, zlabel, cmap='viridis', plotStyle="colz"):
    xedges, yedges, z, _ = TH2toArray(hist)
    X,Y = np.meshgrid(xedges, yedges)
    fig = plt.figure(figsize=(10,6))

    # compute difference between element i and i-1 in xedges 
    # this is the width of the bin
    
    dx = np.diff(xedges)
    dy = np.diff(yedges)

    if plotStyle=="colz":
        plt.pcolormesh(X, Y, z.T, cmap=cmap)
        plt.colorbar(label=zlabel)
    elif plotStyle=="bar3d":
        ax: Axes3D = fig.add_subplot(111, projection='3d')
        x,y,Z = X.ravel(), Y.ravel(), z.T.ravel()
        cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
        max_height = np.max(Z)   # get range of colorbars so we can normalize
        min_height = np.min(Z)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/max_height) for k in Z] 

        ax.bar3d(x, y, np.zeros_like(Z), dx, dy, Z, color=rgba, zsort='average')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt.gcf()

def plot_TH1(hist, title, xlabel, ylabel, ax: Axes, logy=False ):
    xbins = hist.GetXaxis().GetNbins()
    x_centers = np.zeros(xbins)
    for i in range(1,xbins+1):
        x_centers[i-1] = hist.GetXaxis().GetBinCenter(i)
    z = np.zeros(xbins)
    zerr = np.zeros(xbins)
    for i in range(1,xbins+1):
        z[i-1] = hist.GetBinContent(i)
        zerr[i-1] = hist.GetBinError(i)
    
    ax.errorbar(x_centers, z, yerr=zerr, fmt='k+')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    return ax


class PlotMixin:
    def plot_everything(self):
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

        matplotlib.rc('font', **font)
        if self.analysisType in ["central", "semicentral"]: #type:ignore
            numEPBins=4 # in-, mid-, out-, inclusive
        elif self.analysisType in ["pp"]: #type:ignore
            numEPBins=1 #inclusive only
        else:
            numEPBins=1 #inclusive only
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            for j in range(len(self.pTassocBinEdges)-1): #type:ignore
                print(f"pTtrig: {self.pTtrigBinEdges[i]} - {self.pTtrigBinEdges[i+1]}, pTassoc: {self.pTassocBinEdges[j]} - {self.pTassocBinEdges[j+1]}") #type:ignore
                figBG, axBG = plt.subplots(1,numEPBins, figsize=(20,5), sharey=True)
                figSIG, axSig = plt.subplots(1,numEPBins, figsize=(20,5), sharey=True)
                figSigminusBGNS, axSigminusBGNS = plt.subplots(1,numEPBins, figsize=(20,5), sharey=True)
                figSigminusBGAS, axSigminusBGAS = plt.subplots(1,numEPBins, figsize=(20,5), sharey=True)
                figSigminusBGINC, axSigminusBGINC = plt.subplots(1,numEPBins, figsize=(20,5), sharey=True)
                figSigminusBGNormINC, axSigminusBGNormINC = plt.subplots(1,numEPBins, figsize=(20,5), sharey=True)
                figBG.suptitle(f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c") #type:ignore
                figSIG.suptitle(f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c") #type:ignore
                figSigminusBGNS.suptitle(f"Background subtracted $\\Delta \\phi$ in near-side signal region {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c") #type:ignore
                figSigminusBGAS.suptitle(f"Background-subtracted $\\Delta \\phi$ in away-side signal region {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c") #type:ignore
                figSigminusBGINC.suptitle(f"Background-subtracted $\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c") #type:ignore
                figSigminusBGNormINC.suptitle(f"Normalized, background-subtracted $\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c") #type:ignore
                # remove margins 
                figBG.subplots_adjust(wspace=0, hspace=0)
                figSIG.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNS.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGAS.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGINC.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINC.subplots_adjust(wspace=0, hspace=0)
                
                for k in range(len(self.eventPlaneAngleBinEdges)): #type:ignore
                    if self.analysisType =="pp": #type:ignore
                        if k<3:
                            continue
                    epString = "out-of-plane" if k==2 else ("mid-plane" if k==1 else ("in-plane" if k==0 else "inclusive"))

                    if self.analysisType in ["central", "semicentral"]: #type:ignore
                        correlation_func = plot_TH2(self.SEcorrs[i,j,k], f"Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$\\frac{1}{\epsilon}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        correlation_func.savefig(f"{self.base_save_path}{epString}/CorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(correlation_func)

                        normMEcorrelation_function = plot_TH2(self.NormMEcorrs[i,j,k], f"Normalized Mixed Event Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$a_0\\frac{d^2N_{mixed-event}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        normMEcorrelation_function.savefig(f"{self.base_save_path}{epString}/NormalizedMixedEventCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(normMEcorrelation_function)

                        accCorrectedCorrelationFunction = plot_TH2(self.AccCorrectedSEcorrs[i,j,k], f"Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        accCorrectedCorrelationFunction.savefig(f"{self.base_save_path}{epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(accCorrectedCorrelationFunction)

                        normAccCorrectedCorrelationFunction = plot_TH2(self.NormAccCorrectedSEcorrs[i,j,k], f"Normalized Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$\\frac{1}{N_{trig} \\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        normAccCorrectedCorrelationFunction.savefig(f"{self.base_save_path}{epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(normAccCorrectedCorrelationFunction)

                        dPhiBG, dPhiBGax = plt.subplots(1,1, figsize=(10,6))
                        dPhiBGax = plot_TH1(self.dPhiBGcorrs[i,j,k], f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$", ax=dPhiBGax) #type:ignore
                        x_binsBG = np.array([self.dPhiBGcorrs[i,j,k].GetXaxis().GetBinCenter(b) for b in range(1, self.dPhiBGcorrs[i,j,k].GetNbinsX()+1)])
                        bin_contentBG = np.array([self.dPhiBGcorrs[i,j,k].GetBinContent(b) for b in range(1, self.dPhiBGcorrs[i,j,k].GetNbinsX()+1)])
                        # do fill between for me norm systematic 
                        dPhiBGax.fill_between(x_binsBG, bin_contentBG-self.ME_norm_systematics[i,j, k], bin_contentBG+self.ME_norm_systematics[i,j, k], color='red', alpha=0.5, label="ME normalization")
                        dPhiBG.legend()

                        dPhiBG.savefig(f"{self.base_save_path}{epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiBG)
                        axBG[k] = plot_TH1(self.dPhiBGcorrs[i,j,k], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$" if k==0 else "", ax=axBG[k]) #type:ignore
                        axBG[k].fill_between(x_binsBG, bin_contentBG-self.ME_norm_systematics[i,j, k], bin_contentBG+self.ME_norm_systematics[i,j, k], color='red', alpha=0.5, label="ME normalization")

                        dPhiSig, dPhiSigax = plt.subplots(1,1, figsize=(10,6))
                        dPhiSigax= plot_TH1(self.dPhiSigcorrs[i,j,k], f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$", ax=dPhiSigax) #type:ignore
                        x_binsSig = np.array([self.dPhiSigcorrs[i,j,k].GetXaxis().GetBinCenter(b) for b in range(1, self.dPhiSigcorrs[i,j,k].GetNbinsX()+1)])
                        bin_contentSig = np.array([self.dPhiSigcorrs[i,j,k].GetBinContent(b) for b in range(1, self.dPhiSigcorrs[i,j,k].GetNbinsX()+1)])
                        # do fill between for me norm systematic
                        dPhiSigax.fill_between(x_binsSig, bin_contentSig-self.ME_norm_systematics[i,j, k], bin_contentSig+self.ME_norm_systematics[i,j, k], color='red', alpha=0.5, label="ME normalization")
                        dPhiSig.legend()
                        dPhiSig.savefig(f"{self.base_save_path}{epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSig)
                        axSig[k] = plot_TH1(self.dPhiSigcorrs[i,j,k], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$" if k==0 else "", ax=axSig[k]) #type:ignore
                        axSig[k].fill_between(x_binsSig, bin_contentSig-self.ME_norm_systematics[i,j,k], bin_contentSig+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")

                        

                        dPhiSigminusBGNS, dPhiSigminusBGNSax = plt.subplots(1,1, figsize=(10,6))
                        dPhiSigminusBGNSax = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k], f"Background subtracted $\\Delta \\phi$, near side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$", ax=dPhiSigminusBGNSax) #type:ignore
                        # add fit uncertainties as fill_between
                        n_binsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k].GetXaxis().GetNbins() #type:ignore
                        x_binsNS = np.zeros(n_binsNS)
                        for l in range(n_binsNS):
                            x_binsNS[l] = self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k].GetXaxis().GetBinCenter(l+1)
                        bin_contentNS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k].GetBinContent(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k].GetNbinsX()+1)]) #type:ignore
                        bin_errorsNS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k].GetBinError(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k].GetNbinsX()+1)]) #type:ignore
                        RPFErrorNS = np.array([self.RPFObjs[i,j].simultaneous_fit_err(x_binsNS[l],x_binsNS[1]-x_binsNS[0], *self.RPFObjs[i,j].popt) for l in range(len(x_binsNS))]) #type:ignore
                        
                        dPhiSigminusBGNSax.fill_between(x_binsNS, bin_contentNS - RPFErrorNS[:,k] if k!=3 else bin_contentNS - np.sqrt(np.sum(RPFErrorNS**2, axis=1)), bin_contentNS+RPFErrorNS[:,k] if k!=3 else bin_contentNS+np.sqrt(np.sum(RPFErrorNS**2, axis=1)), alpha=0.5, color = 'green', label="RPF fit")
                        # do fill between for me norm systematic
                        dPhiSigminusBGNSax.fill_between(x_binsNS, bin_contentNS-self.ME_norm_systematics[i,j,k], bin_contentNS+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")
                        dPhiSigminusBGNS.legend()
                        dPhiSigminusBGNS.savefig(f"{self.base_save_path}{epString}/dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGNS)
                        axSigminusBGNS[k] = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j,k], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "", ax=axSigminusBGNS[k]) #type:ignore
                        axSigminusBGNS[k].fill_between(x_binsNS, bin_contentNS - RPFErrorNS[:,k] if k!=3 else bin_contentNS - np.sqrt(np.sum(RPFErrorNS**2, axis=1)), bin_contentNS+RPFErrorNS[:,k] if k!=3 else bin_contentNS+np.sqrt(np.sum(RPFErrorNS**2, axis=1)), alpha=0.5, color = 'green', label="RPF fit")
                        axSigminusBGNS[k].fill_between(x_binsNS, bin_contentNS-self.ME_norm_systematics[i,j, k], bin_contentNS+self.ME_norm_systematics[i,j, k], color='red', alpha=0.5, label="ME normalization")
                        
                        dPhiSigminusBGAS, dPhiSigminusBGASax = plt.subplots(1,1, figsize=(10,6))
                        dPhiSigminusBGASax = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k], f"Background subtracted $\\Delta \\phi$, away side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$", ax=dPhiSigminusBGASax) #type:ignore
                        # add fit uncertainties as fill_between
                        n_binsAS = self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k].GetXaxis().GetNbins()#type:ignore
                        x_binsAS = np.zeros(n_binsAS)
                        for l in range(n_binsAS):
                            x_binsAS[l] = self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k].GetXaxis().GetBinCenter(l+1)
                        bin_contentAS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k].GetBinContent(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k].GetNbinsX()+1)]) #type:ignore
                        bin_errorsAS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k].GetBinError(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k].GetNbinsX()+1)]) #type:ignore
                        RPFErrorAS = np.array([self.RPFObjs[i,j].simultaneous_fit_err(x_binsAS[l],x_binsAS[1]-x_binsAS[0], *self.RPFObjs[i,j].popt) for l in range(len(x_binsAS))]) #type:ignore
                        dPhiSigminusBGASax.fill_between(x_binsAS, bin_contentAS - RPFErrorAS[:,k] if k!=3 else bin_contentAS-np.sqrt(np.sum(RPFErrorAS**2, axis=1)), bin_contentAS+RPFErrorAS[:,k]if k!=3 else bin_contentAS+np.sqrt(np.sum(RPFErrorAS**2, axis=1)), alpha=0.5, color="green", label="RPF fit")
                        # do fill between for me norm systematic
                        dPhiSigminusBGASax.fill_between(x_binsAS, bin_contentAS-self.ME_norm_systematics[i,j,k], bin_contentAS+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")
                        dPhiSigminusBGAS.legend()
                        dPhiSigminusBGAS.savefig(f"{self.base_save_path}{epString}/dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGAS)
                        axSigminusBGAS[k] = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j,k], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "", ax=axSigminusBGAS[k]) #type:ignore
                        axSigminusBGAS[k].fill_between(x_binsAS, bin_contentAS - RPFErrorAS[:,k] if k!=3 else bin_contentAS-np.sqrt(np.sum(RPFErrorAS**2, axis=1)), bin_contentAS+RPFErrorAS[:,k]if k!=3 else bin_contentAS+np.sqrt(np.sum(RPFErrorAS**2, axis=1)), alpha=0.5, color="green", label="RPF fit")
                        # do fill between for me norm systematic
                        axSigminusBGAS[k].fill_between(x_binsAS, bin_contentAS-self.ME_norm_systematics[i,j,k], bin_contentAS+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")
                        

                        dPhiSigminusBGINC, dPhiSigminusBGINCax = plt.subplots(1,1, figsize=(10,6))
                        #type:ignore
                        # add fit uncertainties as fill_between
                        n_binsINC = n_binsNS+n_binsAS
                        x_binsINC = np.zeros(n_binsINC)
                        for l in range(n_binsINC):
                            x_binsINC[l] = x_binsNS[l] if l<n_binsNS else x_binsAS[l-n_binsNS]
                        bin_contentINC = np.array([bin_contentNS[l] if l<n_binsNS else bin_contentAS[l-n_binsNS] for l in range(n_binsINC)])
                        bin_errorsINC = np.array([bin_errorsNS[l] if l<n_binsNS else bin_errorsAS[l-n_binsNS] for l in range(n_binsINC)])
                        RPFErrorINC = np.array([RPFErrorNS[l] if l<n_binsNS else RPFErrorAS[l-n_binsNS] for l in range(n_binsINC)]) #type:ignore
                        dPhiSigminusBGINCax.errorbar(x_binsINC, bin_contentINC, yerr=bin_errorsINC,  fmt="o", color="black")
                        dPhiSigminusBGINCax.fill_between(x_binsINC, bin_contentINC - RPFErrorINC[:,k] if k!=3 else bin_contentINC-np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9), bin_contentINC+RPFErrorINC[:,k] if k!=3 else bin_contentINC+np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9), alpha=0.5, color="green", label="RPF fit")
                        # do fill between for me norm systematic
                        dPhiSigminusBGINCax.fill_between(x_binsINC, bin_contentINC-self.ME_norm_systematics[i,j,k], bin_contentINC+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")
                        dPhiSigminusBGINC.legend()
                        dPhiSigminusBGINCax.set_xlabel("$\\Delta \\phi$")
                        dPhiSigminusBGINCax.set_ylabel("$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")
                        dPhiSigminusBGINCax.set_title(f"Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}")
                        dPhiSigminusBGINC.savefig(f"{self.base_save_path}{epString}/dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGINC)
                        axSigminusBGINC[k].errorbar(x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black")
                        axSigminusBGINC[k].fill_between(x_binsINC, bin_contentINC - RPFErrorINC[:,k] if k!=3 else bin_contentINC-np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9), bin_contentINC+RPFErrorINC[:,k]if k!=3 else bin_contentINC+np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9), alpha=0.5, color="green", label="RPF fit")
                        # do fill between for me norm systematic
                        axSigminusBGINC[k].fill_between(x_binsINC, bin_contentINC-self.ME_norm_systematics[i,j,k], bin_contentINC+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")

                        axSigminusBGINC[k].set_xlabel("$\\Delta \\phi$")
                        axSigminusBGINC[k].set_ylabel("$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")

                        dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(1,1, figsize=(10,6)) #type:ignore
                        n_binsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetXaxis().GetNbins() #type:ignore
                        x_binsNormINC = np.zeros(n_binsNormINC) #type:ignore
                        bin_contentNormINC = np.zeros(n_binsNormINC) #type:ignore
                        bin_errorsNormINC = np.zeros(n_binsNormINC) #type:ignore
                        self.set_pT_epAngle_bin(i,j,k)

                        N_trig = self.get_N_trig()
                        for l in range(n_binsNormINC):
                            x_binsNormINC[l] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetXaxis().GetBinCenter(l+1) #type:ignore
                            bin_contentNormINC[l] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetBinContent(l+1) #type:ignore
                            bin_errorsNormINC[l] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetBinError(l+1) #type:ignore
                        if k==3:
                            # divide bin content by 3
                            bin_contentNormINC = bin_contentNormINC/3
                            bin_errorsNormINC = bin_errorsNormINC/3
                            
                        dPhiSigminusBGNormINCax.errorbar(x_binsNormINC, bin_contentNormINC, yerr=bin_errorsNormINC, fmt="o", color="black")
                        dPhiSigminusBGNormINCax.fill_between(x_binsNormINC, bin_contentNormINC - RPFErrorINC[:,k]/N_trig if k!=3 else bin_contentNormINC-np.sqrt(np.sum((RPFErrorINC)**2, axis=1)/9)/N_trig, bin_contentNormINC+RPFErrorINC[:,k]/N_trig if k!=3 else bin_contentNormINC+np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9)/N_trig, alpha=0.5, color="green", label="RPF fit")
                        # do fill between for me norm systematic
                        dPhiSigminusBGNormINCax.fill_between(x_binsNormINC, bin_contentNormINC-self.ME_norm_systematics[i,j,k], bin_contentNormINC+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")
                        dPhiSigminusBGNormINC.legend()
                        dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$")
                        dPhiSigminusBGNormINCax.set_ylabel("$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")
                        dPhiSigminusBGNormINCax.set_title(f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}") #type:ignore
                        dPhiSigminusBGNormINC.savefig(f"{self.base_save_path}{epString}/dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGNormINC)
                        axSigminusBGNormINC[k].errorbar(x_binsNormINC, bin_contentNormINC, yerr=bin_errorsNormINC, fmt="o", color="black")
                        axSigminusBGNormINC[k].fill_between(x_binsNormINC, bin_contentNormINC - RPFErrorINC[:,k]/N_trig if k!=3 else bin_contentNormINC-np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9)/N_trig, bin_contentNormINC+RPFErrorINC[:,k]/N_trig if k!=3 else bin_contentNormINC+np.sqrt(np.sum(RPFErrorINC**2, axis=1)/9)/N_trig, alpha=0.5, color="green", label="RPF fit")
                        # do fill between for me norm systematic
                        axSigminusBGNormINC[k].fill_between(x_binsNormINC, bin_contentNormINC-self.ME_norm_systematics[i,j,k], bin_contentNormINC+self.ME_norm_systematics[i,j,k], color='red', alpha=0.5, label="ME normalization")
                        
                        axSigminusBGNormINC[k].set_xlabel("$\\Delta \\phi$")
                        axSigminusBGNormINC[k].set_ylabel("$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")



                        
                    elif self.analysisType == "pp": #type:ignore
                        correlation_func = plot_TH2(self.SEcorrs[i,j], f"Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$\\frac{1}{\epsilon}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        correlation_func.savefig(f"{self.base_save_path}{epString}/CorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(correlation_func)

                        normMEcorrelation_function = plot_TH2(self.NormMEcorrs[i,j], f"Normalized Mixed Event Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$a_0\\frac{d^2N_{mixed-event}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        normMEcorrelation_function.savefig(f"{self.base_save_path}{epString}/NormalizedMixedEventCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(normMEcorrelation_function)

                        accCorrectedCorrelationFunction = plot_TH2(self.AccCorrectedSEcorrs[i,j], f"Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                       
                        accCorrectedCorrelationFunction.savefig(f"{self.base_save_path}{epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(accCorrectedCorrelationFunction)

                        normAccCorrectedCorrelationFunction = plot_TH2(self.NormAccCorrectedSEcorrs[i,j], f"Normalized Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "$\\frac{1}{N_{trig} \\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$", cmap='jet') #type:ignore
                        
                        normAccCorrectedCorrelationFunction.savefig(f"{self.base_save_path}{epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(normAccCorrectedCorrelationFunction)

                        dPhiBG, dPhiBGax = plt.subplots(1,1, figsize=(10,6))
                        dPhiBGax = plot_TH1(self.dPhiBGcorrs[i,j], f"$\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$", ax=dPhiBGax) #type:ignore
                        x_binsBG = np.array([self.dPhiBGcorrs[i,j].GetXaxis().GetBinCenter(k) for k in range(self.dPhiBGcorrs[i,j].GetNbinsX())]) #type:ignore
                        bin_contentBG = np.array([self.dPhiBGcorrs[i,j].GetBinContent(k) for k in range(1,self.dPhiBGcorrs[i,j].GetNbinsX()+1)]) #type:ignore
                        # do a fill between with self.ME_norm_systematics
                        dPhiBGax.fill_between(x_binsBG, bin_contentBG-self.ME_norm_systematics[i,j], bin_contentBG+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        dPhiBG.legend()
                        dPhiBG.savefig(f"{self.base_save_path}{epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiBG)
                        axBG = plot_TH1(self.dPhiBGcorrs[i,j],f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$" if k==0 else "", ax=axBG) #type:ignore
                        axBG.fill_between(x_binsBG, bin_contentBG-self.ME_norm_systematics[i,j], bin_contentBG+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore

                        dPhiSig, dPhiSigax = plt.subplots(1,1, figsize=(10,6))
                        dPhiSigax= plot_TH1(self.dPhiSigcorrs[i,j], f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$", ax=dPhiSigax) #type:ignore
                        x_binsSig = np.array([self.dPhiSigcorrs[i,j].GetXaxis().GetBinCenter(k) for k in range(self.dPhiSigcorrs[i,j].GetNbinsX())]) #type:ignore
                        bin_contentSig = np.array([self.dPhiSigcorrs[i,j].GetBinContent(k) for k in range(1,self.dPhiSigcorrs[i,j].GetNbinsX()+1)]) #type:ignore
                        # do a fill between with self.ME_norm_systematics
                        dPhiSigax.fill_between(x_binsSig, bin_contentSig-self.ME_norm_systematics[i,j], bin_contentSig+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        dPhiSig.legend()
                        dPhiSig.savefig(f"{self.base_save_path}{epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSig)
                        axSig = plot_TH1(self.dPhiSigcorrs[i,j], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$" if k==0 else "", ax=axSig) #type:ignore
                        axSig.fill_between(x_binsSig, bin_contentSig-self.ME_norm_systematics[i,j], bin_contentSig+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore

                        dPhiSigminusBGNS, dPhiSigminusBGNSax = plt.subplots(1,1, figsize=(10,6))
                        dPhiSigminusBGNSax = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j], f"Background subtracted $\\Delta \\phi$, near side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$", ax=dPhiSigminusBGNSax) #type:ignore
                        # add fit uncertainties as fill_between
                        n_binsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetNbinsX()
                        x_binsNS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetXaxis().GetBinCenter(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetNbinsX()+1)])
                        bin_contentNS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetBinContent(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetNbinsX()+1)])
                        bin_errorsNS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetBinError(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j].GetNbinsX()+1)])
                        norm_errorsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals[i,j]
                        # set pt ep angle bin 
                        self.set_pT_epAngle_bin(i,j, k)
                        # get the number of triggers 
                        N_trig = self.get_N_trig()
                        dPhiSigminusBGNSax.fill_between(x_binsNS, bin_contentNS-norm_errorsNS/N_trig, bin_contentNS+norm_errorsNS/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        #also fill between with me norm systematics
                        dPhiSigminusBGNSax.fill_between(x_binsNS, bin_contentNS-self.ME_norm_systematics[i,j], bin_contentNS+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        dPhiSigminusBGNS.legend()
                        dPhiSigminusBGNS.savefig(f"{self.base_save_path}{epString}/dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGNS)

                        axSigminusBGNS= plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i,j], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "", ax=axSigminusBGNS) #type:ignore
                        axSigminusBGNS.fill_between(x_binsNS, bin_contentNS-norm_errorsNS/N_trig, bin_contentNS+norm_errorsNS/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        axSigminusBGNS.fill_between(x_binsNS, bin_contentNS-self.ME_norm_systematics[i,j], bin_contentNS+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore

                        
                        
                        dPhiSigminusBGAS, dPhiSigminusBGASax = plt.subplots(1,1, figsize=(10,6))
                        dPhiSigminusBGASax = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j], f"Background subtracted $\\Delta \\phi$, away side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$", ax=dPhiSigminusBGASax) #type:ignore
                        # add fit uncertainties as fill_between
                        n_binsAS = self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j].GetXaxis().GetNbins()#type:ignore
                        x_binsAS = np.zeros(n_binsAS)
                        for l in range(n_binsAS):
                            x_binsAS[l] = self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j].GetXaxis().GetBinCenter(l+1) #type:ignore
                        bin_contentAS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j].GetBinContent(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j].GetNbinsX()+1)]) #type:ignore
                        bin_errorsAS = np.array([self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j].GetBinError(b) for b in range(1, self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j].GetNbinsX()+1)]) #type:ignore
                        norm_errorsAS = self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals[i,j] #type:ignore
                        dPhiSigminusBGASax.fill_between(x_binsAS, bin_contentAS-norm_errorsAS/N_trig, bin_contentAS+norm_errorsAS/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        # also fill between with me norm systematics
                        dPhiSigminusBGASax.fill_between(x_binsAS, bin_contentAS-self.ME_norm_systematics[i,j], bin_contentAS+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        dPhiSigminusBGAS.legend()
                        
                        dPhiSigminusBGAS.savefig(f"{self.base_save_path}{epString}/dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGAS)
                        axSigminusBGAS = plot_TH1(self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i,j], f"{epString}", "$\\Delta \\phi$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "", ax=axSigminusBGAS) #type:ignore
                        axSigminusBGAS.fill_between(x_binsAS, bin_contentAS-norm_errorsAS/N_trig, bin_contentAS+norm_errorsAS/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        axSigminusBGAS.fill_between(x_binsAS, bin_contentAS-self.ME_norm_systematics[i,j], bin_contentAS+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore

                        

                        dPhiSigminusBGINC, dPhiSigminusBGINCax = plt.subplots(1,1, figsize=(10,6))
                        #type:ignore
                        # add fit uncertainties as fill_between
                        n_binsINC = n_binsNS+n_binsAS
                        x_binsINC = np.zeros(n_binsINC)
                        for l in range(n_binsINC):
                            x_binsINC[l] = x_binsNS[l] if l<n_binsNS else x_binsAS[l-n_binsNS]
                        bin_contentINC = np.array([bin_contentNS[l] if l<n_binsNS else bin_contentAS[l-n_binsNS] for l in range(n_binsINC)])
                        bin_errorsINC = np.array([bin_errorsNS[l] if l<n_binsNS else bin_errorsAS[l-n_binsNS] for l in range(n_binsINC)])
                        norm_errorsINC = (norm_errorsNS+norm_errorsAS)/2
                        
                        dPhiSigminusBGINCax.errorbar(x_binsINC, bin_contentINC, yerr=bin_errorsINC,  fmt="o", color="black")
                        dPhiSigminusBGINCax.fill_between(x_binsINC, bin_contentINC-norm_errorsINC/N_trig, bin_contentINC+norm_errorsINC/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        # also fill between with me norm systematics
                        dPhiSigminusBGINCax.fill_between(x_binsINC, bin_contentINC-self.ME_norm_systematics[i,j], bin_contentINC+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        dPhiSigminusBGINC.legend()
                        
                        dPhiSigminusBGINCax.set_xlabel("$\\Delta \\phi$")
                        dPhiSigminusBGINCax.set_ylabel("$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")
                        dPhiSigminusBGINCax.set_title(f"Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}") #type:ignore
                        dPhiSigminusBGINC.savefig(f"{self.base_save_path}{epString}/dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGINC)
                        axSigminusBGINC.errorbar(x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black")
                        axSigminusBGINC.fill_between(x_binsINC, bin_contentINC-norm_errorsINC/N_trig, bin_contentINC+norm_errorsINC/N_trig, alpha=0.5, color="red")
                       
                        axSigminusBGINC.set_xlabel("$\\Delta \\phi$")
                        axSigminusBGINC.set_ylabel("$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")

                        dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(1,1, figsize=(10,6))#type:ignore
                        n_binsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j].GetNbinsX() #type:ignore
                        x_binsNormINC = np.array([self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j].GetBinCenter(b) for b in range(1, n_binsNormINC+1)]) #type:ignore
                        bin_contentNormINC = np.array([self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j].GetBinContent(b) for b in range(1, n_binsNormINC+1)]) #type:ignore
                        bin_errorsNormINC = np.array([self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j].GetBinError(b) for b in range(1, n_binsNormINC+1)]) #type:ignore
                        norm_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[i,j] #type:ignore
                        dPhiSigminusBGNormINCax.errorbar(x_binsNormINC, bin_contentNormINC, yerr=bin_errorsNormINC, fmt="o", color="black")
                        dPhiSigminusBGNormINCax.fill_between(x_binsNormINC, bin_contentNormINC-norm_errorsNormINC/N_trig, bin_contentNormINC+norm_errorsNormINC/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        dPhiSigminusBGNormINCax.fill_between(x_binsNormINC, bin_contentNormINC-self.ME_norm_systematics[i,j], bin_contentNormINC+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        dPhiSigminusBGNormINC.legend()
                        dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$")
                        dPhiSigminusBGNormINCax.set_ylabel("$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$" if k==0 else "")
                        #dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

                        dPhiSigminusBGNormINCax.set_title(f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}") #type:ignore
                        dPhiSigminusBGNormINC.savefig(f"{self.base_save_path}{epString}/dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                        plt.close(dPhiSigminusBGNormINC)
                        axSigminusBGNormINC.errorbar(x_binsNormINC, bin_contentNormINC, yerr=bin_errorsNormINC, fmt="o", color="black")
                        axSigminusBGNormINC.fill_between(x_binsNormINC, bin_contentNormINC-norm_errorsNormINC/N_trig, bin_contentNormINC+norm_errorsNormINC/N_trig, alpha=0.5, color="orange", label="Yield normalization")
                        axSigminusBGNormINC.fill_between(x_binsNormINC, bin_contentNormINC-self.ME_norm_systematics[i,j], bin_contentNormINC+self.ME_norm_systematics[i,j], alpha=0.5, color='red', label="ME normalization") #type:ignore
                        axSigminusBGNormINC.set_xlabel("$\\Delta \\phi$")
                        axSigminusBGNormINC.set_ylabel("$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$")
                        #axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                        






                dEta, dEtaax = plt.subplots(1,1, figsize=(10,6))
                dEtaax = plot_TH1(self.dEtacorrs[i,j], f"$\\Delta \\eta$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c", "$\\Delta \\eta$", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\eta}$", ax=dEtaax) #type:ignore
                dEta.savefig(f"{self.base_save_path}dEta{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                plt.close(dEta)

                dPhidPionSigminusBGNS, dPhidPionSigminusBGNSax = plt.subplots(1,1, figsize=(10,6))
                dPhidPionSigminusBGNS = plot_TH2(self.BGSubtractedAccCorrectedSEdPhidPionSigNScorrs[i,j], f"dPhiSig-BG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c", "$\\Delta \\phi$", "Pion Signal", "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$") #type:ignore
                dPhidPionSigminusBGNS.savefig(f"{self.base_save_path}/dPhidPionSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                plt.close(dPhidPionSigminusBGNS)
                
                if self.analysisType=="pp":
                    # add y axis ticklabels to each subplot 
                    yticks = axBG.get_yticks()
                
                    axBG.set_yticks(yticks)
                    axBG.set_yticklabels([f"{x:0.3f}" for x in axBG.get_yticks()])


                    
                    # add y axis ticklabels to each subplot
                    yticks = axSig.get_yticks()
                    
                    axSig.set_yticks(yticks)
                    axSig.set_yticklabels([f"{x:0.3f}" for x in axSig.get_yticks()])
                    axSig.legend()

                    # add y axis ticklabels to each subplot
                    yticks = axSigminusBGNS.get_yticks()
                    
                    axSigminusBGNS.set_yticks(yticks)
                    axSigminusBGNS.set_yticklabels([f"{x:0.3f}" for x in axSigminusBGNS.get_yticks()])
                    # plot horizontal line at 0
                    axSigminusBGNS.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                    axSigminusBGNS.legend()
                    
                    yticks = axSigminusBGAS.get_yticks()
                    axSigminusBGAS.set_yticks(yticks)
                    axSigminusBGAS.set_yticklabels([f"{x:0.3f}" for x in axSigminusBGAS.get_yticks()])
                    # plot horizontal line at 0
                    axSigminusBGAS.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                    axSigminusBGAS.legend()
                    
                    yticks = axSigminusBGINC.get_yticks()
                    axSigminusBGINC.set_yticks(yticks)
                    axSigminusBGINC.set_yticklabels([f"{x:0.3f}" for x in axSigminusBGINC.get_yticks()])
                    # plot horizontal line at 0
                    axSigminusBGINC.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                    axSigminusBGINC.legend()

                    
                    yticks = axSigminusBGNormINC.get_yticks()
                    axSigminusBGNormINC.set_yticks(yticks)
                    axSigminusBGNormINC.set_yticklabels([f"{x:0.1e}" for x in axSigminusBGNormINC.get_yticks()])
                    # plot horizontal line at 0
                    axSigminusBGNormINC.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                    axSigminusBGNormINC.legend()

                if self.analysisType in ["central", "semicentral"]:
                    # add y axis ticklabels to each subplot 
                    yticks = [ax.get_yticks() for ax in axBG]
                
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axBG)]
                    [ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()]) for ax in axBG]


                    
                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axSig]
                    
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSig)]
                    [ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()]) for ax in axSig]
                    axSig[-1].legend()

                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axSigminusBGNS]
                    
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGNS)]
                    [ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()]) for ax in axSigminusBGNS]
                    # plot horizontal line at 0
                    [ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5) for ax in axSigminusBGNS]
                    axSigminusBGNS[-1].legend()
                    
                    yticks = [ax.get_yticks() for ax in axSigminusBGAS]
                    [ax.set_yticks(yticks[i]) for i,ax in enumerate(axSigminusBGAS)]
                    [ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()]) for ax in axSigminusBGAS]
                    # plot horizontal line at 0
                    [ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5) for ax in axSigminusBGAS]
                    axSigminusBGAS[-1].legend()
                    
                    yticks = [ax.get_yticks() for ax in axSigminusBGINC]
                    [ax.set_yticks(yticks[i]) for i,ax in enumerate(axSigminusBGINC)]
                    [ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()]) for ax in axSigminusBGINC]
                    # plot horizontal line at 0
                    [ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5) for ax in axSigminusBGINC]
                    axSigminusBGINC[-1].legend()

                    
                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINC]
                    [ax.set_yticks(yticks[i]) for i,ax in enumerate(axSigminusBGNormINC)]
                    [ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()]) for ax in axSigminusBGNormINC]
                    # plot horizontal line at 0
                    [ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5) for ax in axSigminusBGNormINC]
                    axSigminusBGNormINC[-1].legend()

                
                
                figBG.savefig(f"{self.base_save_path}dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                figSIG.savefig(f"{self.base_save_path}dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                figSigminusBGNS.savefig(f"{self.base_save_path}dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                figSigminusBGAS.savefig(f"{self.base_save_path}dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                figSigminusBGINC.savefig(f"{self.base_save_path}dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                figSigminusBGNormINC.savefig(f"{self.base_save_path}dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
                plt.close(figBG)
                plt.close(figSIG)
                plt.close(figSigminusBGNS)
                plt.close(figSigminusBGAS)
                plt.close(figSigminusBGINC)
                plt.close(figSigminusBGNormINC)

                self.plot_pion_tpc_signal(i,j)
                self.plot_pion_tpc_signal_for_BG_subtracted_dPhi_dPion(i,j)
        if self.analysisType in ["central", "semicentral"]:
            self.plot_RPFs()
            self.plot_RPFs(withSignal=True)
            self.plot_optimal_parameters()
                

    def plot_dPhi_against_pp_reference(PbPbAna, ppAna, i, j):
        figSigminusBGNormINC, axSigminusBGNormINC = plt.subplots(1,4, figsize=(20,10), sharex=True, sharey=True)
        figSigminusBGNormINC.suptitle("Per-Trigger dPhi Signal - BG compared to pp")
        figSigminusBGNormINC.tight_layout()
        figSigminusBGNormINC.subplots_adjust(wspace=0, hspace=0)
        for k in range(4):
            n_binsNormINC = PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetXaxis().GetNbins() #type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC) #type:ignore
            bin_contentNormINCPbPb = np.zeros(n_binsNormINC) #type:ignore
            bin_errorsNormINCPbPb = np.zeros(n_binsNormINC) #type:ignore
            bin_contentNormINCpp = np.zeros(n_binsNormINC) #type:ignore
            bin_errorsNormINCpp = np.zeros(n_binsNormINC) #type:ignore
            PbPbAna.set_pT_epAngle_bin(i,j,k)
            ppAna.set_pT_epAngle_bin(i,j,k)
            N_trig_PbPb = PbPbAna.get_N_trig()
            N_trig_pp = ppAna.get_N_trig()
            for l in range(n_binsNormINC):
                x_binsNormINC[l] = PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetXaxis().GetBinCenter(l+1) #type:ignore
                bin_contentNormINCPbPb[l] = PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetBinContent(l+1) #type:ignore
                bin_errorsNormINCPbPb[l] = PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j,k].GetBinError(l+1) #type:ignore
                bin_contentNormINCpp[l] = ppAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j].GetBinContent(l+1) #type:ignore
                bin_errorsNormINCpp[l] = ppAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i,j].GetBinError(l+1) #type:ignore
            if k==3:
                # divide bin content by 3
                bin_contentNormINCPbPb = bin_contentNormINCPbPb/3
                bin_errorsNormINCPbPb = bin_errorsNormINCPbPb/3

            axSigminusBGNormINC[k].errorbar(x_binsNormINC, bin_contentNormINCPbPb, yerr=bin_errorsNormINCPbPb, fmt='o', label=f"PbPb, N_trig={N_trig_PbPb}")
            axSigminusBGNormINC[k].errorbar(x_binsNormINC, bin_contentNormINCpp, yerr=bin_errorsNormINCpp, fmt='o', label=f"pp, N_trig={N_trig_pp}")
            axSigminusBGNormINC[k].set_title(f"{PbPbAna.pTtrigBinEdges[i]}-{PbPbAna.pTtrigBinEdges[i+1]} GeV, {PbPbAna.pTassocBinEdges[j]}-{PbPbAna.pTassocBinEdges[j+1]} GeV, {'in-plane' if k==0 else 'mid-plane' if k==1 else 'out-of-plane' if k==2 else 'inclusive'}")
            axSigminusBGNormINC[k].legend()
            axSigminusBGNormINC[k].set_xlabel("dPhi")
            # zoom the y axis in to the plotted points 
            axSigminusBGNormINC[k].set_ylim(1.1*min(bin_contentNormINCPbPb.min(), bin_contentNormINCpp.min()), 1.2*max(bin_contentNormINCPbPb.max(), bin_contentNormINCpp.max()))
            

            #Draw a line at 0 to compare to pp
            axSigminusBGNormINC[k].axhline(0, color='black', linestyle='--')
            

        axSigminusBGNormINC[0].set_ylabel("dPhi Signal - BG")
        figSigminusBGNormINC.savefig(f"{PbPbAna.base_save_path}/dPhi_against_pp_reference_{PbPbAna.pTtrigBinEdges[i]}-{PbPbAna.pTtrigBinEdges[i+1]}GeV_{PbPbAna.pTassocBinEdges[j]}-{PbPbAna.pTassocBinEdges[j+1]}GeV.png")
        plt.close(figSigminusBGNormINC)
        



    def plot_RPFs(self, withSignal=False):
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            for j in range(len(self.pTassocBinEdges)-1): #type:ignore
                self.plot_RPF(i,j, withSignal=withSignal)
                

    def plot_RPF(self, i, j, withSignal=False):
        inPlane  = self.dPhiBGcorrs[i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrs[i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrs[i,j,2] #type:ignore
        fit_y = []
        fit_y_err = []
        full_x = self.get_bin_centers_as_array(inPlane, forFitting=False)
        for k in range(0, len(full_x)):
            fit_y.append(self.RPFObjs[i,j].simultaneous_fit(full_x[k], *self.RPFObjs[i,j].popt)) #type:ignore 
            fit_y_err.append(self.RPFObjs[i,j].simultaneous_fit_err(full_x[k],full_x[1]-full_x[0], *self.RPFObjs[i,j].popt)) #type:ignore
        fit_y = np.array(fit_y, dtype=np.float64)
        fit_y_err = np.array(fit_y_err, dtype=np.float64)


        fig, ax = plt.subplots(2,4, figsize=(20,6), sharey='row', sharex=True, gridspec_kw= {'height_ratios': [0.8, 0.2]})
        # remove margins between plots 
        fig.subplots_adjust(wspace=0, hspace=0)



        ax[0][0].plot(full_x, fit_y[:, 0], label='RPF Fit')
        ax[0][0].fill_between(full_x, fit_y[:, 0]-fit_y_err[:, 0], fit_y[:, 0]+fit_y_err[:, 0], alpha=0.5)
        ax[0][0].errorbar(full_x, self.get_bin_contents_as_array(inPlane, False), yerr=self.get_bin_errors_as_array(inPlane, False), fmt='o', ms=2, label='Background')
        ratVal = (self.get_bin_contents_as_array(inPlane, False)-fit_y[:, 0])/fit_y[:, 0]
        ratErr = 1/fit_y[:, 0]*np.sqrt(self.get_bin_errors_as_array(inPlane, False)**2+(self.get_bin_contents_as_array(inPlane, False)/fit_y[:, 0])**2*fit_y_err[:, 0]**2)
        ax[1][0].fill_between(full_x, ratVal-ratErr, ratVal+ratErr, alpha=0.5)

        ax[0][1].plot(full_x, fit_y[:, 1], label="RPF Fit")
        ax[0][1].fill_between(full_x, fit_y[:, 1]-fit_y_err[:, 1], fit_y[:, 1]+fit_y_err[:, 1], alpha=0.5)
        ax[0][1].errorbar(full_x, self.get_bin_contents_as_array(midPlane, False), yerr=self.get_bin_errors_as_array(midPlane, False), fmt='o', ms=2, label='Background')
        ratVal = (self.get_bin_contents_as_array(midPlane, False)-fit_y[:, 1])/fit_y[:, 1]
        ratErr = 1/fit_y[:, 1]*np.sqrt(self.get_bin_errors_as_array(midPlane, False)**2+(self.get_bin_contents_as_array(midPlane, False)/fit_y[:, 1])**2*fit_y_err[:, 1]**2)
        ax[1][1].fill_between(full_x, ratVal-ratErr, ratVal+ratErr, alpha=0.5)

        ax[0][2].plot(full_x, fit_y[:, 2], label="RPF Fit")
        ax[0][2].fill_between(full_x, fit_y[:, 2]-fit_y_err[:, 2], fit_y[:, 2]+fit_y_err[:, 2], alpha=0.5)
        ax[0][2].errorbar(full_x, self.get_bin_contents_as_array(outPlane, False), yerr=self.get_bin_errors_as_array(outPlane, False), fmt='o', ms=2, label='Background')
        ratVal = (self.get_bin_contents_as_array(outPlane, False)-fit_y[:, 2])/fit_y[:, 2]
        ratErr = 1/fit_y[:, 2]*np.sqrt(self.get_bin_errors_as_array(outPlane, False)**2+(self.get_bin_contents_as_array(outPlane, False)/fit_y[:, 2])**2*fit_y_err[:, 2]**2)
        ax[1][2].fill_between(full_x, ratVal-ratErr, ratVal+ratErr, alpha=0.5)

        ax[0][3].plot(full_x, np.sum(fit_y, axis=1)/3, label="RPF Fit")
        ax[0][3].fill_between(full_x, np.sum(fit_y, axis=1)/3-np.sqrt(np.sum(fit_y_err**2, axis=1)/9), np.sum(fit_y, axis=1)/3+np.sqrt(np.sum(fit_y_err**2, axis=1))/9, alpha=0.5)
        ax[0][3].errorbar(full_x, (self.get_bin_contents_as_array(inPlane, False)+self.get_bin_contents_as_array(midPlane, False)+self.get_bin_contents_as_array(outPlane, False))/3, yerr=np.sqrt((self.get_bin_errors_as_array(inPlane, False)**2+self.get_bin_errors_as_array(midPlane, False)**2+self.get_bin_errors_as_array(outPlane, False)**2)/9), fmt='o', ms=2, label='Background')
        ratVal = ((self.get_bin_contents_as_array(inPlane, False)+self.get_bin_contents_as_array(midPlane, False)+self.get_bin_contents_as_array(outPlane, False))/3-np.sum(fit_y, axis=1)/3)/np.sum(fit_y, axis=1)/3
        ratErr = 1/np.sum(fit_y, axis=1)/3*np.sqrt((np.sqrt((self.get_bin_errors_as_array(inPlane, False)**2+self.get_bin_errors_as_array(midPlane, False)**2+self.get_bin_errors_as_array(outPlane, False)**2)/9)/np.sum(fit_y, axis=1)/3)**2+(np.sum(fit_y, axis=1)/3-np.sqrt(np.sum(fit_y_err**2, axis=1)/9))**2*np.sqrt(np.sum(fit_y_err**2, axis=1))/9)
        ax[1][3].fill_between(full_x, ratVal-ratErr, ratVal+ratErr, alpha=0.5)

        if withSignal:
            inPlaneSig = self.dPhiSigcorrs[i,j,0] #type:ignore
            midPlaneSig = self.dPhiSigcorrs[i,j,1] #type:ignore
            outPlaneSig = self.dPhiSigcorrs[i,j,2] #type:ignore
            ax[0][0].errorbar(full_x, self.get_bin_contents_as_array(inPlaneSig, False), yerr=self.get_bin_errors_as_array(inPlaneSig, False), fmt='o', ms=2, label='Signal')
            # plot (data-fit)/fit on axRatio
            # error will be 1/fit*sqrt(data_err**2+(data/fit)**2*fit_err**2)
            
            
            ax[0][1].errorbar(full_x, self.get_bin_contents_as_array(midPlaneSig, False), yerr=self.get_bin_errors_as_array(midPlaneSig, False), fmt='o', ms=2, label='Signal')
            
            ax[0][2].errorbar(full_x, self.get_bin_contents_as_array(outPlaneSig, False), yerr=self.get_bin_errors_as_array(outPlaneSig, False), fmt='o', ms=2, label='Signal')
            
            ax[0][3].errorbar(full_x, (self.get_bin_contents_as_array(inPlaneSig, False)+self.get_bin_contents_as_array(midPlaneSig, False)+self.get_bin_contents_as_array(outPlaneSig, False))/3, yerr=np.sqrt((self.get_bin_errors_as_array(inPlaneSig, False)**2+self.get_bin_errors_as_array(midPlaneSig, False)**2+self.get_bin_errors_as_array(outPlaneSig, False)**2)/9), fmt='o', ms=2, label='Signal')
            
        ax[0][3].legend()
        [x.set_xlabel(r'$\Delta\phi$') for x in ax[0]]
        [x.set_xlabel(r'$\Delta\phi$') for x in ax[1]]
        [x.autoscale(enable=True, axis='y', tight=True) for x in ax[1]]
        [x.autoscale(enable=True, axis='y', tight=True) for x in ax[0]]
        ax[0][0].set_ylabel(r'$\frac{1}{a*\epsilon}\frac{dN}{d\Delta\phi}$')
        ax[0][0].set_title(f'In-Plane')
        ax[0][1].set_title(f'Mid-Plane')
        ax[0][2].set_title(f'Out-of-Plane')
        ax[0][3].set_title(f'Inclusive')

        fig.suptitle(f'RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, Chi2/NDF = {self.RPFObjs[i,j].chi2OverNDF}') #type:ignore
        fig.savefig(f"{self.base_save_path}RPF{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}{'withSig' if withSignal else ''}.png") #type:ignore

    def plot_optimal_parameters(self):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, 7)) #type:ignore
        optimal_param_errors = np.zeros((len(self.pTtrigBinEdges)-1, len(self.pTassocBinEdges)-1, 7)) #type:ignore
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            for j in range(len(self.pTassocBinEdges)-1): #type:ignore
                optimal_params[i,j] = self.RPFObjs[i,j].popt #type:ignore
                optimal_param_errors[i,j] = np.sqrt(np.diag(self.RPFObjs[i,j].pcov)) #type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc 
        B = optimal_params[:,:,0]
        Berr = optimal_param_errors[:,:,0]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), B[i], yerr=Berr[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$B$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$B$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc.png") #type:ignore
        plt.close(fig)
        # now plot v1 vs ptassoc
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        v1 = optimal_params[:,:,1]
        v1err = optimal_param_errors[:,:,1]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), v1[i], yerr=v1err[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$v_1$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$v_1$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}v1_vs_pTassoc.png") #type:ignore
        plt.close(fig)
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:,:,2]
        v2err = optimal_param_errors[:,:,2]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), v2[i], yerr=v2err[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$v_2$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$v_2$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc.png") #type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:,:,3]
        v3err = optimal_param_errors[:,:,3]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), v3[i], yerr=v3err[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$v_3$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$v_3$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc.png") #type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:,:,4]
        v4err = optimal_param_errors[:,:,4]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), v4[i], yerr=v4err[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$v_4$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$v_4$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc.png") #type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc   
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:,:,5]
        va2err = optimal_param_errors[:,:,5]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), va2[i],yerr=va2err[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$v_{a2}$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$v_{a2}$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc.png") #type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1,3, figsize=(15,5), sharey=True)
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[:,:,6]
        va4err = optimal_param_errors[:,:,6]
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            ax[i].errorbar(self.pTassocBinEdges[:-1]+1/2*np.diff(self.pTassocBinEdges), va4[i],yerr=va4err[i], fmt='o', ms=2) #type:ignore
            ax[i].set_xlabel(r'$p_{T,assoc}$ (GeV/c)')
            ax[i].set_ylabel(r'$v_{a4}$')
            ax[i].set_title(f'{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c') #type:ignore
        fig.suptitle(r'$v_{a4}$ vs $p_{T,assoc}$')
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc.png") #type:ignore

    def plot_one_bin(self, i, j, k):
        epString = "out-of-plane" if k==2 else ("mid-plane" if k==1 else ("in-plane" if k==0 else "inclusive"))
        print(f"Plotting bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}") #type:ignore
        correlation_func = plot_TH2(self.SEcorrs[i,j,k], f"Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\eta$", "$\\Delta \\phi$", "Counts", cmap='jet') #type:ignore
        correlation_func.savefig(f"{self.base_save_path}{epString}/CorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(correlation_func)

        normMEcorrelation_function = plot_TH2(self.NormMEcorrs[i,j,k], f"Normalized Mixed Event Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "\\Delta \\eta", "\\Delta \\phi", "Counts", cmap='jet') #type:ignore
        normMEcorrelation_function.savefig(f"{self.base_save_path}{epString}/NormalizedMixedEventCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(normMEcorrelation_function)

        accCorrectedCorrelationFunction = plot_TH2(self.AccCorrectedSEcorrs[i,j,k], f"Acceptance Corrected Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "\\Delta \\eta", "\\Delta \\phi", "Counts", cmap='jet') #type:ignore
        accCorrectedCorrelationFunction.savefig(f"{self.base_save_path}{epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(accCorrectedCorrelationFunction)

        normAccCorrectedCorrelationFunction = plot_TH2(self.NormAccCorrectedSEcorrs[i,j,k], f"Normalized Acceptance Corrected Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "\\Delta \\eta", "\\Delta \\phi", "Counts", cmap='jet') #type:ignore
        normAccCorrectedCorrelationFunction.savefig(f"{self.base_save_path}{epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(normAccCorrectedCorrelationFunction)

        dPhiBG, dPhiBGax = plt.subplots(1,1, figsize=(5,5))
        dPhiBGax = plot_TH1(self.dPhiBGcorrs[i,j,k], f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "Counts", ax=dPhiBGax) #type:ignore
        dPhiBG.savefig(f"{self.base_save_path}{epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(dPhiBG)

        dPhiSig, dPhiSigax = plt.subplots(1,1, figsize=(5,5))
        dPhiSigax = plot_TH1(self.dPhiSigcorrs[i,j,k], f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {epString}", "$\\Delta \\phi$", "Counts", ax=dPhiSigax) #type:ignore
        dPhiSig.savefig(f"{self.base_save_path}{epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(dPhiSig)

    def plot_pion_tpc_signal(self, i, j):
        pionTPCsignal, pionTPCsignalax = plt.subplots(1,1, figsize=(5,5))
        pionTPCsignalax = plot_TH1(self.pionTPCsignals[i,j], f"Pion TPC signal {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c", "$(\\frac{dE}{dx})_{meas}-(\\frac{dE}{dx})_{calc}$", "Counts", ax=pionTPCsignalax) #type:ignore
        pionTPCsignal.savefig(f"{self.base_save_path}pionTPCsignal{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(pionTPCsignal)

    def plot_pion_tpc_signal_for_BG_subtracted_dPhi_dPion(self, i, j):
        pionTPCsignalNS, pionTPCsignalNSax = plt.subplots(1,1, figsize=(5,5))
        pionTPCsignalNSax = plot_TH1(self.dPionNSsignals[i,j], f"Pion TPC signal {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c", "$(\\frac{dE}{dx})_{meas}-(\\frac{dE}{dx})_{calc}$", "$\\frac{1}{N_{trig}}\\frac{1}{a*\\epsilon}\\frac{d(N_{meas}-N_{RPF}}{d\Delta_{\pi}}$", ax=pionTPCsignalNSax) #type:ignore
        pionTPCsignalNS.savefig(f"{self.base_save_path}dPionNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(pionTPCsignalNS)

        pionTPCsignalAS, pionTPCsignalASax = plt.subplots(1,1, figsize=(5,5))
        pionTPCsignalASax = plot_TH1(self.dPionASsignals[i,j], f"Pion TPC signal {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c", "$(\\frac{dE}{dx})_{meas}-(\\frac{dE}{dx})_{calc}$", "$\\frac{1}{N_{trig}}\\frac{1}{a*\\epsilon}\\frac{d(N_{meas}-N_{RPF}}{d\Delta_{\pi}}$", ax=pionTPCsignalASax) #type:ignore
        pionTPCsignalAS.savefig(f"{self.base_save_path}dPionAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png") #type:ignore
        plt.close(pionTPCsignalAS)
    
    def get_bin_contents_as_array(self, th1, forFitting=True, forAS=False):
        bin_contents = []
        assert(not (forFitting and forAS))
        for i in range(1, th1.GetNbinsX()+1):
            if (th1.GetBinCenter(i) < -np.pi/2 or th1.GetBinCenter(i) > np.pi/2) and forFitting:
                continue
            if (th1.GetBinCenter(i) < np.pi/2 or th1.GetBinCenter(i) > 3*np.pi/2) and forAS:
                continue
            bin_contents.append(th1.GetBinContent(i))
        return np.array(bin_contents)

    def get_2D_bin_contents_as_array(self, th2, forFitting=True, forAS=False):
        bin_contents = []
        assert(not (forFitting and forAS))
        for i in range(1, th2.GetNbinsX()+1):
            if (th2.GetXaxis().GetBinCenter(i) < -np.pi/2 or th2.GetXaxis().GetBinCenter(i) > np.pi/2) and forFitting:
                continue
            if (th2.GetXaxis().GetBinCenter(i) < np.pi/2 or th2.GetXaxis().GetBinCenter(i) > 3*np.pi/2) and forAS:
                continue
            y_contents = []
            for j in range(1, th2.GetNbinsY()+1):
                y_contents.append(th2.GetBinContent(i,j))
                
            bin_contents.append(y_contents)
        return np.array(bin_contents)

    def get_bin_centers_as_array(self, th1, forFitting=True, forAS=False):
        bin_centers = []
        for i in range(1, th1.GetNbinsX()+1):
            if (th1.GetBinCenter(i) < -np.pi/2 or th1.GetBinCenter(i) > np.pi/2) and forFitting:
                continue
            if (th1.GetBinCenter(i) < np.pi/2 or th1.GetBinCenter(i) > 3*np.pi/2) and forAS:
                continue
            bin_centers.append(th1.GetBinCenter(i))
        return np.array(bin_centers)

    def get_2D_bin_centers_as_array(self, th2, forFitting=True, forAS=False):
        x_bin_centers = []
        y_bin_centers = []
        for i in range(1, th2.GetNbinsX()+1):
            if (th2.GetXaxis().GetBinCenter(i) < -np.pi/2 or th2.GetXaxis().GetBinCenter(i) > np.pi/2) and forFitting:
                continue
            if (th2.GetXaxis().GetBinCenter(i) < np.pi/2 or th2.GetXaxis().GetBinCenter(i) > 3*np.pi/2) and forAS:
                continue
            x_bin_centers.append(th2.GetXaxis().GetBinCenter(i))
        for i in range(1, th2.GetNbinsY()+1):
            y_bin_centers.append(th2.GetYaxis().GetBinCenter(i))
        return np.array(x_bin_centers), np.array(y_bin_centers)

    def get_bin_errors_as_array(self, th1, forFitting=True, forAS=False):
        bin_errors = []
        for i in range(1, th1.GetNbinsX()+1):
            
            if (th1.GetBinCenter(i) < -np.pi/2 or th1.GetBinCenter(i) > np.pi/2) and forFitting:
                continue
            if (th1.GetBinCenter(i) < np.pi/2 or th1.GetBinCenter(i) > 3*np.pi/2) and forAS:
                continue
            bin_errors.append(th1.GetBinError(i))
        return np.array(bin_errors)

    def get_2D_bin_errors_as_array(self, th2, forFitting=True, forAS=False):
        bin_errors = []
        for i in range(1, th2.GetNbinsX()+1):
            if (th2.GetXaxis().GetBinCenter(i) < -np.pi/2 or th2.GetXaxis().GetBinCenter(i) > np.pi/2) and forFitting:
                continue
            if (th2.GetXaxis().GetBinCenter(i) < np.pi/2 or th2.GetXaxis().GetBinCenter(i) > 3*np.pi/2) and forAS:
                continue
            y_err = []
            for j in range(1, th2.GetNbinsY()+1):
                
                y_err.append(th2.GetBinError(i,j))
            bin_errors.append(y_err)
        return np.array(bin_errors)
