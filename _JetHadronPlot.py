import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
from matplotlib.pyplot import Axes
from abc import ABC, abstractmethod
import ROOT
import uncertainties
from uncertainties import unumpy as unp
from uncertainties.umath import erf, fsum, fabs
from sklearn.preprocessing import normalize
import logging
import seaborn as sns

from PionTPCNSigmaFitter import PionTPCNSigmaFitter
debug_logger = logging.getLogger('debug')
error_logger = logging.getLogger('error')
info_logger = logging.getLogger('info')


def print_function_name_with_description_on_call(description, logging_level=logging.DEBUG):
    """
    Prints the name of the function and a description of what it does
    """

    def function_wrapper(function):
        def method_wrapper(self, *args, **kwargs):
            if logging_level == logging.DEBUG:
                logger = debug_logger
            elif logging_level == logging.INFO:
                logger = info_logger
            elif logging_level == logging.ERROR:
                logger = error_logger
            else:
                raise ValueError(f"Unknown logging level {logging_level}")
            logger.log(level=logging_level, msg=f"{function.__name__} in {self.__class__.__name__}:\n\t{description}")
            return function(self, *args, **kwargs)

        return method_wrapper

    return function_wrapper


def TH2toArray(hist):
    xbins = hist.GetXaxis().GetNbins()
    ybins = hist.GetYaxis().GetNbins()
    x_centers = np.zeros(xbins)
    y_centers = np.zeros(ybins)
    for i in range(1, xbins + 1):  # +1 because of underflow and overflow
        x_centers[i - 1] = hist.GetXaxis().GetBinCenter(i)
    for i in range(1, ybins + 1):  # +1 because of underflow and overflow
        y_centers[i - 1] = hist.GetYaxis().GetBinCenter(i)

    z = np.zeros((xbins, ybins))
    zerr = np.zeros((xbins, ybins))
    for i in range(1, xbins + 1):
        for j in range(1, ybins + 1):
            z[i - 1, j - 1] = hist.GetBinContent(i, j)
            zerr[i - 1, j - 1] = hist.GetBinError(i, j)
    return x_centers, y_centers, z, zerr


def plot_TH2(hist, title, xlabel, ylabel, zlabel, cmap="viridis", plotStyle="bar3d"):
    xedges, yedges, z, _ = TH2toArray(hist)
    X, Y = np.meshgrid(xedges, yedges)
    fig = plt.figure(figsize=(9, 6))

    # compute difference between element i and i-1 in xedges
    # this is the width of the bin

    if plotStyle == "colz":
        plt.pcolormesh(X, Y, z.T, cmap=cmap)
        plt.colorbar(label=zlabel)
    elif plotStyle == "bar3d":
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        # tight layout

        # make a bar3d plot with bars at xedges, yedges with height z
        # x, y, z are all 1D arrays
        x = []
        y = []
        for i in range(len(xedges)):
            for j in range(len(yedges)):
                x.append(xedges[i])
                y.append(yedges[j])
        x, y, Z = np.array(x), np.array(y), z.ravel()
        bin_width_x, bin_width_y = x[len(yedges) + 1] - x[0], y[1] - y[0]
        dx, dy = (
            np.ones_like(x) * bin_width_x / 2,
            np.ones_like(y) * bin_width_y / 2,
        )
        cmap = cm.get_cmap("coolwarm")  # Get desired colormap - you can change this!
        max_height = np.max(Z)  # get range of colorbars so we can normalize
        min_height = np.min(Z)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k - min_height) / (max_height - min_height)) for k in Z]
        ax.bar3d(
            x,
            y,
            np.ones_like(Z)*(min_height*0.8), # shift the bars down so they are visible
            dx,
            dy,
            Z-min_height*0.8,
            color=rgba,
            zsort="average",
            shade=True,
            edgecolor=None,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_zlabel(zlabel, labelpad=20)
    plt.tight_layout()
    return plt.gcf()


def plot_TH1(hist, title, xlabel, ylabel, ax: Axes, logy=False):
    xbins = hist.GetXaxis().GetNbins()
    x_centers = np.zeros(xbins)
    for i in range(1, xbins + 1):
        x_centers[i - 1] = hist.GetXaxis().GetBinCenter(i)
    z = np.zeros(xbins)
    zerr = np.zeros(xbins)
    for i in range(1, xbins + 1):
        z[i - 1] = hist.GetBinContent(i)
        zerr[i - 1] = hist.GetBinError(i)

    ax.errorbar(x_centers, z, yerr=zerr, fmt="k+")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    return ax



class PlotMixin:
    @print_function_name_with_description_on_call(description="")
    def plot_everything(self):

        font = {"family": "normal", "weight": "bold", "size": 22}

        matplotlib.rc("font", **font)
        matplotlib.rcParams['legend.fontsize'] = 16

        if self.analysisType in ["central", "semicentral"]:  # type:ignore
            numEPBins = 4  # in-, mid-, out-, inclusive
        elif self.analysisType in ["pp"]:  # type:ignore
            numEPBins = 1  # inclusive only
        else:
            numEPBins = 1  # inclusive only
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                debug_logger.info(
                    f"pTtrig: {self.pTtrigBinEdges[i]} - {self.pTtrigBinEdges[i+1]}, pTassoc: {self.pTassocBinEdges[j]} - {self.pTassocBinEdges[j+1]}"
                )  # type:ignore
                figBG, axBG = plt.subplots(1, numEPBins, figsize=(20, 8), sharey=True)
                figSIG, axSig = plt.subplots(1, numEPBins, figsize=(20, 8), sharey=True)
                figSigminusBGNS, axSigminusBGNS = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGAS, axSigminusBGAS = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGINC, axSigminusBGINC = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINC, axSigminusBGNormINC = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCPion, axSigminusBGNormINCPion = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCProton, axSigminusBGNormINCProton = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCKaon, axSigminusBGNormINCKaon = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
               
                figBGZV, axBGZV = plt.subplots(1, numEPBins, figsize=(20, 8), sharey=True)
                figSIGZV, axSigZV = plt.subplots(1, numEPBins, figsize=(20, 8), sharey=True)
                figSigminusBGNSZV, axSigminusBGNSZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGASZV, axSigminusBGASZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGINCZV, axSigminusBGINCZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCZV, axSigminusBGNormINCZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCPionZV, axSigminusBGNormINCPionZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCProtonZV, axSigminusBGNormINCProtonZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
                figSigminusBGNormINCKaonZV, axSigminusBGNormINCKaonZV = plt.subplots(
                    1, numEPBins, figsize=(20, 8), sharey=True
                )
               
                figBG.suptitle(
                    f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSIG.suptitle(
                    f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGNS.suptitle(
                    f"Background subtracted $\\Delta \\phi$ in near-side signal region {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGAS.suptitle(
                    f"Background-subtracted $\\Delta \\phi$ in away-side signal region {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGINC.suptitle(
                    f"Background-subtracted $\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGNormINC.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$"
                )  # type:ignore
                figSigminusBGNormINCPion.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$ for pions"
                )  # type:ignore
                figSigminusBGNormINCProton.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$ for protons"
                )  # type:ignore
                figSigminusBGNormINCKaon.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$ for kaons"
                )  # type:ignore

                figBGZV.suptitle(
                    f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSIGZV.suptitle(
                    f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGNSZV.suptitle(
                    f"Background subtracted $\\Delta \\phi$ in near-side signal region {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGASZV.suptitle(
                    f"Background-subtracted $\\Delta \\phi$ in away-side signal region {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGINCZV.suptitle(
                    f"Background-subtracted $\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
                )  # type:ignore
                figSigminusBGNormINCZV.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$"
                )  # type:ignore
                figSigminusBGNormINCPionZV.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$ for pions"
                )  # type:ignore
                figSigminusBGNormINCProtonZV.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$ for protons"
                )  # type:ignore
                figSigminusBGNormINCKaonZV.suptitle(
                    f"Normalized, background-subtracted $\\Delta \\phi$ for kaons"
                )  # type:ignore

                # remove margins
                figBG.subplots_adjust(wspace=0, hspace=0)
                figSIG.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNS.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGAS.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGINC.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINC.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCPion.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCProton.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCKaon.subplots_adjust(wspace=0, hspace=0)
                figBGZV.subplots_adjust(wspace=0, hspace=0)
                figSIGZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNSZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGASZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGINCZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCPionZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCProtonZV.subplots_adjust(wspace=0, hspace=0)
                figSigminusBGNormINCKaonZV.subplots_adjust(wspace=0, hspace=0)
           

                for k in range(len(self.eventPlaneAngleBinEdges)):  # type:ignore
                    if self.analysisType == "pp":  # type:ignore
                        if k < 3:
                            continue
                    self.epString = (
                        "out-of-plane"
                        if k == 2
                        else (
                            "mid-plane"
                            if k == 1
                            else ("in-plane" if k == 0 else "inclusive")
                        )
                    )

                    self.plot_SE_correlation_function(i, j, k)

                    self.plot_ME_correlation_function(i, j, k)

                    self.plot_acceptance_corrected_SE_correlation_function(i, j, k)
                    self.plot_acceptance_corrected_SE_correlation_function_in_z_vertex_bins(i,j,k)

                    self.plot_normalized_acceptance_corrected_correlation_function(
                        i, j, k
                    )
                    self.plot_normalized_acceptance_corrected_correlation_function_in_z_vertex_bins(i,j,k)

                    self.plot_dPhi_in_background_region(i, j, k, axBG)
                    self.plot_dPhi_in_background_region_in_z_vertex_bins(i, j, k, axBGZV)

                    self.plot_dPhi_in_signal_region(i, j, k, axSig)
                    self.plot_dPhi_in_signal_region_in_z_vertex_bins(i, j, k, axSigZV)

                    if self.analysisType in ["central", "semicentral"]:  # type:ignore
                        (
                            n_binsNS,
                            x_binsNS,
                            bin_contentNS,
                            bin_errorsNS,
                            RPFErrorNS,
                        ) = self.plot_background_subtracted_dPhi_NS(
                            i, j, k, axSigminusBGNS
                        )
                        
                        (
                            n_binsNSZV,
                            x_binsNSZV,
                            bin_contentNSZV,
                            bin_errorsNSZV,
                            RPFErrorNSZV,
                        ) = self.plot_background_subtracted_dPhi_NS_in_z_vertex_bins(
                            i, j, k, axSigminusBGNSZV
                        )

                        (
                            n_binsAS,
                            x_binsAS,
                            bin_contentAS,
                            bin_errorsAS,
                            RPFErrorAS,
                        ) = self.plot_background_subtracted_dPhi_AS(
                            i, j, k, axSigminusBGAS
                        )
                       
                        (
                            n_binsASZV,
                            x_binsASZV,
                            bin_contentASZV,
                            bin_errorsASZV,
                            RPFErrorASZV,
                        ) = self.plot_background_subtracted_dPhi_AS_in_z_vertex_bins(
                            i, j, k, axSigminusBGASZV
                        )

                        RPFErrorINC = self.plot_background_subtracted_dPhi_INC(
                            i,
                            j,
                            k,
                            axSigminusBGINC,
                            n_binsNS,
                            x_binsNS,
                            bin_contentNS,
                            bin_errorsNS,
                            RPFErrorNS,
                            n_binsAS,
                            x_binsAS,
                            bin_contentAS,
                            bin_errorsAS,
                            RPFErrorAS,
                        )
                        
                        RPFErrorINCZV = self.plot_background_subtracted_dPhi_INC_in_z_vertex_bins(
                            i,
                            j,
                            k,
                            axSigminusBGINCZV,
                            n_binsNSZV,
                            x_binsNSZV,
                            bin_contentNSZV,
                            bin_errorsNSZV,
                            RPFErrorNSZV,
                            n_binsASZV,
                            x_binsASZV,
                            bin_contentASZV,
                            bin_errorsASZV,
                            RPFErrorASZV,
                        )

                        self.plot_normalized_background_subtracted_dPhi_INC(
                            i, j, k, axSigminusBGNormINC, RPFErrorINC
                        )
                        self.plot_normalized_background_subtracted_dPhi_INC_in_z_vertex_bins(
                            i, j, k, axSigminusBGNormINCZV, RPFErrorINCZV
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_true_species(
                            i, j, k, "pion", axSigminusBGNormINCPion, RPFErrorINC
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                            i, j, k, "pion", axSigminusBGNormINCPionZV, RPFErrorINCZV
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species(i,j,k,"pion", axSigminusBGNormINCPion, RPFErrorINC)
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(i,j,k,"pion", axSigminusBGNormINCPionZV, RPFErrorINCZV)
                        self.plot_pionTPCnSigma_vs_dphi(i, j, k, "pion")

                        self.plot_normalized_background_subtracted_dPhi_for_true_species(
                            i, j, k, "proton", axSigminusBGNormINCProton, RPFErrorINC
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                            i, j, k, "proton", axSigminusBGNormINCProtonZV, RPFErrorINCZV
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species(i,j,k,"proton", axSigminusBGNormINCProton, RPFErrorINC)
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(i,j,k,"proton", axSigminusBGNormINCProtonZV, RPFErrorINCZV)
                        self.plot_pionTPCnSigma_vs_dphi(i, j, k, "proton")

                        self.plot_normalized_background_subtracted_dPhi_for_true_species(
                            i, j, k, "kaon", axSigminusBGNormINCKaon, RPFErrorINC
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                            i, j, k, "kaon", axSigminusBGNormINCKaonZV, RPFErrorINCZV
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species(i,j,k,"kaon", axSigminusBGNormINCKaon, RPFErrorINC)
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(i,j,k,"kaon", axSigminusBGNormINCKaonZV, RPFErrorINCZV)
                        self.plot_pionTPCnSigma_vs_dphi(i, j, k, "kaon")

                    elif self.analysisType in ["pp"]:  # type:ignore
                        (
                            n_binsNS,
                            x_binsNS,
                            bin_contentNS,
                            bin_errorsNS,
                            normErrorNS,
                        ) = self.plot_background_subtracted_dPhi_NS(
                            i, j, k, axSigminusBGNS
                        )
                        
                        (
                            n_binsNSZV,
                            x_binsNSZV,
                            bin_contentNSZV,
                            bin_errorsNSZV,
                            normErrorNSZV,
                        ) = self.plot_background_subtracted_dPhi_NS_in_z_vertex_bins(
                            i, j, k, axSigminusBGNSZV
                        )

                        (
                            n_binsAS,
                            x_binsAS,
                            bin_contentAS,
                            bin_errorsAS,
                            normErrorAS,
                        ) = self.plot_background_subtracted_dPhi_AS(
                            i, j, k, axSigminusBGAS
                        )
                        
                        (
                            n_binsASZV,
                            x_binsASZV,
                            bin_contentASZV,
                            bin_errorsASZV,
                            normErrorASZV,
                        ) = self.plot_background_subtracted_dPhi_AS_in_z_vertex_bins(
                            i, j, k, axSigminusBGASZV
                        )

                        normErrorINC = self.plot_background_subtracted_dPhi_INC(
                            i,
                            j,
                            k,
                            axSigminusBGINC,
                            n_binsNS,
                            x_binsNS,
                            bin_contentNS,
                            bin_errorsNS,
                            normErrorNS,
                            n_binsAS,
                            x_binsAS,
                            bin_contentAS,
                            bin_errorsAS,
                            normErrorAS,
                        )
                        
                        normErrorINCZV = self.plot_background_subtracted_dPhi_INC_in_z_vertex_bins(
                            i,
                            j,
                            k,
                            axSigminusBGINCZV,
                            n_binsNSZV,
                            x_binsNSZV,
                            bin_contentNSZV,
                            bin_errorsNSZV,
                            normErrorNSZV,
                            n_binsASZV,
                            x_binsASZV,
                            bin_contentASZV,
                            bin_errorsASZV,
                            normErrorASZV,
                        )

                        self.plot_normalized_background_subtracted_dPhi_INC(
                            i, j, k, axSigminusBGNormINC, normErrorINC
                        )

                        self.plot_normalized_background_subtracted_dPhi_INC_in_z_vertex_bins(
                            i, j, k, axSigminusBGNormINCZV, normErrorINCZV
                        )

                        self.plot_normalized_background_subtracted_dPhi_for_true_species(
                            i, j, k, "pion", axSigminusBGNormINCPion, normErrorINC
                        )

                        self.plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                            i, j, k, "pion", axSigminusBGNormINCPionZV, normErrorINCZV
                        )

                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species(i,j,k,"pion", axSigminusBGNormINCPion, normErrorINC)
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(i,j,k,"pion", axSigminusBGNormINCPionZV, normErrorINCZV)
                        # self.plot_pionTPCnSigma_vs_dphi(i, j, k, "pion")

                        self.plot_normalized_background_subtracted_dPhi_for_true_species(
                            i, j, k, "proton", axSigminusBGNormINCProton, normErrorINC
                        )
                        
                        self.plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                            i, j, k, "proton", axSigminusBGNormINCProtonZV, normErrorINCZV
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species(i,j,k,"proton", axSigminusBGNormINCProton, normErrorINC)
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(i,j,k,"proton", axSigminusBGNormINCProtonZV, normErrorINCZV)
                        # self.plot_pionTPCnSigma_vs_dphi(i, j, k, "proton")

                        self.plot_normalized_background_subtracted_dPhi_for_true_species(
                            i, j, k, "kaon", axSigminusBGNormINCKaon, normErrorINC
                        )

                        self.plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
                            i, j, k, "kaon", axSigminusBGNormINCKaonZV, normErrorINCZV
                        )
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species(i,j,k,"kaon", axSigminusBGNormINCKaon, normErrorINC)
                        self.plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(i,j,k,"kaon", axSigminusBGNormINCKaonZV, normErrorINCZV)
                        # self.plot_pionTPCnSigma_vs_dphi(i, j, k, "kaon")

                        # self.plot_pion_tpc_nSigma("pion", i, j,k)
                        # self.plot_pion_tpc_nSigma("kaon", i, j,k)
                        # self.plot_pion_tpc_nSigma("proton", i, j,k)

                        self.plot_PionTPCNSigmaFit(i,j,k, "NS")
                        self.plot_PionTPCNSigmaFit(i,j,k, "AS")
                        self.plot_PionTPCNSigmaFit(i,j,k, "BG")

                 

                dEta, dEtaax = plt.subplots(1, 1, figsize=(10, 6))
                dEtaax = plot_TH1(
                    self.dEtacorrs[i, j],
                    f"$\\Delta \\eta$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c",
                    "$\\Delta \\eta$",
                    "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\eta}$",
                    ax=dEtaax,
                )  # type:ignore
                dEta.savefig(
                    f"{self.base_save_path}dEta{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore
                plt.close(dEta)

                # axis management
                if self.analysisType == "pp":
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
                    axSigminusBGNS.set_yticklabels(
                        [f"{x:0.3f}" for x in axSigminusBGNS.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNS.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNS.legend()

                    yticks = axSigminusBGAS.get_yticks()
                    axSigminusBGAS.set_yticks(yticks)
                    axSigminusBGAS.set_yticklabels(
                        [f"{x:0.3f}" for x in axSigminusBGAS.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGAS.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGAS.legend()

                    yticks = axSigminusBGINC.get_yticks()
                    axSigminusBGINC.set_yticks(yticks)
                    axSigminusBGINC.set_yticklabels(
                        [f"{x:0.3f}" for x in axSigminusBGINC.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGINC.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGINC.legend()

                    yticks = axSigminusBGNormINC.get_yticks()
                    axSigminusBGNormINC.set_yticks(yticks)
                    axSigminusBGNormINC.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINC.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINC.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINC.legend()

#                     if self.analysisType in ['central', 'semicentral']:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j, k].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j, k].pcov
#                         fit_func = self.PionTPCNSigmaFitObjs[i,j, k].upiKpInc_generalized_fit
#                     else:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j].pcov
#                         fit_func = self.PionTPCNSigmaFitObjs[i,j].upiKpInc_generalized_fit
#                     mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = uncertainties.correlated_values(fit_params, fit_errors)
#                     int_x = np.linspace(-10, 10, 1000)
#                     int_y = fit_func(int_x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak)
# 
# 
#                     protonEnhNorm = np.trapz(int_y[:1000], int_x)
#                     pionEnhNorm  = np.trapz(int_y[1000:2000], int_x)
#                     kaonEnhNorm  = np.trapz(int_y[2000:3000], int_x)
#                     inclusiveNorm= np.trapz(int_y[3000:], int_x)
#                     
#                     gpp = np.trapz(generalized_gauss(int_x, mup, sigp, app, alphap), int_x)/protonEnhNorm
#                     gppi = np.trapz(gauss(int_x, mupi, sigpi, apip), int_x)/protonEnhNorm
#                     gpk = np.trapz(generalized_gauss(int_x, muk, sigk, akp, alphak), int_x)/protonEnhNorm
#                     gpip = np.trapz(generalized_gauss(int_x, mup, sigp, appi, alphap), int_x)/pionEnhNorm
#                     gpipi = np.trapz(gauss(int_x, mupi, sigpi, apipi), int_x)/pionEnhNorm
#                     gpik = np.trapz(generalized_gauss(int_x, muk, sigk, akpi, alphak), int_x)/pionEnhNorm
#                     gkp = np.trapz(generalized_gauss(int_x, mup, sigp, apk, alphap), int_x)/kaonEnhNorm
#                     gkpi = np.trapz(gauss(int_x, mupi, sigpi, apik), int_x)/kaonEnhNorm
#                     gkk = np.trapz(generalized_gauss(int_x, muk, sigk, akk, alphak), int_x)/kaonEnhNorm
#                     gincp = np.trapz(generalized_gauss(int_x, mup, sigp, apinc, alphap), int_x)/inclusiveNorm
#                     gincpi = np.trapz(gauss(int_x, mupi, sigpi, apiinc), int_x)/inclusiveNorm
#                     ginck = np.trapz(generalized_gauss(int_x, muk, sigk, akinc, alphak), int_x)/inclusiveNorm

                    
#                     determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
# 
#                     inv_mat = unp.ulinalg.pinv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
#                     # Calculate the L1 norm of each row
#                     row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#                     
#                     # Divide each row by its corresponding L1 norm
#                     inv_mat = inv_mat / row_sums[:, np.newaxis]
#                     
#                     
#                     pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#                     proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#                     kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"

                    panel_pp_text1 = "ALICE Work In Progress" + "\n" + r"pp $\sqrt{s_{NN}}$=5.02TeV"+ "\n" + f"{self.pTtrigBinEdges[i]}"+r"$<p_{T unc, jet}^{ch+ne}<$"+f"{self.pTtrigBinEdges[i+1]}" + "\n"  + "anti-$k_T$ R=0.2"
                    panel_pp_text2 =  f"{self.pTassocBinEdges[j]}"+r"$<p_T^{assoc.}<$"+f"{self.pTassocBinEdges[j+1]}" + "\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
                    x_locs_pp = [0.65, 0.8]
                    y_locs_pp = [0.75, 0.75]
                    font_size = 20

                    yticks = axSigminusBGNormINCPion.get_yticks()
                    axSigminusBGNormINCPion.set_yticks(yticks)
                    axSigminusBGNormINCPion.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCPion.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCPion.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCPion.text(x_locs_pp[0], y_locs_pp[0], panel_pp_text1, 
                                                 fontsize=font_size, ha='right', transform=axSigminusBGNormINCPion.transAxes)
                    axSigminusBGNormINCPion.text(x_locs_pp[1], y_locs_pp[1], panel_pp_text2,
                                                    fontsize=font_size, ha='right', transform=axSigminusBGNormINCPion.transAxes)
                    axSigminusBGNormINCPion.legend()

                    yticks = axSigminusBGNormINCProton.get_yticks()
                    axSigminusBGNormINCProton.set_yticks(yticks)
                    axSigminusBGNormINCProton.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCProton.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCProton.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCProton.text(x_locs_pp[0], y_locs_pp[0], panel_pp_text1,
                                                   fontsize=font_size, ha='right', transform=axSigminusBGNormINCProton.transAxes)
                    axSigminusBGNormINCProton.text(x_locs_pp[1], y_locs_pp[1], panel_pp_text2,
                                                    fontsize=font_size, ha='right', transform=axSigminusBGNormINCProton.transAxes)
                    axSigminusBGNormINCProton.legend()

                    yticks = axSigminusBGNormINCKaon.get_yticks()
                    axSigminusBGNormINCKaon.set_yticks(yticks)
                    axSigminusBGNormINCKaon.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCKaon.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCKaon.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCKaon.text(x_locs_pp[0], y_locs_pp[0], panel_pp_text1,
                                                    fontsize=font_size, ha='right', transform=axSigminusBGNormINCKaon.transAxes)
                    axSigminusBGNormINCKaon.text(x_locs_pp[1], y_locs_pp[1], panel_pp_text2,
                                                    fontsize=font_size, ha='right', transform=axSigminusBGNormINCKaon.transAxes)
                    axSigminusBGNormINCKaon.legend()
                    
                    #Z-Vertex binned
                    
                    # add y axis ticklabels to each subplot
                    yticks = axBGZV.get_yticks()

                    axBGZV.set_yticks(yticks)
                    axBGZV.set_yticklabels([f"{x:0.3f}" for x in axBGZV.get_yticks()])

                    # add y axis ticklabels to each subplot
                    yticks = axSigZV.get_yticks()

                    axSigZV.set_yticks(yticks)
                    axSigZV.set_yticklabels([f"{x:0.3f}" for x in axSigZV.get_yticks()])
                    axSigZV.legend()

                    # add y axis ticklabels to each subplot
                    yticks = axSigminusBGNSZV.get_yticks()

                    axSigminusBGNSZV.set_yticks(yticks)
                    axSigminusBGNSZV.set_yticklabels(
                        [f"{x:0.3f}" for x in axSigminusBGNSZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNSZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNSZV.legend()

                    yticks = axSigminusBGASZV.get_yticks()
                    axSigminusBGASZV.set_yticks(yticks)
                    axSigminusBGASZV.set_yticklabels(
                        [f"{x:0.3f}" for x in axSigminusBGASZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGASZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGASZV.legend()

                    yticks = axSigminusBGINCZV.get_yticks()
                    axSigminusBGINCZV.set_yticks(yticks)
                    axSigminusBGINCZV.set_yticklabels(
                        [f"{x:0.3f}" for x in axSigminusBGINCZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGINCZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGINCZV.legend()

                    yticks = axSigminusBGNormINCZV.get_yticks()
                    axSigminusBGNormINCZV.set_yticks(yticks)
                    axSigminusBGNormINCZV.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCZV.legend()

#                     if self.analysisType in ['central', 'semicentral']:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j, k].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j, k].pcov
#                     else:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j].pcov
#                     mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc,  = uncertainties.correlated_values(fit_params, fit_errors)
# 
#                     protonEnhNorm = -(np.pi/2)**.5*(
#                         akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                         app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                         )
#                     pionEnhNorm = -(np.pi/2)**.5*(
#                         akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                         appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                         )
#                     kaonEnhNorm = -(np.pi/2)**.5*(
#                         apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                         apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                         )
#                     
#                     fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#                     fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#                     fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#                     fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#                     fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#                     fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#                     fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#                     fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#                     fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#                     
#                     determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
#                     
#                     
#                     inv_mat = unp.ulinalg.pinv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
#                     
#                     row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
# 
#                     inv_mat = inv_mat/row_sums[:,np.newaxis]
# 
#                     
#                     pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#                     proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#                     kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"

                    panel_pp_text1 = "ALICE Work In Progress" + "\n" + r"pp $\sqrt{s_{NN}}$=5.02TeV"+ "\n" + f"{self.pTtrigBinEdges[i]}"+r"$<p_{T unc, jet}^{ch+ne}<$"+f"{self.pTtrigBinEdges[i+1]}" + "\n"  + "anti-$k_T$ R=0.2"
                    panel_pp_text2 =  f"{self.pTassocBinEdges[j]}"+r"$<p_T^{assoc.}<$"+f"{self.pTassocBinEdges[j+1]}" + "\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
                    x_locs_pp = [0.65, 0.8]
                    y_locs_pp = [0.75, 0.75]
                    font_size = 20


                    yticks = axSigminusBGNormINCPionZV.get_yticks()
                    axSigminusBGNormINCPionZV.set_yticks(yticks)
                    axSigminusBGNormINCPionZV.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCPionZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCPionZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCPionZV.text(x_locs_pp[0], y_locs_pp[0],
                                panel_pp_text1,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGNormINCPionZV.transAxes) # type:ignore
                    axSigminusBGNormINCPionZV.text(x_locs_pp[1], y_locs_pp[1],
                                panel_pp_text2,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGNormINCPionZV.transAxes) # type:ignore
                    axSigminusBGNormINCPionZV.legend()

                    yticks = axSigminusBGNormINCProtonZV.get_yticks()
                    axSigminusBGNormINCProtonZV.set_yticks(yticks)
                    axSigminusBGNormINCProtonZV.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCProtonZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCProtonZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCProtonZV.text(x_locs_pp[0], y_locs_pp[0],
                                                     panel_pp_text1,
                                                     fontsize=font_size,
                                                     ha='right',
                                                     transform=axSigminusBGNormINCProtonZV.transAxes)  # type:ignore
                    axSigminusBGNormINCProtonZV.text(x_locs_pp[1], y_locs_pp[1],
                                                     panel_pp_text2,
                                                        fontsize=font_size,
                                                        ha='right',
                                                        transform=axSigminusBGNormINCProtonZV.transAxes)  # type:ignore
                    axSigminusBGNormINCProtonZV.legend()

                    yticks = axSigminusBGNormINCKaonZV.get_yticks()
                    axSigminusBGNormINCKaonZV.set_yticks(yticks)
                    axSigminusBGNormINCKaonZV.set_yticklabels(
                        [f"{x:0.1e}" for x in axSigminusBGNormINCKaonZV.get_yticks()]
                    )
                    # plot horizontal line at 0
                    axSigminusBGNormINCKaonZV.axhline(
                        y=0, color="k", linestyle="--", linewidth=0.5
                    )
                    axSigminusBGNormINCKaonZV.text(x_locs_pp[0], y_locs_pp[0],
                                                     panel_pp_text1,
                                                     fontsize=font_size,
                                                     ha='right',
                                                     transform=axSigminusBGNormINCKaonZV.transAxes)  # type:ignore
                    axSigminusBGNormINCKaonZV.text(x_locs_pp[1], y_locs_pp[1],
                                                     panel_pp_text2,
                                                        fontsize=font_size,
                                                        ha='right',
                                                        transform=axSigminusBGNormINCKaonZV.transAxes)  # type:ignore
                    axSigminusBGNormINCKaonZV.legend()

                   

                if self.analysisType in ["central", "semicentral"]:
                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axBG]

                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axBG)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axBG
                    ]

                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axSig]

                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSig)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSig
                    ]
                    axSig[-1].legend()

                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axSigminusBGNS]

                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGNS)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNS
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNS
                    ]
                    axSigminusBGNS[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGAS]
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGAS)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigminusBGAS
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGAS
                    ]
                    axSigminusBGAS[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGINC]
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGINC)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigminusBGINC
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGINC
                    ]
                    axSigminusBGINC[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINC]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINC)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINC
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINC
                    ]
                    axSigminusBGNormINC[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCPion]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCPion)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCPion
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCPion
                    ]
                    axSigminusBGNormINCPion[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCProton]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCProton)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCProton
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCProton
                    ]
                    axSigminusBGNormINCProton[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCKaon]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCKaon)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCKaon
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCKaon
                    ]
                    axSigminusBGNormINCKaon[-1].legend()

                    panel1_text = f"{self.pTassocBinEdges[j]}"+r"$<p_T^{assoc.}<$"+f"{self.pTassocBinEdges[j+1]}" + "\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
                    panel2_text = r"Pb-Pb $\sqrt{s_{NN}}$=5.02TeV, "+("0-10\%" if self.analysisType=="central" else "30-50\%") + "\n" + f"{self.pTtrigBinEdges[i]}"+r"$<p_{T unc, jet}^{ch+ne}<$"+f"{self.pTtrigBinEdges[i+1]}" + "\n"  + "anti-$k_T$ R=0.2"
                    panel3_text = "ALICE Work In Progress" + "\n" + r"Background: 0.8 $< |\Delta \eta |< $1.2" + "\n" + r"Signal: $|\Delta \eta | <$ 0.6 for $|\Delta \phi| > \pi/2$" + "\n" +r"Signal: $|\Delta \eta | <$ 1.2 for $\pi/2 < \Delta \phi < 3\pi/2$"
                    
#                     if self.analysisType in ['central', 'semicentral']:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j, k].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j, k].pcov
#                     else:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j].pcov
#                     mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc = uncertainties.correlated_values(fit_params, fit_errors)
# 
#                     protonEnhNorm = -(np.pi/2)**.5*(
#                         akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                         app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                         )
#                     pionEnhNorm = -(np.pi/2)**.5*(
#                         akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                         appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                         )
#                     kaonEnhNorm = -(np.pi/2)**.5*(
#                         apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                         apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                         )
#                     
#                     fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#                     fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#                     fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#                     fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#                     fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#                     fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#                     fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#                     fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#                     fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#                     
#                     determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
#                     
#                     
#                     inv_mat = unp.ulinalg.pinv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
# 
#                     row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#                     inv_mat = inv_mat/row_sums[:,np.newaxis]                    
#                     
#                     pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#                     proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#                     kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"


                    

                    x_locs = [0.9, 0.9, 0.9]
                    y_locs = [0.85, 0.85, 0.85]
                    font_size = 14

                    axBG[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axBG[0].transAxes) # type:ignore
                    axBG[1].text(x_locs[1], y_locs[1],
                                panel2_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axBG[1].transAxes) # type:ignore
                    axBG[2].text(x_locs[2], y_locs[1],
                                panel3_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axBG[2].transAxes)
                    
                    axSig[0].text(x_locs[0], y_locs[0],
                              panel1_text ,
                              fontsize=font_size,
                              ha='right',
                                  transform=axSig[0].transAxes) # type:ignore
                    axSig[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSig[1].transAxes)
                    axSig[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSig[2].transAxes)
                    axSigminusBGNS[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGNS[0].transAxes)
                    axSigminusBGNS[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNS[1].transAxes)
                    axSigminusBGNS[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNS[2].transAxes)
                    
                    axSigminusBGAS[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGAS[0].transAxes)
                    axSigminusBGAS[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGAS[1].transAxes)
                    axSigminusBGAS[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGAS[2].transAxes)
                    
                    axSigminusBGINC[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGINC[0].transAxes)
                    axSigminusBGINC[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGINC[1].transAxes)
                    axSigminusBGINC[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGINC[2].transAxes)
                    
                    axSigminusBGNormINC[0].text(x_locs[0], y_locs[0],
                                    panel1_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNormINC[0].transAxes)
                    axSigminusBGNormINC[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNormINC[1].transAxes)
                    axSigminusBGNormINC[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNormINC[2].transAxes)
                    
                    axSigminusBGNormINCPion[0].text(x_locs[0], y_locs[0],
                                                panel1_text ,
                                                fontsize=font_size,
                                                ha='right',
                                                transform = axSigminusBGNormINCPion[0].transAxes)
                    axSigminusBGNormINCPion[1].text(x_locs[1], y_locs[1],
                                                    panel2_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCPion[1].transAxes)
                    axSigminusBGNormINCPion[2].text(x_locs[2], y_locs[2],
                                                    panel3_text, 
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCPion[2].transAxes)
                    
                    axSigminusBGNormINCProton[0].text(x_locs[0], y_locs[0],
                                                    panel1_text,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCProton[0].transAxes)
                    axSigminusBGNormINCProton[1].text(x_locs[1], y_locs[1],
                                                    panel2_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCProton[1].transAxes)
                    axSigminusBGNormINCProton[2].text(x_locs[2], y_locs[2],
                                                    panel3_text,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCProton[2].transAxes)
                    
                    axSigminusBGNormINCKaon[0].text(x_locs[0], y_locs[0],
                                                    panel1_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCKaon[0].transAxes)
                    axSigminusBGNormINCKaon[1].text(x_locs[1], y_locs[1],
                                                    panel2_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCKaon[1].transAxes)
                    axSigminusBGNormINCKaon[2].text(x_locs[2], y_locs[2],
                                                    panel3_text,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCKaon[2].transAxes)
                    
                    #Z-vertex binned
                    
                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axBGZV]

                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axBGZV)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axBGZV
                    ]

                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axSigZV]

                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigZV)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigZV
                    ]
                    axSigZV[-1].legend()

                    # add y axis ticklabels to each subplot
                    yticks = [ax.get_yticks() for ax in axSigminusBGNSZV]

                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGNSZV)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNSZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNSZV
                    ]
                    axSigminusBGNSZV[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGASZV]
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGASZV)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigminusBGASZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGASZV
                    ]
                    axSigminusBGASZV[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGINCZV]
                    [ax.set_yticks(yticks[i]) for i, ax in enumerate(axSigminusBGINCZV)]
                    [
                        ax.set_yticklabels([f"{x:0.3f}" for x in ax.get_yticks()])
                        for ax in axSigminusBGINCZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGINCZV
                    ]
                    axSigminusBGINCZV[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCZV]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCZV)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCZV
                    ]
                    axSigminusBGNormINCZV[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCPionZV]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCPionZV)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCPionZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCPionZV
                    ]
                    axSigminusBGNormINCPionZV[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCProtonZV]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCProtonZV)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCProtonZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCProtonZV
                    ]
                    axSigminusBGNormINCProtonZV[-1].legend()

                    yticks = [ax.get_yticks() for ax in axSigminusBGNormINCKaonZV]
                    [
                        ax.set_yticks(yticks[i])
                        for i, ax in enumerate(axSigminusBGNormINCKaonZV)
                    ]
                    [
                        ax.set_yticklabels([f"{x:0.1e}" for x in ax.get_yticks()])
                        for ax in axSigminusBGNormINCKaonZV
                    ]
                    # plot horizontal line at 0
                    [
                        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
                        for ax in axSigminusBGNormINCKaonZV
                    ]
                    axSigminusBGNormINCKaonZV[-1].legend()

                    panel1_text = f"{self.pTassocBinEdges[j]}"+r"$<p_T^{assoc.}<$"+f"{self.pTassocBinEdges[j+1]}" + "\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
                    panel2_text = r"Pb-Pb $\sqrt{s_{NN}}$=5.02TeV, "+("0-10\%" if self.analysisType=="central" else "30-50\%") + "\n" + f"{self.pTtrigBinEdges[i]}"+r"$<p_{T unc, jet}^{ch+ne}<$"+f"{self.pTtrigBinEdges[i+1]}" + "\n"  + "anti-$k_T$ R=0.2"
                    panel3_text = "ALICE Work In Progress" + "\n" + r"Background: 0.8 $< |\Delta \eta |< $1.2" + "\n" + r"Signal: $|\Delta \eta | <$ 0.6 for $|\Delta \phi| > \pi/2$" + "\n" +r"Signal: $|\Delta \eta | <$ 1.2 for $\pi/2 < \Delta \phi < 3\pi/2$"
                    
#                     if self.analysisType in ['central', 'semicentral']:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j, k].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j, k].pcov
#                     else:
#                         fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
#                         fit_errors = self.PionTPCNSigmaFitObjs[i,j].pcov
#                     mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc = uncertainties.correlated_values(fit_params, fit_errors)
# 
#                     protonEnhNorm = -(np.pi/2)**.5*(
#                         akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                         app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                         )
#                     pionEnhNorm = -(np.pi/2)**.5*(
#                         akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                         appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                         )
#                     kaonEnhNorm = -(np.pi/2)**.5*(
#                         apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                         apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                         akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                         )
#                     
#                     fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#                     fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#                     fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#                     fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#                     fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#                     fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#                     fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#                     fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#                     fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#                     
#                     determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
#                     
#                     inv_mat = unp.ulinalg.pinv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
#                     
#                     row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#                     inv_mat = inv_mat/row_sums[:,np.newaxis]
#                     
#                     pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#                     proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#                     kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"

                    

                    x_locs = [0.9, 0.9, 0.9]
                    y_locs = [0.85, 0.85, 0.85]
                    font_size = 14

                    axBGZV[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axBGZV[0].transAxes) # type:ignore
                    axBGZV[1].text(x_locs[1], y_locs[1],
                                panel2_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axBGZV[1].transAxes) # type:ignore
                    axBGZV[2].text(x_locs[2], y_locs[1],
                                panel3_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axBGZV[2].transAxes)
                    
                    axSigZV[0].text(x_locs[0], y_locs[0],
                              panel1_text ,
                              fontsize=font_size,
                              ha='right',
                                  transform=axSigZV[0].transAxes) # type:ignore
                    axSigZV[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigZV[1].transAxes)
                    axSigZV[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigZV[2].transAxes)
                    axSigminusBGNSZV[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGNSZV[0].transAxes)
                    axSigminusBGNSZV[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNSZV[1].transAxes)
                    axSigminusBGNSZV[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNSZV[2].transAxes)
                    
                    axSigminusBGASZV[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGASZV[0].transAxes)
                    axSigminusBGASZV[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGASZV[1].transAxes)
                    axSigminusBGASZV[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGASZV[2].transAxes)
                    
                    axSigminusBGINCZV[0].text(x_locs[0], y_locs[0],
                                panel1_text ,
                                fontsize=font_size,
                                ha='right',
                                    transform=axSigminusBGINCZV[0].transAxes)
                    axSigminusBGINCZV[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGINCZV[1].transAxes)
                    axSigminusBGINCZV[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGINCZV[2].transAxes)
                    
                    axSigminusBGNormINCZV[0].text(x_locs[0], y_locs[0],
                                    panel1_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNormINCZV[0].transAxes)
                    axSigminusBGNormINCZV[1].text(x_locs[1], y_locs[1],
                                    panel2_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNormINCZV[1].transAxes)
                    axSigminusBGNormINCZV[2].text(x_locs[2], y_locs[2],
                                    panel3_text ,
                                    fontsize=font_size,
                                    ha='right',
                                        transform=axSigminusBGNormINCZV[2].transAxes)
                    
                    axSigminusBGNormINCPionZV[0].text(x_locs[0], y_locs[0],
                                                panel1_text ,
                                                fontsize=font_size,
                                                ha='right',
                                                transform = axSigminusBGNormINCPionZV[0].transAxes)
                    axSigminusBGNormINCPionZV[1].text(x_locs[1], y_locs[1],
                                                    panel2_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCPionZV[1].transAxes)
                    axSigminusBGNormINCPionZV[2].text(x_locs[2], y_locs[2],
                                                    panel3_text,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCPionZV[2].transAxes)
                    
                    axSigminusBGNormINCProtonZV[0].text(x_locs[0], y_locs[0],
                                                    panel1_text,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCProtonZV[0].transAxes)
                    axSigminusBGNormINCProtonZV[1].text(x_locs[1], y_locs[1],
                                                    panel2_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCProtonZV[1].transAxes)
                    axSigminusBGNormINCProtonZV[2].text(x_locs[2], y_locs[2],
                                                    panel3_text  + '\n'+proton_comp_str,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCProtonZV[2].transAxes)
                    
                    axSigminusBGNormINCKaonZV[0].text(x_locs[0], y_locs[0],
                                                    panel1_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCKaonZV[0].transAxes)
                    axSigminusBGNormINCKaonZV[1].text(x_locs[1], y_locs[1],
                                                    panel2_text ,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCKaonZV[1].transAxes)
                    axSigminusBGNormINCKaonZV[2].text(x_locs[2], y_locs[2],
                                                    panel3_text,
                                                    fontsize=font_size,
                                                    ha='right',
                                                    transform = axSigminusBGNormINCKaonZV[2].transAxes)

                figBG.tight_layout()
                figBG.savefig(
                    f"{self.base_save_path}dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore

                
                figSIG.tight_layout()
                figSIG.savefig(
                    f"{self.base_save_path}dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore

                
                figSigminusBGNS.tight_layout()
                figSigminusBGNS.savefig(
                    f"{self.base_save_path}dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore

                
                figSigminusBGAS.tight_layout()
                figSigminusBGAS.savefig(
                    f"{self.base_save_path}dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore

                
                figSigminusBGINC.tight_layout()
                figSigminusBGINC.savefig(
                    f"{self.base_save_path}dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore

                
                figSigminusBGNormINC.tight_layout()
                figSigminusBGNormINC.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore

                
                figSigminusBGNormINCPion.tight_layout()
                figSigminusBGNormINCPion.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm-pion{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore
                
                figSigminusBGNormINCProton.tight_layout()
                figSigminusBGNormINCProton.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm-proton{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore
                
                figSigminusBGNormINCKaon.tight_layout()
                figSigminusBGNormINCKaon.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm-kaon{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
                )  # type:ignore
                
                
                figBGZV.tight_layout()
                figBGZV.savefig(
                    f"{self.base_save_path}dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore

                
                figSIGZV.tight_layout()
                figSIGZV.savefig(
                    f"{self.base_save_path}dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore

                
                figSigminusBGNSZV.tight_layout()
                figSigminusBGNSZV.savefig(
                    f"{self.base_save_path}dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore

                
                figSigminusBGASZV.tight_layout()
                figSigminusBGASZV.savefig(
                    f"{self.base_save_path}dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore

                
                figSigminusBGINCZV.tight_layout()
                figSigminusBGINCZV.savefig(
                    f"{self.base_save_path}dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore

                
                figSigminusBGNormINCZV.tight_layout()
                figSigminusBGNormINCZV.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore

                
                figSigminusBGNormINCPionZV.tight_layout()
                figSigminusBGNormINCPionZV.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm-pion{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore
                
                figSigminusBGNormINCProtonZV.tight_layout()
                figSigminusBGNormINCProtonZV.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm-proton{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore
                
                figSigminusBGNormINCKaonZV.tight_layout()
                figSigminusBGNormINCKaonZV.savefig(
                    f"{self.base_save_path}dPhiSig-BG-Norm-kaon{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
                )  # type:ignore
               
                plt.close(figBG)
                plt.close(figSIG)
                plt.close(figSigminusBGNS)
                plt.close(figSigminusBGAS)
                plt.close(figSigminusBGINC)
                plt.close(figSigminusBGNormINC)
                plt.close(figSigminusBGNormINCPion)
                plt.close(figSigminusBGNormINCProton)
                plt.close(figSigminusBGNormINCKaon)
                
                plt.close(figBGZV)
                plt.close(figSIGZV)
                plt.close(figSigminusBGNSZV)
                plt.close(figSigminusBGASZV)
                plt.close(figSigminusBGINCZV)
                plt.close(figSigminusBGNormINCZV)
                plt.close(figSigminusBGNormINCPionZV)
                plt.close(figSigminusBGNormINCProtonZV)
                plt.close(figSigminusBGNormINCKaonZV)

                # self.plot_pion_tpc_signal(i, j)
                
        self.plot_inclusive_yield()
        self.plot_inclusive_yield_in_z_vertex_bins()

        self.plot_yield_for_true_species("pion")
        self.plot_yield_for_true_species("kaon")
        self.plot_yield_for_true_species("proton")

        self.plot_yield_for_true_species_in_z_vertex_bins("pion")
        self.plot_yield_for_true_species_in_z_vertex_bins("kaon")
        self.plot_yield_for_true_species_in_z_vertex_bins("proton")



        self.plot_Ntrig()

        if self.analysisType in ["central", "semicentral"]:
            self.plot_RPFs()
            self.plot_RPFs(withSignal=True)
            self.plot_RPFs(withSignal=True, withCharles=True)
            self.plot_RPFs_for_species(withSignal=True)
            self.plot_RPFs_in_z_vertex_bins()
            self.plot_RPFs_in_z_vertex_bins(withSignal=True)
            self.plot_RPFs_in_z_vertex_bins(withSignal=True, withCharles=True)
            self.plot_RPFs_for_species_in_z_vertex_bins(withSignal=True)
            self.plot_optimal_parameters(0)
            self.plot_optimal_parameters(1)
            self.plot_optimal_parameters_for_true_species(0, "pion")
            self.plot_optimal_parameters_for_true_species(0, "proton")
            self.plot_optimal_parameters_for_true_species(0, "kaon")
            self.plot_optimal_parameters_for_true_species(1, "pion")
            self.plot_optimal_parameters_for_true_species(1, "proton")
            self.plot_optimal_parameters_for_true_species(1, "kaon")
            self.plot_optimal_parameters_in_z_vertex_bins(0)
            self.plot_optimal_parameters_in_z_vertex_bins(1)
            self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(0, "pion")
            self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(0, "proton")
            self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(0, "kaon")
            self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(1, "pion")
            self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(1, "proton")
            self.plot_optimal_parameters_for_true_species_in_z_vertex_bins(1, "kaon")


        if self.analysisType in ["central", "semicentral"]:
            self.plot_event_plane_angle()

    @print_function_name_with_description_on_call(description="")
    def plot_Ntrig(self):
        """
        Plot the number of triggers for each event plane angle
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if self.analysisType =='pp':
            ax.errorbar(
                self.pTtrigBinCenters,
                [self.N_trigs[i] for i in range(len(self.pTtrigBinCenters))],
                yerr=np.sqrt(
                    np.array([self.N_trigs[i] for i in range(len(self.pTtrigBinCenters))])
                ),
                xerr=np.array(self.pTtrigBinWidths)/2,
                fmt="o",
            )
        else:
            for k in range(len(self.eventPlaneAngleBinEdges)):
                ax.errorbar(
                    self.pTtrigBinCenters,
                    [self.N_trigs[i][k] for i in range(len(self.pTtrigBinCenters))],
                    yerr=np.sqrt(
                        np.array(
                            [self.N_trigs[i][k] for i in range(len(self.pTtrigBinCenters))]
                        )
                    ),
                    xerr=np.array(self.pTtrigBinWidths)/2,
                    fmt="o" if k==3 else "+" if k==0 else "x" if k==1 else "d" if k==2 else "o",
                    label = "in" if k==0 else "mid" if k==1 else "out" if k==2 else "inclusive"
                )
        ax.set_xlabel("Trigger $p_T$ (GeV/c)")
        ax.set_ylabel("$N_{trigger}$")
        ax.set_title(f"Ntrig, {self.analysisType}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}Ntrig.png")  # type:ignore
        plt.close(fig)

    @print_function_name_with_description_on_call(description="")
    def plot_event_plane_angle(self):
        """
        Plot the event plane angle for each run
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        epAngle = self.EventPlaneAngleHist[0]
        for i in range(len(self.EventPlaneAngleHist)):
            if self.EventPlaneAngleHist[i] == None:
                continue
            epAngle.Add(self.EventPlaneAngleHist[i])
        if epAngle==None:
            plt.close(fig)
            return 
        plot_TH1(
            epAngle,
            f"Event Plane Angle, {self.analysisType}",
            "Event Plane Angle",
            "Counts",
            ax=ax,
        )  # type:ignore
        fig.savefig(f"{self.base_save_path}EventPlaneAngle.png")  # type:ignore
        plt.close(fig)

    @print_function_name_with_description_on_call(description="")
    def plot_dPhi_against_pp_reference(PbPbAna, ppAna, i, j, plot_ME_systematic=False):
        figSigminusBGNormINC, axSigminusBGNormINC = plt.subplots(
            1, 4, figsize=(20, 10), sharex=True, sharey=True
        )
        figSigminusBGNormINC.suptitle("Per-Trigger $\Delta\phi$ Signal - BG compared to pp")
        figSigminusBGNormINC.tight_layout()
        figSigminusBGNormINC.subplots_adjust(wspace=0, hspace=0)

        for k in range(4):
            n_binsNormINC = (
                PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINCPbPb = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINCPbPb = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINCpp = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINCpp = np.zeros(n_binsNormINC)  # type:ignore
            norm_errorsNormINC = (
                ppAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[i, j]
            )  # type:ignore
            RPFErrors = np.array(
                [
                    PbPbAna.RPFObjs[i, j].simultaneous_fit_err(
                        x_binsNormINC[l], x_binsNormINC[1] - x_binsNormINC[0], *PbPbAna.RPFObjs[i, j].popt
                    )
                    for l in range(len(x_binsNormINC))
                ]
            ) 
            PbPbAna.set_pT_epAngle_bin(i, j, k)
            ppAna.set_pT_epAngle_bin(i, j, k)
            N_trig_PbPb = PbPbAna.get_N_trig()
            N_trig_pp = ppAna.get_N_trig()
            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINCPbPb[
                    l
                ] = PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINCPbPb[
                    l
                ] = PbPbAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore
                bin_contentNormINCpp[
                    l
                ] = ppAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                    i, j
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINCpp[
                    l
                ] = ppAna.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                    i, j
                ].GetBinError(
                    l + 1
                )  # type:ignore
            if k == 3:
                # divide bin content by 3
                bin_contentNormINCPbPb = bin_contentNormINCPbPb 
                bin_errorsNormINCPbPb = bin_errorsNormINCPbPb 

            axSigminusBGNormINC[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINCPbPb,
                yerr=bin_errorsNormINCPbPb,
                fmt="o",
                label=f"PbPb, N_trig={N_trig_PbPb}",
            )
            axSigminusBGNormINC[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINCPbPb - RPFErrors[:, k] / N_trig_PbPb
                if k != 3
                else bin_contentNormINCPbPb*3
                - np.sqrt(np.sum(RPFErrors ** 2, axis=1) ) / N_trig_PbPb,
                bin_contentNormINCPbPb + RPFErrors[:, k] / N_trig_PbPb
                if k != 3
                else bin_contentNormINCPbPb*3
                + np.sqrt(np.sum(RPFErrors**2, axis=1) ) / N_trig_PbPb,
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            if plot_ME_systematic:
                axSigminusBGNormINC[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINCPbPb - PbPbAna.ME_norm_systematics[i, j, k]/N_trig_PbPb,
                    bin_contentNormINCPbPb + PbPbAna.ME_norm_systematics[i, j, k]/N_trig_PbPb,
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            axSigminusBGNormINC[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINCpp,
                yerr=bin_errorsNormINCpp,
                fmt="o",
                label=f"pp, N_trig={N_trig_pp}",
            )
            axSigminusBGNormINC[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINCpp - norm_errorsNormINC / N_trig_pp,
                bin_contentNormINCpp + norm_errorsNormINC / N_trig_pp,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNormINC[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINCpp - ppAna.ME_norm_systematics[i, j]/N_trig_pp,
                    bin_contentNormINCpp + ppAna.ME_norm_systematics[i, j]/N_trig_pp,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            axSigminusBGNormINC[(k+1)%4].set_title(
                f"{PbPbAna.pTtrigBinEdges[i]}-{PbPbAna.pTtrigBinEdges[i+1]} GeV, {PbPbAna.pTassocBinEdges[j]}-{PbPbAna.pTassocBinEdges[j+1]} GeV, {'in-plane' if k==0 else 'mid-plane' if k==1 else 'out-of-plane' if k==2 else 'inclusive'}"
            )
            axSigminusBGNormINC[(k+1)%4].set_xlabel("$\Delta\phi$")
            # zoom the y axis in to the plotted points
            axSigminusBGNormINC[(k+1)%4].set_ylim(
                1.1 * min(bin_contentNormINCPbPb.min(), bin_contentNormINCpp.min()),
                1.2 * max(bin_contentNormINCPbPb.max(), bin_contentNormINCpp.max()),
            )

            # Draw a line at 0 to compare to pp
            axSigminusBGNormINC[(k+1)%4].axhline(0, color="black", linestyle="--")

        axSigminusBGNormINC[-1].legend()
        axSigminusBGNormINC[0].set_ylabel("$\\frac{1}{N_{trig}}\\frac{1}{a\\epsilon}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$")
        figSigminusBGNormINC.tight_layout()
        figSigminusBGNormINC.savefig(
            f"{PbPbAna.base_save_path}/dPhi_against_pp_reference_{PbPbAna.pTtrigBinEdges[i]}-{PbPbAna.pTtrigBinEdges[i+1]}GeV_{PbPbAna.pTassocBinEdges[j]}-{PbPbAna.pTassocBinEdges[j+1]}GeV.png"
        )
        plt.close(figSigminusBGNormINC)

    @print_function_name_with_description_on_call(description="")
    def plot_RPFs(self, withSignal=False, withCharles=False):
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.plot_RPF(i, j, withSignal=withSignal, withCharles=withCharles)

    @print_function_name_with_description_on_call(description="")
    def plot_RPFs_in_z_vertex_bins(self, withSignal=False, withCharles=False):
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.plot_RPF_in_z_vertex_bins(i, j, withSignal=withSignal, withCharles=withCharles)

    def plot_RPFs_for_species(self, withSignal=False):
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                for species in ['pion', 'kaon', 'proton']:
                    self.plot_RPF_for_species(i, j, species, withSignal=withSignal)

    def plot_RPFs_for_species_in_z_vertex_bins(self, withSignal=False):
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                for species in ['pion', 'kaon', 'proton']:
                    self.plot_RPF_for_species_in_z_vertex_bins(i, j, species, withSignal=withSignal)

    @print_function_name_with_description_on_call(description="")
    def plot_RPF(self, i, j, withSignal=False, withCharles=False):
        inPlane = self.dPhiBGcorrs[i, j, 0]  # type:ignore
        midPlane = self.dPhiBGcorrs[i, j, 1]  # type:ignore
        outPlane = self.dPhiBGcorrs[i, j, 2]  # type:ignore
        fit_y = []
        fit_y_err = []
        full_x = self.get_bin_centers_as_array(inPlane, forFitting=False)
        for k in range(0, len(full_x)):
            fit_y.append(
                self.RPFObjs[i, j].simultaneous_fit(full_x[k], *self.RPFObjs[i, j].popt)
            )  # type:ignore
            fit_y_err.append(
                self.RPFObjs[i, j].simultaneous_fit_err(
                    full_x[k], full_x[1] - full_x[0], *self.RPFObjs[i, j].popt
                )
            )  # type:ignore
        fit_y = np.array(fit_y, dtype=np.float64)
        fit_y_err = np.array(fit_y_err, dtype=np.float64)

        fig, ax = plt.subplots(
            2,
            4,
            figsize=(20, 8),
            sharey="row",
            sharex=True,
            gridspec_kw={"height_ratios": [0.8, 0.2]},
        )
        # remove margins between plots
        fig.subplots_adjust(wspace=0, hspace=0)

        N_trig = self.N_trigs[i] # type:ignore

        #++++++++++++++++++IN-PLANE+++++++++++++++++++
        ax[0][1].plot(full_x, fit_y[:, 0]/N_trig[0], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,0]**2/N_trig[0]**5 + fit_y_err[:,0]**2/N_trig[0]**2)
        ax[0][1].fill_between(
            full_x,
            fit_y[:, 0]/N_trig[0] - normalized_err,
            fit_y[:, 0]/N_trig[0] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlane, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlane, False)**2/N_trig[0]**2)
        ax[0][1].errorbar(
            full_x,
            self.get_bin_contents_as_array(inPlane, False)/N_trig[0],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (self.get_bin_contents_as_array(inPlane, False) - fit_y[:, 0]) / fit_y[
            :, 0
        ]
        ratErr = (
            1
            / fit_y[:, 0]
            * np.sqrt(
                self.get_bin_errors_as_array(inPlane, False) ** 2
                + (self.get_bin_contents_as_array(inPlane, False) / fit_y[:, 0]) ** 2
                * fit_y_err[:, 0] ** 2
            )
        )
        ax[1][1].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        
        #++++++++++++++++++MID-PLANE+++++++++++++++++++
        ax[0][2].plot(full_x, fit_y[:, 1]/N_trig[1], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,1]**2/N_trig[1]**5 + fit_y_err[:,1]**2/N_trig[1]**2)
        ax[0][2].fill_between(
            full_x,
            fit_y[:, 1]/N_trig[1] - normalized_err,
            fit_y[:, 1]/N_trig[1] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlane, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlane, False)**2/N_trig[1]**2)
        ax[0][2].errorbar(
            full_x,
            self.get_bin_contents_as_array(midPlane, False)/N_trig[1],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(midPlane, False) - fit_y[:, 1]
        ) / fit_y[:, 1]
        ratErr = (
            1
            / fit_y[:, 1]
            * np.sqrt(
                self.get_bin_errors_as_array(midPlane, False) ** 2
                + (self.get_bin_contents_as_array(midPlane, False) / fit_y[:, 1]) ** 2
                * fit_y_err[:, 1] ** 2
            )
        )
        ax[1][2].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][3].plot(full_x, fit_y[:, 2]/N_trig[2], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,2]**2/N_trig[2]**5 + fit_y_err[:,2]**2/N_trig[2]**2)
        ax[0][3].fill_between(
            full_x,
            fit_y[:, 2]/N_trig[2] - normalized_err,
            fit_y[:, 2]/N_trig[2] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlane, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlane, False)**2/N_trig[2]**2)
        ax[0][3].errorbar(
            full_x,
            self.get_bin_contents_as_array(outPlane, False)/N_trig[2],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(outPlane, False) - fit_y[:, 2]
        ) / fit_y[:, 2]
        ratErr = (
            1
            / fit_y[:, 2]
            * np.sqrt(
                self.get_bin_errors_as_array(outPlane, False) ** 2
                + (self.get_bin_contents_as_array(outPlane, False) / fit_y[:, 2]) ** 2
                * fit_y_err[:, 2] ** 2
            )
        )
        ax[1][3].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][0].plot(full_x, (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3, label="RPF Fit")
        normalized_err = np.sqrt(np.sum(fit_y, axis=1)**2/N_trig[3]**5 + np.sum(fit_y_err**2, axis=1)/N_trig[3]**2)
        ax[0][0].fill_between(
            full_x,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 - normalized_err/3,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 + normalized_err/3,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlane, False) + self.get_bin_contents_as_array(midPlane, False) + self.get_bin_contents_as_array(outPlane, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlane, False)**2 + self.get_bin_errors_as_array(midPlane, False)**2 + self.get_bin_errors_as_array(outPlane, False)**2)/N_trig[3]**2)
        ax[0][0].errorbar(
            full_x,
            ((
                self.get_bin_contents_as_array(inPlane, False)/N_trig[0]
                + self.get_bin_contents_as_array(midPlane, False)/N_trig[1]
                + self.get_bin_contents_as_array(outPlane, False)/N_trig[2]
            )
            )/3,
            yerr=normalized_data_err/3,
            fmt="o",
            ms=2,
            label="Background",
        )
        
        ratVal = (
            (
                self.get_bin_contents_as_array(inPlane, False)
                - fit_y[:, 0]
            ) / fit_y[:, 0]
                + 
            (
            self.get_bin_contents_as_array(midPlane, False)
                - fit_y[:, 1]
            ) / fit_y[:, 1]
                + 
            (
            self.get_bin_contents_as_array(outPlane, False)
                - fit_y[:, 2]
            ) / fit_y[:, 2]
            ) / 3

        ratErr = 1/3 * np.sqrt(
            self.get_bin_contents_as_array(inPlane, False)**2/fit_y[:, 0]**4 * fit_y_err[:, 0]**2 
            +
            self.get_bin_contents_as_array(midPlane, False)**2/fit_y[:, 1]**4 * fit_y_err[:, 1]**2
            +
            self.get_bin_contents_as_array(outPlane, False)**2/fit_y[:, 2]**4 * fit_y_err[:, 2]**2
            + 
            1/fit_y[:, 0]**2 * self.get_bin_errors_as_array(inPlane, False)**2
            +
            1/fit_y[:, 1]**2 * self.get_bin_errors_as_array(midPlane, False)**2
            +
            1/fit_y[:, 2]**2 * self.get_bin_errors_as_array(outPlane, False)**2 
         )
        
        ax[1][0].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        if withSignal:
            inPlaneSig = self.dPhiSigcorrs[i, j, 0]  # type:ignore
            midPlaneSig = self.dPhiSigcorrs[i, j, 1]  # type:ignore
            outPlaneSig = self.dPhiSigcorrs[i, j, 2]  # type:ignore
            
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlaneSig, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlaneSig, False)**2/N_trig[0]**2)
            ax[0][1].errorbar(
                full_x,
                self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            # plot (data-fit)/fit on axRatio
            # error will be 1/fit*sqrt(data_err**2+(data/fit)**2*fit_err**2)
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlaneSig, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlaneSig, False)**2/N_trig[1]**2)
            ax[0][2].errorbar(
                full_x,
                self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlaneSig, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlaneSig, False)**2/N_trig[2]**2)
            ax[0][3].errorbar(
                full_x,
                self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlaneSig, False) + self.get_bin_contents_as_array(midPlaneSig, False) + self.get_bin_contents_as_array(outPlaneSig, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlaneSig, False)**2 + self.get_bin_errors_as_array(midPlaneSig, False)**2 + self.get_bin_errors_as_array(outPlaneSig, False)**2)/N_trig[3]**2)
            ax[0][0].errorbar(
                full_x,
                ((
                    self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0]
                    + self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1]
                    + self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2]
                )
                )/3,
                yerr=normalized_data_err/3,
                fmt="o",
                ms=2,
                label="Signal",
            )

        if withCharles and i==0:
            IPfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsIP.root")
            OOPfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsOOP.root")
            MPfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsMP.root")
            INCfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsINC.root")
            # first get the number of triggers in the correct pt assoc and trig bins
            num_trig_hist_IP = IPfile.Get("jetH_number_of_triggers")
            num_trig_hist_MP = MPfile.Get("jetH_number_of_triggers")
            num_trig_hist_OOP = OOPfile.Get("jetH_number_of_triggers")
            num_trig_hist_INC = INCfile.Get("jetH_number_of_triggers")
            # sum the bin values between the correct pt bins
            N_trig_IP = np.sum(
                np.array(
                    [
                        num_trig_hist_IP.GetBinContent(i) for i in range(
                            num_trig_hist_IP.GetXaxis().FindBin(self.pTtrigBinEdges[i]),   
                            num_trig_hist_IP.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            N_trig_MP = np.sum(
                np.array(
                    [
                        num_trig_hist_MP.GetBinContent(i) for i in range(
                            num_trig_hist_MP.GetXaxis().FindBin(self.pTtrigBinEdges[i]),
                            num_trig_hist_MP.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            N_trig_OOP = np.sum(
                np.array(
                    [
                        num_trig_hist_OOP.GetBinContent(i) for i in range(
                            num_trig_hist_OOP.GetXaxis().FindBin(self.pTtrigBinEdges[i]),
                            num_trig_hist_OOP.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            N_trig_INC = np.sum(
                np.array(
                    [
                        num_trig_hist_INC.GetBinContent(i) for i in range(
                            num_trig_hist_INC.GetXaxis().FindBin(self.pTtrigBinEdges[i]),
                            num_trig_hist_INC.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            # Now get the deltphi histograms and scale by ntrig 
            
            IP_sig_hist = IPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            IP_bg_hist = IPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            MP_sig_hist = MPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            MP_bg_hist = MPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            OOP_sig_hist = OOPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            OOP_bg_hist = OOPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            INC_sig_hist = INCfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            INC_bg_hist = INCfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            # scale by ntrig
            IP_sig_hist.Scale(1/N_trig_IP)
            IP_bg_hist.Scale(1/N_trig_IP)

            MP_sig_hist.Scale(1/N_trig_MP)
            MP_bg_hist.Scale(1/N_trig_MP)

            OOP_sig_hist.Scale(1/N_trig_OOP)
            OOP_bg_hist.Scale(1/N_trig_OOP)

            INC_sig_hist.Scale(1/N_trig_INC)
            INC_bg_hist.Scale(1/N_trig_INC)

            # scale by the bin width
            dphiBinWidth = IP_sig_hist.GetXaxis().GetBinWidth(1)
            IP_sig_hist.Scale(1/dphiBinWidth)
            IP_bg_hist.Scale(1/dphiBinWidth)

            MP_sig_hist.Scale(1/dphiBinWidth)
            MP_bg_hist.Scale(1/dphiBinWidth)

            OOP_sig_hist.Scale(1/dphiBinWidth)
            OOP_bg_hist.Scale(1/dphiBinWidth)

            INC_sig_hist.Scale(1/dphiBinWidth)
            INC_bg_hist.Scale(1/dphiBinWidth)

            # now get the bin contents and errors as arrays 

            IP_sig_contents = self.get_bin_contents_as_array(IP_sig_hist, False)
            IP_bg_contents = self.get_bin_contents_as_array(IP_bg_hist, False)
            IP_sig_err = self.get_bin_errors_as_array(IP_sig_hist, False)
            IP_bg_err = self.get_bin_errors_as_array(IP_bg_hist, False)

            MP_sig_contents = self.get_bin_contents_as_array(MP_sig_hist, False)
            MP_bg_contents = self.get_bin_contents_as_array(MP_bg_hist, False)
            MP_sig_err = self.get_bin_errors_as_array(MP_sig_hist, False)
            MP_bg_err = self.get_bin_errors_as_array(MP_bg_hist, False)

            OOP_sig_contents = self.get_bin_contents_as_array(OOP_sig_hist, False)
            OOP_bg_contents = self.get_bin_contents_as_array(OOP_bg_hist, False)
            OOP_sig_err = self.get_bin_errors_as_array(OOP_sig_hist, False)
            OOP_bg_err = self.get_bin_errors_as_array(OOP_bg_hist, False)

            INC_sig_contents = self.get_bin_contents_as_array(INC_sig_hist, False)
            INC_bg_contents = self.get_bin_contents_as_array(INC_bg_hist, False)
            INC_sig_err = self.get_bin_errors_as_array(INC_sig_hist, False)
            INC_bg_err = self.get_bin_errors_as_array(INC_bg_hist, False)

            # now plot the results
            ax[0][1].errorbar(
                full_x,
                IP_sig_contents,
                yerr=IP_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][1].errorbar(
                full_x,
                IP_bg_contents,
                yerr=IP_bg_err,
                fmt="o",
                label=f"Charles Background",
            )

            ax[0][2].errorbar(
                full_x,
                MP_sig_contents,
                yerr=MP_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][2].errorbar(
                full_x,
                MP_bg_contents,
                yerr=MP_bg_err,
                fmt="o",
                label=f"Charles Background",
            )

            ax[0][3].errorbar(
                full_x,
                OOP_sig_contents,
                yerr=OOP_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][3].errorbar(
                full_x,
                OOP_bg_contents,
                yerr=OOP_bg_err,
                fmt="o",
                label=f"Charles Background",
            )

            ax[0][0].errorbar(
                full_x,
                INC_sig_contents,
                yerr=INC_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][0].errorbar(
                full_x,
                INC_bg_contents,
                yerr=INC_bg_err,
                fmt="o",
                label=f"Charles Background",
            )


        ax[0][3].legend()
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[0]]
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[0]]
        ax[0][0].set_ylabel(r"$\frac{1}{N_trig}\frac{1}{a*\epsilon}\frac{dN}{d\Delta\phi}$")
        ax[0][1].set_title(f"In-Plane")
        ax[0][2].set_title(f"Mid-Plane")
        ax[0][3].set_title(f"Out-of-Plane")
        ax[0][0].set_title(f"Inclusive")
        # add text to the axes that says "inclusive = (in+mid+out)/3"
        #ax[0][3].text( 0.1, 0.1, "Inclusive = (In+Mid+Out)", transform = ax[0][3].transAxes,)
        fig.suptitle(
            f"RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, Chi2/NDF = {self.RPFObjs[i,j].chi2OverNDF}"
        )  # type:ignore
        fig.tight_layout()
        fig.savefig(
            f"{self.base_save_path}RPF{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}{'withSig' if withSignal else ''}{'withCharles' if withCharles else ''}.png"
        )  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_RPF_in_z_vertex_bins(self, i, j, withSignal=False, withCharles=False):
        inPlane = self.dPhiBGcorrsZV[i, j, 0]  # type:ignore
        midPlane = self.dPhiBGcorrsZV[i, j, 1]  # type:ignore
        outPlane = self.dPhiBGcorrsZV[i, j, 2]  # type:ignore
        fit_y = []
        fit_y_err = []
        full_x = self.get_bin_centers_as_array(inPlane, forFitting=False)
        for k in range(0, len(full_x)):
            fit_y.append(
                self.RPFObjsZV[i, j].simultaneous_fit(full_x[k], *self.RPFObjsZV[i, j].popt)
            )  # type:ignore
            fit_y_err.append(
                self.RPFObjsZV[i, j].simultaneous_fit_err(
                    full_x[k], full_x[1] - full_x[0], *self.RPFObjsZV[i, j].popt
                )
            )  # type:ignore
        fit_y = np.array(fit_y, dtype=np.float64)
        fit_y_err = np.array(fit_y_err, dtype=np.float64)

        fig, ax = plt.subplots(
            2,
            4,
            figsize=(20, 8),
            sharey="row",
            sharex=True,
            gridspec_kw={"height_ratios": [0.8, 0.2]},
        )
        # remove margins between plots
        fig.subplots_adjust(wspace=0, hspace=0)

        N_trig = self.N_trigs[i] # type:ignore

        #++++++++++++++++++IN-PLANE+++++++++++++++++++
        ax[0][1].plot(full_x, fit_y[:, 0]/N_trig[0], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,0]**2/N_trig[0]**5 + fit_y_err[:,0]**2/N_trig[0]**2)
        ax[0][1].fill_between(
            full_x,
            fit_y[:, 0]/N_trig[0] - normalized_err,
            fit_y[:, 0]/N_trig[0] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlane, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlane, False)**2/N_trig[0]**2)
        ax[0][1].errorbar(
            full_x,
            self.get_bin_contents_as_array(inPlane, False)/N_trig[0],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (self.get_bin_contents_as_array(inPlane, False) - fit_y[:, 0]) / fit_y[
            :, 0
        ]
        ratErr = (
            1
            / fit_y[:, 0]
            * np.sqrt(
                self.get_bin_errors_as_array(inPlane, False) ** 2
                + (self.get_bin_contents_as_array(inPlane, False) / fit_y[:, 0]) ** 2
                * fit_y_err[:, 0] ** 2
            )
        )
        ax[1][1].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        
        #++++++++++++++++++MID-PLANE+++++++++++++++++++
        ax[0][2].plot(full_x, fit_y[:, 1]/N_trig[1], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,1]**2/N_trig[1]**5 + fit_y_err[:,1]**2/N_trig[1]**2)
        ax[0][2].fill_between(
            full_x,
            fit_y[:, 1]/N_trig[1] - normalized_err,
            fit_y[:, 1]/N_trig[1] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlane, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlane, False)**2/N_trig[1]**2)
        ax[0][2].errorbar(
            full_x,
            self.get_bin_contents_as_array(midPlane, False)/N_trig[1],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(midPlane, False) - fit_y[:, 1]
        ) / fit_y[:, 1]
        ratErr = (
            1
            / fit_y[:, 1]
            * np.sqrt(
                self.get_bin_errors_as_array(midPlane, False) ** 2
                + (self.get_bin_contents_as_array(midPlane, False) / fit_y[:, 1]) ** 2
                * fit_y_err[:, 1] ** 2
            )
        )
        ax[1][2].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][3].plot(full_x, fit_y[:, 2]/N_trig[2], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,2]**2/N_trig[2]**5 + fit_y_err[:,2]**2/N_trig[2]**2)
        ax[0][3].fill_between(
            full_x,
            fit_y[:, 2]/N_trig[2] - normalized_err,
            fit_y[:, 2]/N_trig[2] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlane, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlane, False)**2/N_trig[2]**2)
        ax[0][3].errorbar(
            full_x,
            self.get_bin_contents_as_array(outPlane, False)/N_trig[2],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(outPlane, False) - fit_y[:, 2]
        ) / fit_y[:, 2]
        ratErr = (
            1
            / fit_y[:, 2]
            * np.sqrt(
                self.get_bin_errors_as_array(outPlane, False) ** 2
                + (self.get_bin_contents_as_array(outPlane, False) / fit_y[:, 2]) ** 2
                * fit_y_err[:, 2] ** 2
            )
        )
        ax[1][3].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][0].plot(full_x, (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3, label="RPF Fit")
        normalized_err = np.sqrt(np.sum(fit_y, axis=1)**2/N_trig[3]**5 + np.sum(fit_y_err**2, axis=1)/N_trig[3]**2)
        ax[0][0].fill_between(
            full_x,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 - normalized_err/3,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 + normalized_err/3,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlane, False) + self.get_bin_contents_as_array(midPlane, False) + self.get_bin_contents_as_array(outPlane, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlane, False)**2 + self.get_bin_errors_as_array(midPlane, False)**2 + self.get_bin_errors_as_array(outPlane, False)**2)/N_trig[3]**2)
        ax[0][0].errorbar(
            full_x,
            ((
                self.get_bin_contents_as_array(inPlane, False)/N_trig[0]
                + self.get_bin_contents_as_array(midPlane, False)/N_trig[1]
                + self.get_bin_contents_as_array(outPlane, False)/N_trig[2]
            )
            )/3,
            yerr=normalized_data_err/3,
            fmt="o",
            ms=2,
            label="Background",
        )
        
        ratVal = (
            (
                self.get_bin_contents_as_array(inPlane, False)
                - fit_y[:, 0]
            ) / fit_y[:, 0]
                + 
            (
            self.get_bin_contents_as_array(midPlane, False)
                - fit_y[:, 1]
            ) / fit_y[:, 1]
                + 
            (
            self.get_bin_contents_as_array(outPlane, False)
                - fit_y[:, 2]
            ) / fit_y[:, 2]
            ) / 3

        ratErr = 1/3 * np.sqrt(
            self.get_bin_contents_as_array(inPlane, False)**2/fit_y[:, 0]**4 * fit_y_err[:, 0]**2 
            +
            self.get_bin_contents_as_array(midPlane, False)**2/fit_y[:, 1]**4 * fit_y_err[:, 1]**2
            +
            self.get_bin_contents_as_array(outPlane, False)**2/fit_y[:, 2]**4 * fit_y_err[:, 2]**2
            + 
            1/fit_y[:, 0]**2 * self.get_bin_errors_as_array(inPlane, False)**2
            +
            1/fit_y[:, 1]**2 * self.get_bin_errors_as_array(midPlane, False)**2
            +
            1/fit_y[:, 2]**2 * self.get_bin_errors_as_array(outPlane, False)**2 
         )
        
        ax[1][0].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        if withSignal:
            inPlaneSig = self.dPhiSigcorrsZV[i, j, 0]  # type:ignore
            midPlaneSig = self.dPhiSigcorrsZV[i, j, 1]  # type:ignore
            outPlaneSig = self.dPhiSigcorrsZV[i, j, 2]  # type:ignore
            
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlaneSig, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlaneSig, False)**2/N_trig[0]**2)
            ax[0][1].errorbar(
                full_x,
                self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            # plot (data-fit)/fit on axRatio
            # error will be 1/fit*sqrt(data_err**2+(data/fit)**2*fit_err**2)
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlaneSig, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlaneSig, False)**2/N_trig[1]**2)
            ax[0][2].errorbar(
                full_x,
                self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlaneSig, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlaneSig, False)**2/N_trig[2]**2)
            ax[0][3].errorbar(
                full_x,
                self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlaneSig, False) + self.get_bin_contents_as_array(midPlaneSig, False) + self.get_bin_contents_as_array(outPlaneSig, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlaneSig, False)**2 + self.get_bin_errors_as_array(midPlaneSig, False)**2 + self.get_bin_errors_as_array(outPlaneSig, False)**2)/N_trig[3]**2)
            ax[0][0].errorbar(
                full_x,
                ((
                    self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0]
                    + self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1]
                    + self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2]
                )
                )/3,
                yerr=normalized_data_err/3,
                fmt="o",
                ms=2,
                label="Signal",
            )

        if withCharles and i==0:
            IPfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsIP.root")
            OOPfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsOOP.root")
            MPfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsMP.root")
            INCfile = ROOT.TFile("/home/steffanic/Projects/Thesis/python_backend/PbPbCorrelationsINC.root")
            # first get the number of triggers in the correct pt assoc and trig bins
            num_trig_hist_IP = IPfile.Get("jetH_number_of_triggers")
            num_trig_hist_MP = MPfile.Get("jetH_number_of_triggers")
            num_trig_hist_OOP = OOPfile.Get("jetH_number_of_triggers")
            num_trig_hist_INC = INCfile.Get("jetH_number_of_triggers")
            # sum the bin values between the correct pt bins
            N_trig_IP = np.sum(
                np.array(
                    [
                        num_trig_hist_IP.GetBinContent(i) for i in range(
                            num_trig_hist_IP.GetXaxis().FindBin(self.pTtrigBinEdges[i]),   
                            num_trig_hist_IP.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            N_trig_MP = np.sum(
                np.array(
                    [
                        num_trig_hist_MP.GetBinContent(i) for i in range(
                            num_trig_hist_MP.GetXaxis().FindBin(self.pTtrigBinEdges[i]),
                            num_trig_hist_MP.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            N_trig_OOP = np.sum(
                np.array(
                    [
                        num_trig_hist_OOP.GetBinContent(i) for i in range(
                            num_trig_hist_OOP.GetXaxis().FindBin(self.pTtrigBinEdges[i]),
                            num_trig_hist_OOP.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            N_trig_INC = np.sum(
                np.array(
                    [
                        num_trig_hist_INC.GetBinContent(i) for i in range(
                            num_trig_hist_INC.GetXaxis().FindBin(self.pTtrigBinEdges[i]),
                            num_trig_hist_INC.GetXaxis().FindBin(self.pTtrigBinEdges[i+1])
                        )
                    ]
                )
            )

            # Now get the deltphi histograms and scale by ntrig 
            
            IP_sig_hist = IPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            IP_bg_hist = IPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            MP_sig_hist = MPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            MP_bg_hist = MPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            OOP_sig_hist = OOPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            OOP_bg_hist = OOPfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            INC_sig_hist = INCfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_signal_dominated")
            INC_bg_hist = INCfile.Get(f"jetH_delta_phi_jetPtBiased_20_40_trackPt_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}_background_dominated")

            # scale by ntrig
            IP_sig_hist.Scale(1/N_trig_IP)
            IP_bg_hist.Scale(1/N_trig_IP)

            MP_sig_hist.Scale(1/N_trig_MP)
            MP_bg_hist.Scale(1/N_trig_MP)

            OOP_sig_hist.Scale(1/N_trig_OOP)
            OOP_bg_hist.Scale(1/N_trig_OOP)

            INC_sig_hist.Scale(1/N_trig_INC)
            INC_bg_hist.Scale(1/N_trig_INC)

            # scale by the bin width
            dphiBinWidth = IP_sig_hist.GetXaxis().GetBinWidth(1)
            IP_sig_hist.Scale(1/dphiBinWidth)
            IP_bg_hist.Scale(1/dphiBinWidth)

            MP_sig_hist.Scale(1/dphiBinWidth)
            MP_bg_hist.Scale(1/dphiBinWidth)

            OOP_sig_hist.Scale(1/dphiBinWidth)
            OOP_bg_hist.Scale(1/dphiBinWidth)

            INC_sig_hist.Scale(1/dphiBinWidth)
            INC_bg_hist.Scale(1/dphiBinWidth)

            # now get the bin contents and errors as arrays 

            IP_sig_contents = self.get_bin_contents_as_array(IP_sig_hist, False)
            IP_bg_contents = self.get_bin_contents_as_array(IP_bg_hist, False)
            IP_sig_err = self.get_bin_errors_as_array(IP_sig_hist, False)
            IP_bg_err = self.get_bin_errors_as_array(IP_bg_hist, False)

            MP_sig_contents = self.get_bin_contents_as_array(MP_sig_hist, False)
            MP_bg_contents = self.get_bin_contents_as_array(MP_bg_hist, False)
            MP_sig_err = self.get_bin_errors_as_array(MP_sig_hist, False)
            MP_bg_err = self.get_bin_errors_as_array(MP_bg_hist, False)

            OOP_sig_contents = self.get_bin_contents_as_array(OOP_sig_hist, False)
            OOP_bg_contents = self.get_bin_contents_as_array(OOP_bg_hist, False)
            OOP_sig_err = self.get_bin_errors_as_array(OOP_sig_hist, False)
            OOP_bg_err = self.get_bin_errors_as_array(OOP_bg_hist, False)

            INC_sig_contents = self.get_bin_contents_as_array(INC_sig_hist, False)
            INC_bg_contents = self.get_bin_contents_as_array(INC_bg_hist, False)
            INC_sig_err = self.get_bin_errors_as_array(INC_sig_hist, False)
            INC_bg_err = self.get_bin_errors_as_array(INC_bg_hist, False)

            # now plot the results
            ax[0][1].errorbar(
                full_x,
                IP_sig_contents,
                yerr=IP_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][1].errorbar(
                full_x,
                IP_bg_contents,
                yerr=IP_bg_err,
                fmt="o",
                label=f"Charles Background",
            )

            ax[0][2].errorbar(
                full_x,
                MP_sig_contents,
                yerr=MP_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][2].errorbar(
                full_x,
                MP_bg_contents,
                yerr=MP_bg_err,
                fmt="o",
                label=f"Charles Background",
            )

            ax[0][3].errorbar(
                full_x,
                OOP_sig_contents,
                yerr=OOP_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][3].errorbar(
                full_x,
                OOP_bg_contents,
                yerr=OOP_bg_err,
                fmt="o",
                label=f"Charles Background",
            )

            ax[0][0].errorbar(
                full_x,
                INC_sig_contents,
                yerr=INC_sig_err,
                fmt="o",
                label=f"Charles Signal",
            )
            ax[0][0].errorbar(
                full_x,
                INC_bg_contents,
                yerr=INC_bg_err,
                fmt="o",
                label=f"Charles Background",
            )


        ax[0][3].legend()
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[0]]
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[0]]
        ax[0][0].set_ylabel(r"$\frac{1}{N_trig}\frac{1}{a*\epsilon}\frac{dN}{d\Delta\phi}$")
        ax[0][1].set_title(f"In-Plane")
        ax[0][2].set_title(f"Mid-Plane")
        ax[0][3].set_title(f"Out-of-Plane")
        ax[0][0].set_title(f"Inclusive")
        # add text to the axes that says "inclusive = (in+mid+out)/3"
        #ax[0][3].text( 0.1, 0.1, "Inclusive = (In+Mid+Out)", transform = ax[0][3].transAxes,)
        fig.suptitle(
            f"RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, Chi2/NDF = {self.RPFObjsZV[i,j].chi2OverNDF}"
        )  # type:ignore
        fig.tight_layout()
        fig.savefig(
            f"{self.base_save_path}RPF{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}{'withSig' if withSignal else ''}{'withCharles' if withCharles else ''}ZV.png"
        )  # type:ignore

    @print_function_name_with_description_on_call(description="")
    def plot_PionTPCNSigmaFit(self, i,j, k, region):
        """
        Plot the fit of the pion TPC nSigma distribution
        """
        if self.analysisType in ["central", "semicentral"]:
            pionEnh = self.pionTPCnSigma_pionTOFcut[region][i,j, k]
            protonEnh = self.pionTPCnSigma_protonTOFcut[region][i,j, k]
            kaonEnh = self.pionTPCnSigma_kaonTOFcut[region][i,j, k]
            inclusive = self.pionTPCnSigmaInc[region][i,j,k]
        else:
            pionEnh = self.pionTPCnSigma_pionTOFcut[region][i,j]
            protonEnh = self.pionTPCnSigma_protonTOFcut[region][i,j]
            kaonEnh = self.pionTPCnSigma_kaonTOFcut[region][i,j]
            inclusive= self.pionTPCnSigmaInc[region][i,j]


        x_vals = [pionEnh.GetBinCenter(bin) for bin in range(1, pionEnh.GetNbinsX() + 1)]
        y_vals_pi = [pionEnh.GetBinContent(bin) for bin in range(1, pionEnh.GetNbinsX() + 1)]
        y_vals_pr = [protonEnh.GetBinContent(bin) for bin in range(1, protonEnh.GetNbinsX() + 1)]
        y_vals_ka = [kaonEnh.GetBinContent(bin) for bin in range(1, kaonEnh.GetNbinsX() + 1)]
        y_vals_inc = [inclusive.GetBinContent(bin) for bin in range(1, inclusive.GetNbinsX() + 1)]
        y_err_pi = [pionEnh.GetBinError(bin) for bin in range(1, pionEnh.GetNbinsX() + 1)]
        y_err_pr = [protonEnh.GetBinError(bin) for bin in range(1, protonEnh.GetNbinsX() + 1)]
        y_err_ka = [kaonEnh.GetBinError(bin) for bin in range(1, kaonEnh.GetNbinsX() + 1)]
        y_err_inc =  [inclusive.GetBinError(bin) for bin in range(1, inclusive.GetNbinsX() + 1)]


        fig, ax = plt.subplots(3, 2, figsize=(20, 20))
        ax[0,0].errorbar(x_vals, y_vals_pi, yerr=y_err_pi, fmt="o", label="Pion Enhanced")
        ax[0,1].errorbar(x_vals, y_vals_pr, yerr=y_err_pr, fmt="o", label="Proton Enhanced")
        ax[1,0].errorbar(x_vals, y_vals_ka, yerr=y_err_ka, fmt="o", label="Kaon Enhanced")
        ax[1,1].errorbar(x_vals, y_vals_inc, yerr=y_err_inc, fmt='o', label='Inclusive')

        if self.analysisType in ["central", "semicentral"]:
            fitter = self.PionTPCNSigmaFitObjs[region][i,j,k]
        else:
            fitter = self.PionTPCNSigmaFitObjs[region][i,j]
        mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = fitter.popt
        mup_err, mupi_err, muk_err, sigp_err, sigpi_err, sigk_err, app_err, apip_err, akp_err, appi_err, apipi_err, akpi_err, apk_err, apik_err, akk_err, apinc_err, apiinc_err, akinc_err, alphap_err, alphak_err = np.sqrt(np.diag(fitter.pcov))
        x_vals = np.linspace(-10, 10, 1000)
        y = fitter.piKpInc_generalized_fit(x_vals, *fitter.popt)
        fit_err = fitter.piKpInc_generalized_error(x_vals, *fitter.popt, fitter.pcov)
        ax[0,0].plot(x_vals, y[:1000], label=f"Total Fit")
        ax[0,1].plot(x_vals, y[1000:2000], label=f"Total Fit")
        ax[1,0].plot(x_vals, y[2000:3000], label=f"Total Fit")
        ax[1,1].plot(x_vals, y[3000:], label='Total Fit')

        ax[0,0].fill_between(x_vals, y[:1000] - fit_err[:1000], y[:1000] + fit_err[:1000], alpha=0.5)
        ax[0,1].fill_between(x_vals, y[1000:2000] - fit_err[1000:2000], y[1000:2000] + fit_err[1000:2000], alpha=0.5)
        ax[1,0].fill_between(x_vals, y[2000:3000] - fit_err[2000:3000], y[2000:3000] + fit_err[2000:3000], alpha=0.5)
        ax[1,1].fill_between(x_vals, y[3000:] - fit_err[3000:], y[3000:]+fit_err[3000:], alpha=0.5)
        #TODO: This fraction is incorrectly cited
        ax[0,0].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apipi), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apipi/(apipi+appi+akpi):.2f}+-{np.sqrt(1/(apipi+appi+akpi)**4*((appi+akpi)**2*apipi_err**2+apipi**2*(appi_err+akpi_err)**2)):.2f}")
        ax[0,0].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, appi, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {appi/(apipi+appi+akpi):.2f}+-{np.sqrt(1/(apipi+appi+akpi)**4*((apipi+akpi)**2*appi_err**2+appi**2*(apipi_err+akpi_err)**2)):.2f}")
        ax[0,0].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akpi, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akpi/(apipi+appi+akpi):.2f}+-{np.sqrt(1/(apipi+appi+akpi)**4*((apipi+appi)**2*akpi_err**2+akpi**2*(apipi_err+appi_err)**2)):.2f}")

        ax[0,1].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apip), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apip/(apip+app+akp):.2f}+-{np.sqrt(1/(apip+app+akp)**4*((app+akp)**2*apip_err**2+apip**2*(app_err+akp_err)**2)):.2f}")
        ax[0,1].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, app, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {app/(apip+app+akp):.2f}+-{np.sqrt(1/(apip+app+akp)**4*((apip+akp)**2*app_err**2+app**2*(apip_err+akp_err)**2)):.2f}")
        ax[0,1].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akp, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akp/(apip+app+akp):.2f}+-{np.sqrt(1/(apip+app+akp)**4*((apip+app)**2*akp_err**2+akp**2*(apip_err+app_err)**2)):.2f}")

        ax[1,0].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apik), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apik/(apik+apk+akk):.2f}+-{np.sqrt(1/(apik+apk+akk)**4*((apk+akk)**2*apik_err**2+apik**2*(apk_err+akk_err)**2)):.2f}")
        ax[1,0].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, apk, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {apk/(apik+apk+akk):.2f}+-{np.sqrt(1/(apik+apk+akk)**4*((apik+akk)**2*apk_err**2+apk**2*(apik_err+akk_err)**2)):.2f}")
        ax[1,0].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akk, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akk/(apik+apk+akk):.2f}+-{np.sqrt(1/(apik+apk+akk)**4*((apik+apk)**2*akk_err**2+akk**2*(apik_err+apk_err)**2)):.2f}")
        
        ax[1,1].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apiinc), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apiinc/(apiinc+apinc+akinc):.2f}+-{np.sqrt(1/(apiinc+apinc+akinc)**4*((apinc+akinc)**2*apiinc_err**2+apiinc**2*(apinc_err+akinc_err)**2)):.2f}")
        ax[1,1].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, apinc, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {apinc/(apiinc+apinc+akinc):.2f}+-{np.sqrt(1/(apiinc+apinc+akinc)**4*((apiinc+akinc)**2*apk_err**2+apinc**2*(apiinc_err+akinc_err)**2)):.2f}")
        ax[1,1].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akinc, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akinc/(apiinc+apinc+akinc):.2f}+-{np.sqrt(1/(apiinc+apinc+akinc)**4*((apiinc+apinc)**2*akinc_err**2+akinc**2*(apiinc_err+apinc_err)**2)):.2f}")

        response = self.get_response_matrix(i,j,k, fitter.popt, fitter.pcov, fitter.upiKpInc_generalized_fit, fitter.ugauss, fitter.ugeneralized_gauss, region)
        nom_vec = np.vectorize(lambda x:x.n)
        std_vec = np.vectorize(lambda x:x.s)
        response_nom = nom_vec(response)
        response_err = std_vec(response)
        sns.heatmap(response_nom, ax=ax[2,0], vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")
        sns.heatmap(response_err, ax=ax[2,1], vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis")

        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        
        fig.suptitle(
            f"TPC nSigma Fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.savefig(
            f"{self.base_save_path}{self.epString}/TPCnSigmaFit_{region}_pTtrig_{self.pTtrigBinEdges[i]}_{self.pTtrigBinEdges[i+1]}_pTassoc_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}.png"
        )  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_PionTPCNSigmaVsDphi(self, i,j, k, dphi_range: tuple=None, do_fit: bool=True):
        """
        Plot the fit of the pion TPC nSigma distribution
        """
        if self.analysisType in ["central", "semicentral"]:
            pionEnh = self.pionTPCnSigma_pionTOFcut_vs_dphi[i,j, k]
            protonEnh = self.pionTPCnSigma_protonTOFcut_vs_dphi[i,j, k]
            kaonEnh = self.pionTPCnSigma_kaonTOFcut_vs_dphi[i,j, k]
            inclusive = self.pionTPCnSigmaInc_vs_dphi[i,j,k]
        else:
            pionEnh = self.pionTPCnSigma_pionTOFcut_vs_dphi[i,j]
            protonEnh = self.pionTPCnSigma_protonTOFcut_vs_dphi[i,j]
            kaonEnh = self.pionTPCnSigma_kaonTOFcut_vs_dphi[i,j]
            inclusive= self.pionTPCnSigmaInc_vs_dphi[i,j]


        # I want to draw each of pionEnh, protonEnh, and kaonEnh and inclusive as a lego plot

        fig, ax = plt.subplots(3, 2, figsize=(20, 20))
        fig.suptitle(
            f"TPC nSigma for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c"
        )  # type:ignore

        if dphi_range is not None:
            pionEnh.GetXaxis().SetRangeUser(dphi_range[0], dphi_range[1])
            protonEnh.GetXaxis().SetRangeUser(dphi_range[0], dphi_range[1])
            kaonEnh.GetXaxis().SetRangeUser(dphi_range[0], dphi_range[1])
            inclusive.GetXaxis().SetRangeUser(dphi_range[0], dphi_range[1])

            pionEnh = pionEnh.ProjectionY(f"pionEnh_{i}_{j}_{k}")
            protonEnh = protonEnh.ProjectionY(f"protonEnh_{i}_{j}_{k}")
            kaonEnh = kaonEnh.ProjectionY(f"kaonEnh_{i}_{j}_{k}")
            inclusive = inclusive.ProjectionY(f"inclusive_{i}_{j}_{k}")

            x = np.array([pionEnh.GetBinCenter(bin_x) for bin_x in range(1, pionEnh.GetNbinsX() + 1)])
            pion_bin_contents = np.zeros(pionEnh.GetNbinsX())
            pion_bin_errors = np.zeros(pionEnh.GetNbinsX())
            proton_bin_contents = np.zeros(protonEnh.GetNbinsX())
            proton_bin_errors = np.zeros(protonEnh.GetNbinsX())
            kaon_bin_contents = np.zeros(kaonEnh.GetNbinsX())
            kaon_bin_errors = np.zeros(kaonEnh.GetNbinsX())
            inclusive_bin_contents = np.zeros(inclusive.GetNbinsX())
            inclusive_bin_errors = np.zeros(inclusive.GetNbinsX())

            for _xind in range(0, pionEnh.GetNbinsX()):
                pion_bin_contents[_xind] = pionEnh.GetBinContent(_xind+1)
                pion_bin_errors[_xind] = pionEnh.GetBinError(_xind+1)
                proton_bin_contents[_xind] = protonEnh.GetBinContent(_xind+1)
                proton_bin_errors[_xind] = protonEnh.GetBinError(_xind+1)
                kaon_bin_contents[_xind] = kaonEnh.GetBinContent(_xind+1)
                kaon_bin_errors[_xind] = kaonEnh.GetBinError(_xind+1)
                inclusive_bin_contents[_xind] = inclusive.GetBinContent(_xind+1)
                inclusive_bin_errors[_xind] = inclusive.GetBinError(_xind+1)


            ax[0,0].errorbar(
                x, pion_bin_contents, yerr=pion_bin_errors, fmt="o", label="Pion"
            )
            ax[0,1].errorbar(
                x, proton_bin_contents, yerr=proton_bin_errors, fmt="o", label="Proton"
            )
            ax[1,0].errorbar(
                x, kaon_bin_contents, yerr=kaon_bin_errors, fmt="o", label="Kaon"
            )
            ax[1,1].errorbar(
                x, inclusive_bin_contents, yerr=inclusive_bin_errors, fmt="o", label="Inclusive"
            )

            if do_fit:
                y = [pion_bin_contents, proton_bin_contents, kaon_bin_contents, inclusive_bin_contents]
                yerr = [pion_bin_errors, proton_bin_errors, kaon_bin_errors, inclusive_bin_errors]
                if self.analysisType in ["central", "semicentral"]:
                    fitter = PionTPCNSigmaFitter(p0=self.PionTPCNSigmaFitObjs[i,j,k].p0, p0_bounds=self.PionTPCNSigmaFitObjs[i,j,k].p0_bounds, w_inclusive=True, generalized=True)
                else:
                    fitter = PionTPCNSigmaFitter(p0=self.PionTPCNSigmaFitObjs[i,j].p0, p0_bounds=self.PionTPCNSigmaFitObjs[i,j].p0_bounds, w_inclusive=True, generalized=True)

                _, _, chi2 = fitter.fit(x, y, yerr)
                fig.suptitle(f"TPC nSigma for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c; Chi2 {chi2}")

                mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = fitter.popt
                mup_err, mupi_err, muk_err, sigp_err, sigpi_err, sigk_err, app_err, apip_err, akp_err, appi_err, apipi_err, akpi_err, apk_err, apik_err, akk_err, apinc_err, apiinc_err, akinc_err, alphap_err, alphak_err = np.sqrt(np.diag(fitter.pcov))
                x_vals = np.linspace(-10, 10, 1000)
                y = fitter.piKpInc_generalized_fit(x_vals, *fitter.popt)
                fit_err = fitter.piKpInc_generalized_error(x_vals, *fitter.popt, fitter.pcov)
                ax[0,0].plot(x_vals, y[:1000], label=f"Total Fit")
                ax[0,1].plot(x_vals, y[1000:2000], label=f"Total Fit")
                ax[1,0].plot(x_vals, y[2000:3000], label=f"Total Fit")
                ax[1,1].plot(x_vals, y[3000:], label='Total Fit')

                ax[0,0].fill_between(x_vals, y[:1000] - fit_err[:1000], y[:1000] + fit_err[:1000], alpha=0.5)
                ax[0,1].fill_between(x_vals, y[1000:2000] - fit_err[1000:2000], y[1000:2000] + fit_err[1000:2000], alpha=0.5)
                ax[1,0].fill_between(x_vals, y[2000:3000] - fit_err[2000:3000], y[2000:3000] + fit_err[2000:3000], alpha=0.5)
                ax[1,1].fill_between(x_vals, y[3000:] - fit_err[3000:], y[3000:]+fit_err[3000:], alpha=0.5)
                #TODO: This fraction is incorrectly cited
                ax[0,0].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apipi), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apipi/(apipi+appi+akpi):.2f}+-{np.sqrt(1/(apipi+appi+akpi)**4*((appi+akpi)**2*apipi_err**2+apipi**2*(appi_err+akpi_err)**2)):.2f}")
                ax[0,0].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, appi, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {appi/(apipi+appi+akpi):.2f}+-{np.sqrt(1/(apipi+appi+akpi)**4*((apipi+akpi)**2*appi_err**2+appi**2*(apipi_err+akpi_err)**2)):.2f}")
                ax[0,0].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akpi, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akpi/(apipi+appi+akpi):.2f}+-{np.sqrt(1/(apipi+appi+akpi)**4*((apipi+appi)**2*akpi_err**2+akpi**2*(apipi_err+appi_err)**2)):.2f}")

                ax[0,1].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apip), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apip/(apip+app+akp):.2f}+-{np.sqrt(1/(apip+app+akp)**4*((app+akp)**2*apip_err**2+apip**2*(app_err+akp_err)**2)):.2f}")
                ax[0,1].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, app, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {app/(apip+app+akp):.2f}+-{np.sqrt(1/(apip+app+akp)**4*((apip+akp)**2*app_err**2+app**2*(apip_err+akp_err)**2)):.2f}")
                ax[0,1].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akp, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akp/(apip+app+akp):.2f}+-{np.sqrt(1/(apip+app+akp)**4*((apip+app)**2*akp_err**2+akp**2*(apip_err+app_err)**2)):.2f}")

                ax[1,0].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apik), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apik/(apik+apk+akk):.2f}+-{np.sqrt(1/(apik+apk+akk)**4*((apk+akk)**2*apik_err**2+apik**2*(apk_err+akk_err)**2)):.2f}")
                ax[1,0].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, apk, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {apk/(apik+apk+akk):.2f}+-{np.sqrt(1/(apik+apk+akk)**4*((apik+akk)**2*apk_err**2+apk**2*(apik_err+akk_err)**2)):.2f}")
                ax[1,0].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akk, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akk/(apik+apk+akk):.2f}+-{np.sqrt(1/(apik+apk+akk)**4*((apik+apk)**2*akk_err**2+akk**2*(apik_err+apk_err)**2)):.2f}")
                
                ax[1,1].plot(x_vals, fitter.gauss(x_vals, mupi, sigpi, apiinc), label=f"Pion, Mean: {mupi:.2f}, Sigma: {sigpi:.2f}, Fraction: {apiinc/(apiinc+apinc+akinc):.2f}+-{np.sqrt(1/(apiinc+apinc+akinc)**4*((apinc+akinc)**2*apiinc_err**2+apiinc**2*(apinc_err+akinc_err)**2)):.2f}")
                ax[1,1].plot(x_vals, fitter.generalized_gauss(x_vals, mup, sigp, apinc, alphap), label=f"Proton, Mean: {mup:.2f}, Sigma: {sigp:.2f}, Fraction: {apinc/(apiinc+apinc+akinc):.2f}+-{np.sqrt(1/(apiinc+apinc+akinc)**4*((apiinc+akinc)**2*apk_err**2+apinc**2*(apiinc_err+akinc_err)**2)):.2f}")
                ax[1,1].plot(x_vals, fitter.generalized_gauss(x_vals, muk, sigk, akinc, alphak), label=f"Kaon, Mean: {muk:.2f}, Sigma: {sigk:.2f}, Fraction: {akinc/(apiinc+apinc+akinc):.2f}+-{np.sqrt(1/(apiinc+apinc+akinc)**4*((apiinc+apinc)**2*akinc_err**2+akinc**2*(apiinc_err+apinc_err)**2)):.2f}")

                # Response matrix 
                int_x = np.linspace(-10, 10, 1000)
                int_y = fitter.piKpInc_generalized_fit(int_x, mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak)


                pionEnhNorm  = np.trapz(int_y[:1000], int_x)
                protonEnhNorm = np.trapz(int_y[1000:2000], int_x)
                kaonEnhNorm  = np.trapz(int_y[2000:3000], int_x)
                inclusiveNorm= np.trapz(int_y[3000:], int_x)
                mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc, alphap, alphak = uncertainties.correlated_values(fitter.popt, fitter.pcov)
                gpp = np.trapz(fitter.ugeneralized_gauss(int_x, mup, sigp, app, alphap), int_x)/protonEnhNorm
                gppi = np.trapz(fitter.ugauss(int_x, mupi, sigpi, apip), int_x)/protonEnhNorm
                gpk = np.trapz(fitter.ugeneralized_gauss(int_x, muk, sigk, akp, alphak), int_x)/protonEnhNorm

                gpip = np.trapz(fitter.ugeneralized_gauss(int_x, mup, sigp, appi, alphap), int_x)/pionEnhNorm
                gpipi = np.trapz(fitter.ugauss(int_x, mupi, sigpi, apipi), int_x)/pionEnhNorm
                gpik = np.trapz(fitter.ugeneralized_gauss(int_x, muk, sigk, akpi, alphak), int_x)/pionEnhNorm

                gkp = np.trapz(fitter.ugeneralized_gauss(int_x, mup, sigp, apk, alphap), int_x)/kaonEnhNorm
                gkpi = np.trapz(fitter.ugauss(int_x, mupi, sigpi, apik), int_x)/kaonEnhNorm
                gkk = np.trapz(fitter.ugeneralized_gauss(int_x, muk, sigk, akk, alphak), int_x)/kaonEnhNorm

                gincp = np.trapz(fitter.ugeneralized_gauss(int_x, mup, sigp, apinc, alphap), int_x)/inclusiveNorm
                gincpi = np.trapz(fitter.ugauss(int_x, mupi, sigpi, apiinc), int_x)/inclusiveNorm
                ginck = np.trapz(fitter.ugeneralized_gauss(int_x, muk, sigk, akinc, alphak), int_x)/inclusiveNorm
                

                debug_logger.debug(f"{[[gpipi, gppi, gkpi], [gpip, gpp, gkp], [gpik, gpk, gkk]]=}")
                debug_logger.debug(f"{[gincpi, gincp, ginck]=}")

                if self.analysisType in ['central', 'semicentral']:
                    A = np.array([
                        [
                            (gpipi/gincpi)*self.N_assoc_for_species['pion'][i,j,k], (gpip/gincp)*self.N_assoc_for_species['pion'][i,j,k], (gpik/ginck)*self.N_assoc_for_species['pion'][i,j,k]
                        ],
                        [
                            (gppi/gincpi)*self.N_assoc_for_species['proton'][i,j,k], (gpp/gincp)*self.N_assoc_for_species['proton'][i,j,k], (gpk/ginck)*self.N_assoc_for_species['proton'][i,j,k]
                        ],
                        [
                            (gkpi/gincpi)*self.N_assoc_for_species['kaon'][i,j,k], (gkp/gincp)*self.N_assoc_for_species['kaon'][i,j,k], (gkk/ginck)*self.N_assoc_for_species['kaon'][i,j,k]
                        ]
                    ])/self.N_assoc[i,j,k]
                else:
                    A = np.array([
                        [
                            (gpipi/gincpi)*self.N_assoc_for_species['pion'][i,j], (gpip/gincp)*self.N_assoc_for_species['pion'][i,j], (gpik/ginck)*self.N_assoc_for_species['pion'][i,j]
                        ],
                        [
                            (gppi/gincpi)*self.N_assoc_for_species['proton'][i,j], (gpp/gincp)*self.N_assoc_for_species['proton'][i,j], (gpk/ginck)*self.N_assoc_for_species['proton'][i,j]
                        ],
                        [
                            (gkpi/gincpi)*self.N_assoc_for_species['kaon'][i,j], (gkp/gincp)*self.N_assoc_for_species['kaon'][i,j], (gkk/ginck)*self.N_assoc_for_species['kaon'][i,j]
                        ]
                    ])/self.N_assoc[i,j]
                nom = lambda x: x.n
                std = lambda x: x.s
                nom_vec = np.vectorize(nom)
                std_vec = np.vectorize(std)
                A_nom = nom_vec(A)
                A_std = std_vec(A)
                sns.heatmap(A_nom, annot=True, cmap='viridis', fmt='.2f', ax = ax[2,0], vmin=0, vmax=1)
                sns.heatmap(A_std, annot=True, cmap='viridis', fmt='.2f', ax = ax[2,1], vmin=0, vmax=1)

            ax[0,0].legend()
            ax[0,1].legend()
            ax[1,0].legend()
            ax[1,1].legend()

            ax[0,0].set_title("Pion")
            ax[0,1].set_title("Proton")
            ax[1,0].set_title("Kaon")
            ax[1,1].set_title("Inclusive")

            ax[0,0].set_xlabel("TPC nSigma")
            ax[0,1].set_xlabel("TPC nSigma")
            ax[1,0].set_xlabel("TPC nSigma")
            ax[1,1].set_xlabel("TPC nSigma")

            fig.tight_layout()
            fig.savefig(
                f"{self.base_save_path}{self.epString}/TPCnSigma_dphi_{dphi_range[0]}_{dphi_range[1]}_pTtrig_{self.pTtrigBinEdges[i]}_{self.pTtrigBinEdges[i+1]}_pTassoc_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore


        else:

            pion_hist_array = np.zeros((pionEnh.GetNbinsX(), pionEnh.GetNbinsY()))
            proton_hist_array = np.zeros((protonEnh.GetNbinsX(), protonEnh.GetNbinsY()))
            kaon_hist_array = np.zeros((kaonEnh.GetNbinsX(), kaonEnh.GetNbinsY()))
            inclusive_hist_array = np.zeros((inclusive.GetNbinsX(), inclusive.GetNbinsY()))


            for _xind in range(0, pionEnh.GetNbinsX()):
                for _yind in range(0, pionEnh.GetNbinsY()):
                    pion_hist_array[_xind, _yind] = pionEnh.GetBinContent(_xind+1, _yind+1)
                    proton_hist_array[_xind, _yind] = protonEnh.GetBinContent(_xind+1, _yind+1)
                    kaon_hist_array[_xind, _yind] = kaonEnh.GetBinContent(_xind+1, _yind+1)
                    inclusive_hist_array[_xind, _yind] = inclusive.GetBinContent(_xind+1, _yind+1)

            ax[0,0].imshow(pion_hist_array, origin="lower", extent=[-10, 10, -np.pi/2, 3*np.pi/2], aspect="auto")
            ax[0,0].set_title("Pion")
            ax[0,0].set_xlabel(r"$\Delta\phi$")

            ax[0,1].imshow(proton_hist_array, origin="lower", extent=[-10, 10, -np.pi/2, 3*np.pi/2], aspect="auto")
            ax[0,1].set_title("Proton")
            ax[0,1].set_xlabel(r"$\Delta\phi$")

            ax[1,0].imshow(kaon_hist_array, origin="lower", extent=[-10, 10, -np.pi/2, 3*np.pi/2], aspect="auto")
            ax[1,0].set_title("Kaon")
            ax[1,0].set_xlabel(r"$\Delta\phi$")

            ax[1,1].imshow(inclusive_hist_array, origin="lower", extent=[-10, 10, -np.pi/2, 3*np.pi/2], aspect="auto")
            ax[1,1].set_title("Inclusive")
            ax[1,1].set_xlabel(r"$\Delta\phi$")

            fig.tight_layout()
            fig.savefig(
                f"{self.base_save_path}{self.epString}/TPCnSigmaVsDphi_pTtrig_{self.pTtrigBinEdges[i]}_{self.pTtrigBinEdges[i+1]}_pTassoc_{self.pTassocBinEdges[j]}_{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore



        

    @print_function_name_with_description_on_call(description="")
    def plot_RPF_for_species(self, i, j, species, withSignal=False):
        inPlane = self.dPhiBGcorrsForTrueSpecies[species][i, j, 0]  # type:ignore
        midPlane = self.dPhiBGcorrsForTrueSpecies[species][i, j, 1]  # type:ignore
        outPlane = self.dPhiBGcorrsForTrueSpecies[species][i, j, 2]  # type:ignore
        fit_y = []
        fit_y_err = []
        full_x = self.get_bin_centers_as_array(inPlane, forFitting=False)
        for _xind in range(0, len(full_x)):
            fit_y.append(
                self.RPFObjsForTrueSpecies[species][i, j].simultaneous_fit(full_x[_xind], *self.RPFObjsForTrueSpecies[species][i, j].popt)
            )  # type:ignore
            fit_y_err.append(
                self.RPFObjsForTrueSpecies[species][i, j].simultaneous_fit_err(
                    full_x[_xind], full_x[1] - full_x[0], *self.RPFObjsForTrueSpecies[species][i, j].popt
                )
            )  # type:ignore
        fit_y = np.array(fit_y, dtype=np.float64)
        fit_y_err = np.array(fit_y_err, dtype=np.float64)

        fig, ax = plt.subplots(
            2,
            4,
            figsize=(20, 8),
            sharey="row",
            sharex=True,
            gridspec_kw={"height_ratios": [0.8, 0.2]},
        )
        # remove margins between plots
        fig.subplots_adjust(wspace=0, hspace=0)

        N_trig = self.N_trigs[i] # type:ignore

        #++++++++++++++++++IN-PLANE+++++++++++++++++++
        ax[0][1].plot(full_x, fit_y[:, 0]/N_trig[0], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,0]**2/N_trig[0]**5 + fit_y_err[:,0]**2/N_trig[0]**2)
        ax[0][1].fill_between(
            full_x,
            fit_y[:, 0]/N_trig[0] - normalized_err,
            fit_y[:, 0]/N_trig[0] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlane, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlane, False)**2/N_trig[0]**2)
        ax[0][1].errorbar(
            full_x,
            self.get_bin_contents_as_array(inPlane, False)/N_trig[0],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (self.get_bin_contents_as_array(inPlane, False) - fit_y[:, 0]) / fit_y[
            :, 0
        ]
        ratErr = (
            1
            / fit_y[:, 0]
            * np.sqrt(
                self.get_bin_errors_as_array(inPlane, False) ** 2
                + (self.get_bin_contents_as_array(inPlane, False) / fit_y[:, 0]) ** 2
                * fit_y_err[:, 0] ** 2
            )
        )
        ax[1][1].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        
        #++++++++++++++++++MID-PLANE+++++++++++++++++++
        ax[0][2].plot(full_x, fit_y[:, 1]/N_trig[1], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,1]**2/N_trig[1]**5 + fit_y_err[:,1]**2/N_trig[1]**2)
        ax[0][2].fill_between(
            full_x,
            fit_y[:, 1]/N_trig[1] - normalized_err,
            fit_y[:, 1]/N_trig[1] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlane, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlane, False)**2/N_trig[1]**2)
        ax[0][2].errorbar(
            full_x,
            self.get_bin_contents_as_array(midPlane, False)/N_trig[1],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(midPlane, False) - fit_y[:, 1]
        ) / fit_y[:, 1]
        ratErr = (
            1
            / fit_y[:, 1]
            * np.sqrt(
                self.get_bin_errors_as_array(midPlane, False) ** 2
                + (self.get_bin_contents_as_array(midPlane, False) / fit_y[:, 1]) ** 2
                * fit_y_err[:, 1] ** 2
            )
        )
        ax[1][2].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][3].plot(full_x, fit_y[:, 2]/N_trig[2], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,2]**2/N_trig[2]**5 + fit_y_err[:,2]**2/N_trig[2]**2)
        ax[0][3].fill_between(
            full_x,
            fit_y[:, 2]/N_trig[2] - normalized_err,
            fit_y[:, 2]/N_trig[2] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlane, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlane, False)**2/N_trig[2]**2)
        ax[0][3].errorbar(
            full_x,
            self.get_bin_contents_as_array(outPlane, False)/N_trig[2],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(outPlane, False) - fit_y[:, 2]
        ) / fit_y[:, 2]
        ratErr = (
            1
            / fit_y[:, 2]
            * np.sqrt(
                self.get_bin_errors_as_array(outPlane, False) ** 2
                + (self.get_bin_contents_as_array(outPlane, False) / fit_y[:, 2]) ** 2
                * fit_y_err[:, 2] ** 2
            )
        )
        ax[1][3].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][0].plot(full_x, (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3, label="RPF Fit")
        normalized_err = np.sqrt(np.sum(fit_y, axis=1)**2/N_trig[3]**5 + np.sum(fit_y_err**2, axis=1)/N_trig[3]**2)
        ax[0][0].fill_between(
            full_x,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 - normalized_err/3,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 + normalized_err/3,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlane, False) + self.get_bin_contents_as_array(midPlane, False) + self.get_bin_contents_as_array(outPlane, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlane, False)**2 + self.get_bin_errors_as_array(midPlane, False)**2 + self.get_bin_errors_as_array(outPlane, False)**2)/N_trig[3]**2)
        ax[0][0].errorbar(
            full_x,
            ((
                self.get_bin_contents_as_array(inPlane, False)/N_trig[0]
                + self.get_bin_contents_as_array(midPlane, False)/N_trig[1]
                + self.get_bin_contents_as_array(outPlane, False)/N_trig[2]
            )
            )/3,
            yerr=normalized_data_err/3,
            fmt="o",
            ms=2,
            label="Background",
        )
        
        ratVal = (
            (
                self.get_bin_contents_as_array(inPlane, False)
                - fit_y[:, 0]
            ) / fit_y[:, 0]
                + 
            (
            self.get_bin_contents_as_array(midPlane, False)
                - fit_y[:, 1]
            ) / fit_y[:, 1]
                + 
            (
            self.get_bin_contents_as_array(outPlane, False)
                - fit_y[:, 2]
            ) / fit_y[:, 2]
            ) / 3

        ratErr = 1/3 * np.sqrt(
            self.get_bin_contents_as_array(inPlane, False)**2/fit_y[:, 0]**4 * fit_y_err[:, 0]**2 
            +
            self.get_bin_contents_as_array(midPlane, False)**2/fit_y[:, 1]**4 * fit_y_err[:, 1]**2
            +
            self.get_bin_contents_as_array(outPlane, False)**2/fit_y[:, 2]**4 * fit_y_err[:, 2]**2
            + 
            1/fit_y[:, 0]**2 * self.get_bin_errors_as_array(inPlane, False)**2
            +
            1/fit_y[:, 1]**2 * self.get_bin_errors_as_array(midPlane, False)**2
            +
            1/fit_y[:, 2]**2 * self.get_bin_errors_as_array(outPlane, False)**2 
         )
        
        ax[1][0].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        if withSignal:
            inPlaneSig = self.dPhiSigcorrsForTrueSpecies[species][i, j, 0]  # type:ignore
            midPlaneSig = self.dPhiSigcorrsForTrueSpecies[species][i, j, 1]  # type:ignore
            outPlaneSig = self.dPhiSigcorrsForTrueSpecies[species][i, j, 2]  # type:ignore
            
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlaneSig, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlaneSig, False)**2/N_trig[0]**2)
            ax[0][1].errorbar(
                full_x,
                self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            # plot (data-fit)/fit on axRatio
            # error will be 1/fit*sqrt(data_err**2+(data/fit)**2*fit_err**2)
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlaneSig, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlaneSig, False)**2/N_trig[1]**2)
            ax[0][2].errorbar(
                full_x,
                self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlaneSig, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlaneSig, False)**2/N_trig[2]**2)
            ax[0][3].errorbar(
                full_x,
                self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlaneSig, False) + self.get_bin_contents_as_array(midPlaneSig, False) + self.get_bin_contents_as_array(outPlaneSig, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlaneSig, False)**2 + self.get_bin_errors_as_array(midPlaneSig, False)**2 + self.get_bin_errors_as_array(outPlaneSig, False)**2)/N_trig[3]**2)
            ax[0][0].errorbar(
                full_x,
                ((
                    self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0]
                    + self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1]
                    + self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2]
                )
                )/3,
                yerr=normalized_data_err/3,
                fmt="o",
                ms=2,
                label="Signal",
            )

        ax[0][3].legend()
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[0]]
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[0]]
        ax[0][0].set_ylabel(r"$\frac{1}{N_trig}\frac{1}{a*\epsilon}\frac{dN}{d\Delta\phi}$")
        ax[0][1].set_title(f"In-Plane")
        ax[0][2].set_title(f"Mid-Plane")
        ax[0][3].set_title(f"Out-of-Plane")
        ax[0][0].set_title(f"Inclusive")

#         if self.analysisType in ['central', 'semicentral']:
#             fit_params_all = [self.PionTPCNSigmaFitObjs[i,j,k].popt for k in range(4)]
#             fit_errs_all =[self.PionTPCNSigmaFitObjs[i,j,k].pcov for k in range(4)]
#             for ev_i, (fit_params, fit_errs) in enumerate(zip(fit_params_all, fit_errs_all)):
#                 mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc = uncertainties.correlated_values(fit_params, fit_errs)
# 
#                 protonEnhNorm = -(np.pi/2)**.5*(
#                     akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                     app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                     apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                     )
#                 pionEnhNorm = -(np.pi/2)**.5*(
#                     akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                     appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                     apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                     )
#                 kaonEnhNorm = -(np.pi/2)**.5*(
#                     apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                     apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                     akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                     )
#                 
#                 fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#                 fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#                 fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#                 fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#                 fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#                 fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#                 fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#                 fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#                 fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#                 
#                 determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
# 
# 
#                 inv_mat = unp.ulinalg.inv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
# 
#                 row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#                 
#                 inv_mat = inv_mat/row_sums[:,np.newaxis]
#                     
#                     
#                 pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#                 proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#                 kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"
#                 
#                 comp_str = pion_comp_str if species=="pion" else proton_comp_str if species=="proton" else kaon_comp_str
#                 
# 
#                 #ax[0][(ev_i+1)%4].text( 0.9, 0.85, f"Composition: {comp_str}", transform = ax[0][(ev_i+1)%4].transAxes, ha='right')
#         else:
#             fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
#             mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk = fit_params
# 
#             protonEnhNorm = -(np.pi/2)**.5*(
#                 akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                 app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                 apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                 )
#             pionEnhNorm = -(np.pi/2)**.5*(
#                 akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                 appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                 apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                 )
#             kaonEnhNorm = -(np.pi/2)**.5*(
#                 apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                 apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                 akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                 )
#             
#             fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#             fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#             fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#             fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#             fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#             fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#             fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#             fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#             fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#             
#             determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
#             inv_mat = unp.ulinalg.inv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
# 
#             row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#             inv_mat = inv_mat/row_sums[:,np.newaxis]
#                     
#                     
#             pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#             proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#             kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"
#             
#             comp_str = pion_comp_str if species=="pion" else proton_comp_str if species=="proton" else kaon_comp_str

            #ax[0][0].text( 0.9, 0.85, f"Composition: {comp_str}", transform = ax[0][0].transAxes, ha='right')
        # add text to the axes that says "inclusive = (in+mid+out)/3"
        #ax[0][3].text( 0.1, 0.1, "Inclusive = (In+Mid+Out)", transform = ax[0][3].transAxes,)
        fig.suptitle(
            f"RPF fit for {species}s for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, Chi2/NDF = {self.RPFObjs[i,j].chi2OverNDF}"
        )  # type:ignore
        fig.tight_layout()
        fig.savefig(
            f"{self.base_save_path}RPF{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}{'withSig' if withSignal else ''}{species}.png"
        )  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_RPF_for_species_in_z_vertex_bins(self, i, j, species, withSignal=False):
        inPlane = self.dPhiBGcorrsForTrueSpeciesZV[species][i, j, 0]  # type:ignore
        midPlane = self.dPhiBGcorrsForTrueSpeciesZV[species][i, j, 1]  # type:ignore
        outPlane = self.dPhiBGcorrsForTrueSpeciesZV[species][i, j, 2]  # type:ignore
        fit_y = []
        fit_y_err = []
        full_x = self.get_bin_centers_as_array(inPlane, forFitting=False)
        for _xind in range(0, len(full_x)):
            fit_y.append(
                self.RPFObjsForTrueSpeciesZV[species][i, j].simultaneous_fit(full_x[_xind], *self.RPFObjsForTrueSpeciesZV[species][i, j].popt)
            )  # type:ignore
            fit_y_err.append(
                self.RPFObjsForTrueSpeciesZV[species][i, j].simultaneous_fit_err(
                    full_x[_xind], full_x[1] - full_x[0], *self.RPFObjsForTrueSpeciesZV[species][i, j].popt
                )
            )  # type:ignore
        fit_y = np.array(fit_y, dtype=np.float64)
        fit_y_err = np.array(fit_y_err, dtype=np.float64)

        fig, ax = plt.subplots(
            2,
            4,
            figsize=(20, 8),
            sharey="row",
            sharex=True,
            gridspec_kw={"height_ratios": [0.8, 0.2]},
        )
        # remove margins between plots
        fig.subplots_adjust(wspace=0, hspace=0)

        N_trig = self.N_trigs[i] # type:ignore

        #++++++++++++++++++IN-PLANE+++++++++++++++++++
        ax[0][1].plot(full_x, fit_y[:, 0]/N_trig[0], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,0]**2/N_trig[0]**5 + fit_y_err[:,0]**2/N_trig[0]**2)
        ax[0][1].fill_between(
            full_x,
            fit_y[:, 0]/N_trig[0] - normalized_err,
            fit_y[:, 0]/N_trig[0] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlane, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlane, False)**2/N_trig[0]**2)
        ax[0][1].errorbar(
            full_x,
            self.get_bin_contents_as_array(inPlane, False)/N_trig[0],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (self.get_bin_contents_as_array(inPlane, False) - fit_y[:, 0]) / fit_y[
            :, 0
        ]
        ratErr = (
            1
            / fit_y[:, 0]
            * np.sqrt(
                self.get_bin_errors_as_array(inPlane, False) ** 2
                + (self.get_bin_contents_as_array(inPlane, False) / fit_y[:, 0]) ** 2
                * fit_y_err[:, 0] ** 2
            )
        )
        ax[1][1].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        
        #++++++++++++++++++MID-PLANE+++++++++++++++++++
        ax[0][2].plot(full_x, fit_y[:, 1]/N_trig[1], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,1]**2/N_trig[1]**5 + fit_y_err[:,1]**2/N_trig[1]**2)
        ax[0][2].fill_between(
            full_x,
            fit_y[:, 1]/N_trig[1] - normalized_err,
            fit_y[:, 1]/N_trig[1] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlane, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlane, False)**2/N_trig[1]**2)
        ax[0][2].errorbar(
            full_x,
            self.get_bin_contents_as_array(midPlane, False)/N_trig[1],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(midPlane, False) - fit_y[:, 1]
        ) / fit_y[:, 1]
        ratErr = (
            1
            / fit_y[:, 1]
            * np.sqrt(
                self.get_bin_errors_as_array(midPlane, False) ** 2
                + (self.get_bin_contents_as_array(midPlane, False) / fit_y[:, 1]) ** 2
                * fit_y_err[:, 1] ** 2
            )
        )
        ax[1][2].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][3].plot(full_x, fit_y[:, 2]/N_trig[2], label="RPF Fit")
        normalized_err = np.sqrt(fit_y[:,2]**2/N_trig[2]**5 + fit_y_err[:,2]**2/N_trig[2]**2)
        ax[0][3].fill_between(
            full_x,
            fit_y[:, 2]/N_trig[2] - normalized_err,
            fit_y[:, 2]/N_trig[2] + normalized_err,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlane, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlane, False)**2/N_trig[2]**2)
        ax[0][3].errorbar(
            full_x,
            self.get_bin_contents_as_array(outPlane, False)/N_trig[2],
            yerr=normalized_data_err,
            fmt="o",
            ms=2,
            label="Background",
        )
        ratVal = (
            self.get_bin_contents_as_array(outPlane, False) - fit_y[:, 2]
        ) / fit_y[:, 2]
        ratErr = (
            1
            / fit_y[:, 2]
            * np.sqrt(
                self.get_bin_errors_as_array(outPlane, False) ** 2
                + (self.get_bin_contents_as_array(outPlane, False) / fit_y[:, 2]) ** 2
                * fit_y_err[:, 2] ** 2
            )
        )
        ax[1][3].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)
        
        ax[0][0].plot(full_x, (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3, label="RPF Fit")
        normalized_err = np.sqrt(np.sum(fit_y, axis=1)**2/N_trig[3]**5 + np.sum(fit_y_err**2, axis=1)/N_trig[3]**2)
        ax[0][0].fill_between(
            full_x,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 - normalized_err/3,
            (fit_y[:,0]/N_trig[0]+fit_y[:,1]/N_trig[1]+fit_y[:,2]/N_trig[2])/3 + normalized_err/3,
            alpha=0.3,
        )
        normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlane, False) + self.get_bin_contents_as_array(midPlane, False) + self.get_bin_contents_as_array(outPlane, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlane, False)**2 + self.get_bin_errors_as_array(midPlane, False)**2 + self.get_bin_errors_as_array(outPlane, False)**2)/N_trig[3]**2)
        ax[0][0].errorbar(
            full_x,
            ((
                self.get_bin_contents_as_array(inPlane, False)/N_trig[0]
                + self.get_bin_contents_as_array(midPlane, False)/N_trig[1]
                + self.get_bin_contents_as_array(outPlane, False)/N_trig[2]
            )
            )/3,
            yerr=normalized_data_err/3,
            fmt="o",
            ms=2,
            label="Background",
        )
        
        ratVal = (
            (
                self.get_bin_contents_as_array(inPlane, False)
                - fit_y[:, 0]
            ) / fit_y[:, 0]
                + 
            (
            self.get_bin_contents_as_array(midPlane, False)
                - fit_y[:, 1]
            ) / fit_y[:, 1]
                + 
            (
            self.get_bin_contents_as_array(outPlane, False)
                - fit_y[:, 2]
            ) / fit_y[:, 2]
            ) / 3

        ratErr = 1/3 * np.sqrt(
            self.get_bin_contents_as_array(inPlane, False)**2/fit_y[:, 0]**4 * fit_y_err[:, 0]**2 
            +
            self.get_bin_contents_as_array(midPlane, False)**2/fit_y[:, 1]**4 * fit_y_err[:, 1]**2
            +
            self.get_bin_contents_as_array(outPlane, False)**2/fit_y[:, 2]**4 * fit_y_err[:, 2]**2
            + 
            1/fit_y[:, 0]**2 * self.get_bin_errors_as_array(inPlane, False)**2
            +
            1/fit_y[:, 1]**2 * self.get_bin_errors_as_array(midPlane, False)**2
            +
            1/fit_y[:, 2]**2 * self.get_bin_errors_as_array(outPlane, False)**2 
         )
        
        ax[1][0].fill_between(full_x, ratVal - ratErr, ratVal + ratErr, alpha=0.3)

        if withSignal:
            inPlaneSig = self.dPhiSigcorrsForTrueSpeciesZV[species][i, j, 0]  # type:ignore
            midPlaneSig = self.dPhiSigcorrsForTrueSpeciesZV[species][i, j, 1]  # type:ignore
            outPlaneSig = self.dPhiSigcorrsForTrueSpeciesZV[species][i, j, 2]  # type:ignore
            
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(inPlaneSig, False)**2/N_trig[0]**5 + self.get_bin_errors_as_array(inPlaneSig, False)**2/N_trig[0]**2)
            ax[0][1].errorbar(
                full_x,
                self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            # plot (data-fit)/fit on axRatio
            # error will be 1/fit*sqrt(data_err**2+(data/fit)**2*fit_err**2)
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(midPlaneSig, False)**2/N_trig[1]**5 + self.get_bin_errors_as_array(midPlaneSig, False)**2/N_trig[1]**2)
            ax[0][2].errorbar(
                full_x,
                self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt(self.get_bin_contents_as_array(outPlaneSig, False)**2/N_trig[2]**5 + self.get_bin_errors_as_array(outPlaneSig, False)**2/N_trig[2]**2)
            ax[0][3].errorbar(
                full_x,
                self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2],
                yerr=normalized_data_err,
                fmt="o",
                ms=2,
                label="Signal",
            )
            normalized_data_err = np.sqrt((self.get_bin_contents_as_array(inPlaneSig, False) + self.get_bin_contents_as_array(midPlaneSig, False) + self.get_bin_contents_as_array(outPlaneSig, False))**2/N_trig[3]**5 + (self.get_bin_errors_as_array(inPlaneSig, False)**2 + self.get_bin_errors_as_array(midPlaneSig, False)**2 + self.get_bin_errors_as_array(outPlaneSig, False)**2)/N_trig[3]**2)
            ax[0][0].errorbar(
                full_x,
                ((
                    self.get_bin_contents_as_array(inPlaneSig, False)/N_trig[0]
                    + self.get_bin_contents_as_array(midPlaneSig, False)/N_trig[1]
                    + self.get_bin_contents_as_array(outPlaneSig, False)/N_trig[2]
                )
                )/3,
                yerr=normalized_data_err/3,
                fmt="o",
                ms=2,
                label="Signal",
            )

        ax[0][3].legend()
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[0]]
        [x.set_xlabel(r"$\Delta\phi$") for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[1]]
        [x.autoscale(enable=True, axis="y", tight=True) for x in ax[0]]
        ax[0][0].set_ylabel(r"$\frac{1}{N_trig}\frac{1}{a*\epsilon}\frac{dN}{d\Delta\phi}$")
        ax[0][1].set_title(f"In-Plane")
        ax[0][2].set_title(f"Mid-Plane")
        ax[0][3].set_title(f"Out-of-Plane")
        ax[0][0].set_title(f"Inclusive")

#         if self.analysisType in ['central', 'semicentral']:
#             fit_params_all = [self.PionTPCNSigmaFitObjs[i,j,k].popt for k in range(4)]
#             fit_errs_all =[self.PionTPCNSigmaFitObjs[i,j,k].pcov for k in range(4)]
#             for ev_i, (fit_params, fit_errs) in enumerate(zip(fit_params_all, fit_errs_all)):
#                 mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk, apinc, apiinc, akinc = uncertainties.correlated_values(fit_params, fit_errs)
# 
#                 protonEnhNorm = -(np.pi/2)**.5*(
#                     akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                     app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                     apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                     )
#                 pionEnhNorm = -(np.pi/2)**.5*(
#                     akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                     appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                     apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                     )
#                 kaonEnhNorm = -(np.pi/2)**.5*(
#                     apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                     apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                     akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                     )
#                 
#                 fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#                 fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#                 fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#                 fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#                 fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#                 fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#                 fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#                 fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#                 fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#                 
#                 determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
# 
# 
#                 inv_mat = unp.ulinalg.inv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
# 
#                 row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#                 
#                 inv_mat = inv_mat/row_sums[:,np.newaxis]
#                     
#                     
#                 pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#                 proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#                 kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"
#                 
#                 comp_str = pion_comp_str if species=="pion" else proton_comp_str if species=="proton" else kaon_comp_str
#                 
# 
#                 #ax[0][(ev_i+1)%4].text( 0.9, 0.85, f"Composition: {comp_str}", transform = ax[0][(ev_i+1)%4].transAxes, ha='right')
#         else:
#             fit_params = self.PionTPCNSigmaFitObjs[i,j].popt
#             mup, mupi, muk, sigp, sigpi, sigk, app, apip, akp, appi, apipi, akpi, apk, apik, akk = fit_params
# 
#             protonEnhNorm = -(np.pi/2)**.5*(
#                 akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                 app*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                 apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                 )
#             pionEnhNorm = -(np.pi/2)**.5*(
#                 akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)) + 
#                 appi*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                 apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi))
#                 )
#             kaonEnhNorm = -(np.pi/2)**.5*(
#                 apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)) + 
#                 apk*sigp*( erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)) +
#                 akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk))
#                 )
#             
#             fpp = -(np.pi/2)**.5*(app*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/protonEnhNorm
#             fpip = -(np.pi/2)**.5*(apip*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/protonEnhNorm
#             fkp = -(np.pi/2)**.5*(akp*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/protonEnhNorm
#             fppi = -(np.pi/2)**.5*(appi*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/pionEnhNorm
#             fpipi = -(np.pi/2)**.5*(apipi*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/pionEnhNorm
#             fkpi = -(np.pi/2)**.5*(akpi*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/pionEnhNorm
#             fpk = -(np.pi/2)**.5*(apk*sigp*(erf(0.707107*(-10+mup)/sigp) - erf(0.707107*(10+mup)/sigp)))/kaonEnhNorm
#             fpik = -(np.pi/2)**.5*(apik*sigpi*(erf(0.707107*(-10+mupi)/sigpi) - erf(0.707107*(10+mupi)/sigpi)))/kaonEnhNorm
#             fkk = -(np.pi/2)**.5*(akk*sigk*(erf(0.707107*(-10+muk)/sigk) - erf(0.707107*(10+muk)/sigk)))/kaonEnhNorm
# 
#             
#             determinant_factor = -(fkk*fppi*fpip-fkk*fpipi*fpp+fkp*fpik*fppi-fkp*fpipi*fpk-fkpi*fpik*fpp+fkpi*fppi*fpk)
#             inv_mat = unp.ulinalg.inv(np.array([[fpipi, fppi, fkpi], [fpip, fpp, fkp], [fpik, fpk, fkk]]))
# 
#             row_sums = np.array([fsum(unp.fabs(inv_mat[0])), fsum(unp.fabs(inv_mat[1])), fsum(unp.fabs(inv_mat[2]))])
#             inv_mat = inv_mat/row_sums[:,np.newaxis]
#                     
#                     
#             pion_comp_str = f"pi: {inv_mat[0,0]:.3f} p:{inv_mat[0,1]:.3f} k:{inv_mat[0,2]:.3f}"
#             proton_comp_str = f"pi: {inv_mat[1,0]:.3f} p:{inv_mat[1,1]:.3f} k:{inv_mat[1,2]:.3f}"
#             kaon_comp_str = f"pi: {inv_mat[2,0]:.3f} p:{inv_mat[2,1]:.3f} k:{inv_mat[2,2]:.3f}"
#             
#             comp_str = pion_comp_str if species=="pion" else proton_comp_str if species=="proton" else kaon_comp_str

            #ax[0][0].text( 0.9, 0.85, f"Composition: {comp_str}", transform = ax[0][0].transAxes, ha='right')
        # add text to the axes that says "inclusive = (in+mid+out)/3"
        #ax[0][3].text( 0.1, 0.1, "Inclusive = (In+Mid+Out)", transform = ax[0][3].transAxes,)
        fig.suptitle(
            f"RPF fit for {species}s for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, Chi2/NDF = {self.RPFObjs[i,j].chi2OverNDF}"
        )  # type:ignore
        fig.tight_layout()
        fig.savefig(
            f"{self.base_save_path}RPF{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}{'withSig' if withSignal else ''}{species}ZV.png"
        )  # type:ignore


    @print_function_name_with_description_on_call(description="")
    def plot_optimal_parameters(self, i, withALICEData=True):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        optimal_param_errors = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        #for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
        for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
            optimal_params[j] = self.RPFObjs[i, j].popt  # type:ignore
            optimal_param_errors[j] = np.sqrt(
                    np.diag(self.RPFObjs[i, j].pcov)
                )  # type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc
        B = optimal_params[:, 0]
        Berr = optimal_param_errors[:, 0]
        ax.errorbar(
            self.pTassocBinCenters,
            B,
            yerr=Berr,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$B$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.suptitle(r"$B$ vs $p_{T,assoc}$")
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}.png")  # type:ignore
        plt.close(fig)
        
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:, 1]
        v2err = optimal_param_errors[:, 1]
        ax.errorbar(
            self.pTassocBinCenters,
            v2,
            yerr=v2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_2$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.12)
        fig.suptitle(r"$v_2$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}.png")  # type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:, 2]
        v3err = optimal_param_errors[:, 2]
        ax.errorbar(
                self.pTassocBinCenters,
                v3,
                yerr=v3err,
                xerr=np.array(self.pTassocBinWidths)/2, 
                fmt="o",
                ms=2,
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
                f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
            )  # type:ignore
        ax.set_ylim(-0.2, 0.1)
        fig.suptitle(r"$v_3$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}.png")  # type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:, 3]
        v4err = optimal_param_errors[:, 3]
        ax.errorbar(
            self.pTassocBinCenters,
            v4,
            yerr=v4err,\
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_4$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.2, 0.2)
        fig.suptitle(r"$v_4$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}.png")  # type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:, 4]
        va2err = optimal_param_errors[:, 4]
        ax.errorbar(
            self.pTassocBinCenters,
            va2,
            yerr=va2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        # load the va2 measurement from ALICE at ./Centralvn.csv and SemiCentralvn.csv
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else [np.loadtxt("./30-40.csv", delimiter=",", skiprows=1), np.loadtxt("./40-50.csv", delimiter=",", skiprows=1)]
            if self.analysisType=='central': 
                x_vals = alice_vn[:, 0]
                delta_x = x_vals-alice_vn[:,1]
                alice_va2 = alice_vn[:,3]
                alice_va2_error = np.sqrt(alice_vn[:,4]**2+alice_vn[:,5]**2)
                ax.errorbar(
                    x_vals,
                    alice_va2,
                    yerr=alice_va2_error,
                    xerr=delta_x,
                    fmt="o",
                    ms=2,
                    label="ALICE"
                )  # type:ignore
            if self.analysisType == 'semicentral':
                x_vals_3040 = alice_vn[0][:, 0]
                x_vals_4050 = alice_vn[1][:, 0]
                
                delta_x_3040 = x_vals_3040-alice_vn[0][:,1]
                delta_x_4050 = x_vals_4050-alice_vn[1][:,1]
                alice_va2_3040 = alice_vn[0][:,3]
                alice_va2_4050 = alice_vn[1][:,3]
                alice_va2_error_3040 = np.sqrt(alice_vn[0][:,4]**2+alice_vn[0][:,5]**2)
                alice_va2_error_4050 = np.sqrt(alice_vn[1][:,4]**2+alice_vn[1][:,5]**2)

                
                '''ax.errorbar(
                    x_vals_3040,
                    alice_va2_3040,
                    yerr=alice_va2_error_3040,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-40%"
                )'''
                '''ax.errorbar(
                    x_vals_4050,
                    alice_va2_4050,
                    yerr=alice_va2_error_4050,
                    xerr=delta_x_4050,
                    fmt="o",
                    ms=2,
                    label="ALICE 40-50%"
                )'''
                # make a list of tuples which are the x_vals in the 40-50% bin that correspond to the x_vals in the 30-40% bin
                x_vals_4050_3040 = []
                for ind,x in enumerate(x_vals_3040):
                    x_vals_4050_3040.append(np.argwhere(np.abs(x_vals_4050-x)<delta_x_3040[ind]))
                # average the va2 values and errors for each tuple in the list
                alice_va2_4050_3040 = []
                alice_va2_error_4050_3040 = []
                for inds in x_vals_4050_3040:
                    alice_va2_4050_3040.append(np.mean(alice_va2_4050[inds]))
                    alice_va2_error_4050_3040.append(np.mean(alice_va2_error_4050[inds]))
                alice_va2_4050_3040 = np.array(alice_va2_4050_3040)
                alice_va2_error_4050_3040 = np.array(alice_va2_error_4050_3040)


                ax.errorbar(
                    x_vals_3040,
                    (alice_va2_3040+alice_va2_4050_3040)/2,
                    yerr=np.sqrt(alice_va2_error_3040**2+alice_va2_error_4050_3040**2)/2,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-50\%"
                )



        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a2}$")
        ax.legend()
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.3)
        ax.set_xlim(0,10)
        fig.suptitle(r"$v_{a2}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}.png")  # type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[ :, 5]
        va4err = optimal_param_errors[ :, 5]
        ax.errorbar(
            self.pTassocBinCenters,
            va4,
            yerr=va4err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else np.loadtxt("./SemiCentralvn.csv", delimiter=",", skiprows=1)
            x_vals = alice_vn[:, 0]
            delta_x = x_vals-alice_vn[:,1]
            alice_va4 = alice_vn[:,9]
            alice_va4_error = np.sqrt(alice_vn[:,10]**2+alice_vn[:,11]**2)
            ax.errorbar(
                x_vals,
                alice_va4,
                yerr=alice_va4_error,
                xerr=delta_x,
                fmt="o",
                ms=2,
                label="ALICE 30-40\%"
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a4}$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.2)
        fig.suptitle(r"$v_{a4}$ vs $p_{T,assoc}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}.png")  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_optimal_parameters_in_z_vertex_bins(self, i, withALICEData=True):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        optimal_param_errors = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        #for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
        for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
            optimal_params[j] = self.RPFObjsZV[i, j].popt  # type:ignore
            optimal_param_errors[j] = np.sqrt(
                    np.diag(self.RPFObjsZV[i, j].pcov)
                )  # type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc
        B = optimal_params[:, 0]
        Berr = optimal_param_errors[:, 0]
        ax.errorbar(
            self.pTassocBinCenters,
            B,
            yerr=Berr,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$B$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.suptitle(r"$B$ vs $p_{T,assoc}$")
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}ZV.png")  # type:ignore
        plt.close(fig)
        
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:, 1]
        v2err = optimal_param_errors[:, 1]
        ax.errorbar(
            self.pTassocBinCenters,
            v2,
            yerr=v2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_2$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.12)
        fig.suptitle(r"$v_2$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:, 2]
        v3err = optimal_param_errors[:, 2]
        ax.errorbar(
                self.pTassocBinCenters,
                v3,
                yerr=v3err,
                xerr=np.array(self.pTassocBinWidths)/2, 
                fmt="o",
                ms=2,
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
                f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
            )  # type:ignore
        ax.set_ylim(-0.2, 0.1)
        fig.suptitle(r"$v_3$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:, 3]
        v4err = optimal_param_errors[:, 3]
        ax.errorbar(
            self.pTassocBinCenters,
            v4,
            yerr=v4err,\
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_4$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.2, 0.2)
        fig.suptitle(r"$v_4$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:, 4]
        va2err = optimal_param_errors[:, 4]
        ax.errorbar(
            self.pTassocBinCenters,
            va2,
            yerr=va2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        # load the va2 measurement from ALICE at ./Centralvn.csv and SemiCentralvn.csv
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else [np.loadtxt("./30-40.csv", delimiter=",", skiprows=1), np.loadtxt("./40-50.csv", delimiter=",", skiprows=1)]
            if self.analysisType=='central': 
                x_vals = alice_vn[:, 0]
                delta_x = x_vals-alice_vn[:,1]
                alice_va2 = alice_vn[:,3]
                alice_va2_error = np.sqrt(alice_vn[:,4]**2+alice_vn[:,5]**2)
                ax.errorbar(
                    x_vals,
                    alice_va2,
                    yerr=alice_va2_error,
                    xerr=delta_x,
                    fmt="o",
                    ms=2,
                    label="ALICE"
                )  # type:ignore
            if self.analysisType == 'semicentral':
                x_vals_3040 = alice_vn[0][:, 0]
                x_vals_4050 = alice_vn[1][:, 0]
                
                delta_x_3040 = x_vals_3040-alice_vn[0][:,1]
                delta_x_4050 = x_vals_4050-alice_vn[1][:,1]
                alice_va2_3040 = alice_vn[0][:,3]
                alice_va2_4050 = alice_vn[1][:,3]
                alice_va2_error_3040 = np.sqrt(alice_vn[0][:,4]**2+alice_vn[0][:,5]**2)
                alice_va2_error_4050 = np.sqrt(alice_vn[1][:,4]**2+alice_vn[1][:,5]**2)

                
                '''ax.errorbar(
                    x_vals_3040,
                    alice_va2_3040,
                    yerr=alice_va2_error_3040,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-40%"
                )'''
                '''ax.errorbar(
                    x_vals_4050,
                    alice_va2_4050,
                    yerr=alice_va2_error_4050,
                    xerr=delta_x_4050,
                    fmt="o",
                    ms=2,
                    label="ALICE 40-50%"
                )'''
                # make a list of tuples which are the x_vals in the 40-50% bin that correspond to the x_vals in the 30-40% bin
                x_vals_4050_3040 = []
                for ind,x in enumerate(x_vals_3040):
                    x_vals_4050_3040.append(np.argwhere(np.abs(x_vals_4050-x)<delta_x_3040[ind]))
                # average the va2 values and errors for each tuple in the list
                alice_va2_4050_3040 = []
                alice_va2_error_4050_3040 = []
                for inds in x_vals_4050_3040:
                    alice_va2_4050_3040.append(np.mean(alice_va2_4050[inds]))
                    alice_va2_error_4050_3040.append(np.mean(alice_va2_error_4050[inds]))
                alice_va2_4050_3040 = np.array(alice_va2_4050_3040)
                alice_va2_error_4050_3040 = np.array(alice_va2_error_4050_3040)


                ax.errorbar(
                    x_vals_3040,
                    (alice_va2_3040+alice_va2_4050_3040)/2,
                    yerr=np.sqrt(alice_va2_error_3040**2+alice_va2_error_4050_3040**2)/2,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-50\%"
                )



        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a2}$")
        ax.legend()
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.3)
        ax.set_xlim(0,10)
        fig.suptitle(r"$v_{a2}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[ :, 5]
        va4err = optimal_param_errors[ :, 5]
        ax.errorbar(
            self.pTassocBinCenters,
            va4,
            yerr=va4err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else np.loadtxt("./SemiCentralvn.csv", delimiter=",", skiprows=1)
            x_vals = alice_vn[:, 0]
            delta_x = x_vals-alice_vn[:,1]
            alice_va4 = alice_vn[:,9]
            alice_va4_error = np.sqrt(alice_vn[:,10]**2+alice_vn[:,11]**2)
            ax.errorbar(
                x_vals,
                alice_va4,
                yerr=alice_va4_error,
                xerr=delta_x,
                fmt="o",
                ms=2,
                label="ALICE 30-40\%"
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a4}$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.2)
        fig.suptitle(r"$v_{a4}$ vs $p_{T,assoc}$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}ZV.png")  # type:ignore

    @print_function_name_with_description_on_call(description="")
    def plot_optimal_parameters_for_true_species(self, i, species, withALICEData=True):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        optimal_param_errors = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        #for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
        for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
            optimal_params[j] = self.RPFObjsForTrueSpecies[species][i, j].popt  # type:ignore
            optimal_param_errors[j] = np.sqrt(
                    np.diag(self.RPFObjsForTrueSpecies[species][i, j].pcov)
                )  # type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc
        B = optimal_params[:, 0]
        Berr = optimal_param_errors[:, 0]
        ax.errorbar(
            self.pTassocBinCenters,
            B,
            yerr=Berr,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$B$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.suptitle(r"$B$ vs $p_{T,assoc}$")
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:, 1]
        v2err = optimal_param_errors[:, 1]
        ax.errorbar(
            self.pTassocBinCenters,
            v2,
            yerr=v2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_2$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.12)
        fig.suptitle(r"$v_2$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:, 2]
        v3err = optimal_param_errors[:, 2]
        ax.errorbar(
                self.pTassocBinCenters,
                v3,
                yerr=v3err,
                xerr=np.array(self.pTassocBinWidths)/2, 
                fmt="o",
                ms=2,
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
                f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
            )  # type:ignore
        ax.set_ylim(-0.2, 0.1)
        fig.suptitle(r"$v_3$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:, 3]
        v4err = optimal_param_errors[:, 3]
        ax.errorbar(
            self.pTassocBinCenters,
            v4,
            yerr=v4err,\
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_4$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.2, 0.2)
        fig.suptitle(r"$v_4$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:, 4]
        va2err = optimal_param_errors[:, 4]
        ax.errorbar(
            self.pTassocBinCenters,
            va2,
            yerr=va2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        # load the va2 measurement from ALICE at ./Centralvn.csv and SemiCentralvn.csv
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else [np.loadtxt("./30-40.csv", delimiter=",", skiprows=1), np.loadtxt("./40-50.csv", delimiter=",", skiprows=1)]
            if self.analysisType=='central': 
                x_vals = alice_vn[:, 0]
                delta_x = x_vals-alice_vn[:,1]
                alice_va2 = alice_vn[:,3]
                alice_va2_error = np.sqrt(alice_vn[:,4]**2+alice_vn[:,5]**2)
                ax.errorbar(
                    x_vals,
                    alice_va2,
                    yerr=alice_va2_error,
                    xerr=delta_x,
                    fmt="o",
                    ms=2,
                    label="ALICE"
                )  # type:ignore
            if self.analysisType == 'semicentral':
                x_vals_3040 = alice_vn[0][:, 0]
                x_vals_4050 = alice_vn[1][:, 0]
                
                delta_x_3040 = x_vals_3040-alice_vn[0][:,1]
                delta_x_4050 = x_vals_4050-alice_vn[1][:,1]
                alice_va2_3040 = alice_vn[0][:,3]
                alice_va2_4050 = alice_vn[1][:,3]
                alice_va2_error_3040 = np.sqrt(alice_vn[0][:,4]**2+alice_vn[0][:,5]**2)
                alice_va2_error_4050 = np.sqrt(alice_vn[1][:,4]**2+alice_vn[1][:,5]**2)

                
                ax.errorbar(
                    x_vals_3040,
                    alice_va2_3040,
                    yerr=alice_va2_error_3040,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-40%"
                )
                ax.errorbar(
                    x_vals_4050,
                    alice_va2_4050,
                    yerr=alice_va2_error_4050,
                    xerr=delta_x_4050,
                    fmt="o",
                    ms=2,
                    label="ALICE 40-50%"
                )
                # make a list of tuples which are the x_vals in the 40-50% bin that correspond to the x_vals in the 30-40% bin
                x_vals_4050_3040 = []
                for ind,x in enumerate(x_vals_3040):
                    x_vals_4050_3040.append(np.argwhere(np.abs(x_vals_4050-x)<delta_x_3040[ind]))
                # average the va2 values and errors for each tuple in the list
                alice_va2_4050_3040 = []
                alice_va2_error_4050_3040 = []
                for inds in x_vals_4050_3040:
                    alice_va2_4050_3040.append(np.mean(alice_va2_4050[inds]))
                    alice_va2_error_4050_3040.append(np.mean(alice_va2_error_4050[inds]))
                alice_va2_4050_3040 = np.array(alice_va2_4050_3040)
                alice_va2_error_4050_3040 = np.array(alice_va2_error_4050_3040)


                ax.errorbar(
                    x_vals_3040,
                    (alice_va2_3040+alice_va2_4050_3040)/2,
                    yerr=np.sqrt(alice_va2_error_3040**2+alice_va2_error_4050_3040**2)/2,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-50%"
                )



        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a2}$")
        ax.legend()
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.3)
        ax.set_xlim(0,10)
        fig.suptitle(r"$v_{a2}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[ :, 5]
        va4err = optimal_param_errors[ :, 5]
        ax.errorbar(
            self.pTassocBinCenters,
            va4,
            yerr=va4err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else np.loadtxt("./SemiCentralvn.csv", delimiter=",", skiprows=1)
            x_vals = alice_vn[:, 0]
            delta_x = x_vals-alice_vn[:,1]
            alice_va4 = alice_vn[:,9]
            alice_va4_error = np.sqrt(alice_vn[:,10]**2+alice_vn[:,11]**2)
            ax.errorbar(
                x_vals,
                alice_va4,
                yerr=alice_va4_error,
                xerr=delta_x,
                fmt="o",
                ms=2,
                label="ALICE"
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a4}$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.2)
        fig.suptitle(r"$v_{a4}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_optimal_parameters_for_true_species_in_z_vertex_bins(self, i, species, withALICEData=True):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        optimal_param_errors = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        #for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
        for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
            optimal_params[j] = self.RPFObjsForTrueSpeciesZV[species][i, j].popt  # type:ignore
            optimal_param_errors[j] = np.sqrt(
                    np.diag(self.RPFObjsForTrueSpeciesZV[species][i, j].pcov)
                )  # type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc
        B = optimal_params[:, 0]
        Berr = optimal_param_errors[:, 0]
        ax.errorbar(
            self.pTassocBinCenters,
            B,
            yerr=Berr,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$B$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.suptitle(r"$B$ vs $p_{T,assoc}$")
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:, 1]
        v2err = optimal_param_errors[:, 1]
        ax.errorbar(
            self.pTassocBinCenters,
            v2,
            yerr=v2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_2$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.12)
        fig.suptitle(r"$v_2$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:, 2]
        v3err = optimal_param_errors[:, 2]
        ax.errorbar(
                self.pTassocBinCenters,
                v3,
                yerr=v3err,
                xerr=np.array(self.pTassocBinWidths)/2, 
                fmt="o",
                ms=2,
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
                f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
            )  # type:ignore
        ax.set_ylim(-0.2, 0.1)
        fig.suptitle(r"$v_3$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:, 3]
        v4err = optimal_param_errors[:, 3]
        ax.errorbar(
            self.pTassocBinCenters,
            v4,
            yerr=v4err,\
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_4$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.2, 0.2)
        fig.suptitle(r"$v_4$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:, 4]
        va2err = optimal_param_errors[:, 4]
        ax.errorbar(
            self.pTassocBinCenters,
            va2,
            yerr=va2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        # load the va2 measurement from ALICE at ./Centralvn.csv and SemiCentralvn.csv
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else [np.loadtxt("./30-40.csv", delimiter=",", skiprows=1), np.loadtxt("./40-50.csv", delimiter=",", skiprows=1)]
            if self.analysisType=='central': 
                x_vals = alice_vn[:, 0]
                delta_x = x_vals-alice_vn[:,1]
                alice_va2 = alice_vn[:,3]
                alice_va2_error = np.sqrt(alice_vn[:,4]**2+alice_vn[:,5]**2)
                ax.errorbar(
                    x_vals,
                    alice_va2,
                    yerr=alice_va2_error,
                    xerr=delta_x,
                    fmt="o",
                    ms=2,
                    label="ALICE"
                )  # type:ignore
            if self.analysisType == 'semicentral':
                x_vals_3040 = alice_vn[0][:, 0]
                x_vals_4050 = alice_vn[1][:, 0]
                
                delta_x_3040 = x_vals_3040-alice_vn[0][:,1]
                delta_x_4050 = x_vals_4050-alice_vn[1][:,1]
                alice_va2_3040 = alice_vn[0][:,3]
                alice_va2_4050 = alice_vn[1][:,3]
                alice_va2_error_3040 = np.sqrt(alice_vn[0][:,4]**2+alice_vn[0][:,5]**2)
                alice_va2_error_4050 = np.sqrt(alice_vn[1][:,4]**2+alice_vn[1][:,5]**2)

                
                ax.errorbar(
                    x_vals_3040,
                    alice_va2_3040,
                    yerr=alice_va2_error_3040,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-40%"
                )
                ax.errorbar(
                    x_vals_4050,
                    alice_va2_4050,
                    yerr=alice_va2_error_4050,
                    xerr=delta_x_4050,
                    fmt="o",
                    ms=2,
                    label="ALICE 40-50%"
                )
                # make a list of tuples which are the x_vals in the 40-50% bin that correspond to the x_vals in the 30-40% bin
                x_vals_4050_3040 = []
                for ind,x in enumerate(x_vals_3040):
                    x_vals_4050_3040.append(np.argwhere(np.abs(x_vals_4050-x)<delta_x_3040[ind]))
                # average the va2 values and errors for each tuple in the list
                alice_va2_4050_3040 = []
                alice_va2_error_4050_3040 = []
                for inds in x_vals_4050_3040:
                    alice_va2_4050_3040.append(np.mean(alice_va2_4050[inds]))
                    alice_va2_error_4050_3040.append(np.mean(alice_va2_error_4050[inds]))
                alice_va2_4050_3040 = np.array(alice_va2_4050_3040)
                alice_va2_error_4050_3040 = np.array(alice_va2_error_4050_3040)


                ax.errorbar(
                    x_vals_3040,
                    (alice_va2_3040+alice_va2_4050_3040)/2,
                    yerr=np.sqrt(alice_va2_error_3040**2+alice_va2_error_4050_3040**2)/2,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-50%"
                )



        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a2}$")
        ax.legend()
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.3)
        ax.set_xlim(0,10)
        fig.suptitle(r"$v_{a2}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[ :, 5]
        va4err = optimal_param_errors[ :, 5]
        ax.errorbar(
            self.pTassocBinCenters,
            va4,
            yerr=va4err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else np.loadtxt("./SemiCentralvn.csv", delimiter=",", skiprows=1)
            x_vals = alice_vn[:, 0]
            delta_x = x_vals-alice_vn[:,1]
            alice_va4 = alice_vn[:,9]
            alice_va4_error = np.sqrt(alice_vn[:,10]**2+alice_vn[:,11]**2)
            ax.errorbar(
                x_vals,
                alice_va4,
                yerr=alice_va4_error,
                xerr=delta_x,
                fmt="o",
                ms=2,
                label="ALICE"
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a4}$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.2)
        fig.suptitle(r"$v_{a4}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_optimal_parameters_for_enhanced_species(self, i, species, withALICEData=True):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        optimal_param_errors = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        #for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
        for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
            optimal_params[j] = self.RPFObjsForEnhancedSpecies[species][i, j].popt  # type:ignore
            optimal_param_errors[j] = np.sqrt(
                    np.diag(self.RPFObjsForEnhancedSpecies[species][i, j].pcov)
                )  # type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc
        B = optimal_params[:, 0]
        Berr = optimal_param_errors[:, 0]
        ax.errorbar(
            self.pTassocBinCenters,
            B,
            yerr=Berr,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$B$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.suptitle(r"$B$ vs $p_{T,assoc}$")
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:, 1]
        v2err = optimal_param_errors[:, 1]
        ax.errorbar(
            self.pTassocBinCenters,
            v2,
            yerr=v2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_2$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.12)
        fig.suptitle(r"$v_2$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:, 2]
        v3err = optimal_param_errors[:, 2]
        ax.errorbar(
                self.pTassocBinCenters,
                v3,
                yerr=v3err,
                xerr=np.array(self.pTassocBinWidths)/2, 
                fmt="o",
                ms=2,
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
                f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
            )  # type:ignore
        ax.set_ylim(-0.2, 0.1)
        fig.suptitle(r"$v_3$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:, 3]
        v4err = optimal_param_errors[:, 3]
        ax.errorbar(
            self.pTassocBinCenters,
            v4,
            yerr=v4err,\
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_4$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.2, 0.2)
        fig.suptitle(r"$v_4$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:, 4]
        va2err = optimal_param_errors[:, 4]
        ax.errorbar(
            self.pTassocBinCenters,
            va2,
            yerr=va2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        # load the va2 measurement from ALICE at ./Centralvn.csv and SemiCentralvn.csv
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else [np.loadtxt("./30-40.csv", delimiter=",", skiprows=1), np.loadtxt("./40-50.csv", delimiter=",", skiprows=1)]
            if self.analysisType=='central': 
                x_vals = alice_vn[:, 0]
                delta_x = x_vals-alice_vn[:,1]
                alice_va2 = alice_vn[:,3]
                alice_va2_error = np.sqrt(alice_vn[:,4]**2+alice_vn[:,5]**2)
                ax.errorbar(
                    x_vals,
                    alice_va2,
                    yerr=alice_va2_error,
                    xerr=delta_x,
                    fmt="o",
                    ms=2,
                    label="ALICE"
                )  # type:ignore
            if self.analysisType == 'semicentral':
                x_vals_3040 = alice_vn[0][:, 0]
                x_vals_4050 = alice_vn[1][:, 0]
                
                delta_x_3040 = x_vals_3040-alice_vn[0][:,1]
                delta_x_4050 = x_vals_4050-alice_vn[1][:,1]
                alice_va2_3040 = alice_vn[0][:,3]
                alice_va2_4050 = alice_vn[1][:,3]
                alice_va2_error_3040 = np.sqrt(alice_vn[0][:,4]**2+alice_vn[0][:,5]**2)
                alice_va2_error_4050 = np.sqrt(alice_vn[1][:,4]**2+alice_vn[1][:,5]**2)

                
                ax.errorbar(
                    x_vals_3040,
                    alice_va2_3040,
                    yerr=alice_va2_error_3040,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-40%"
                )
                ax.errorbar(
                    x_vals_4050,
                    alice_va2_4050,
                    yerr=alice_va2_error_4050,
                    xerr=delta_x_4050,
                    fmt="o",
                    ms=2,
                    label="ALICE 40-50%"
                )
                # make a list of tuples which are the x_vals in the 40-50% bin that correspond to the x_vals in the 30-40% bin
                x_vals_4050_3040 = []
                for ind,x in enumerate(x_vals_3040):
                    x_vals_4050_3040.append(np.argwhere(np.abs(x_vals_4050-x)<delta_x_3040[ind]))
                # average the va2 values and errors for each tuple in the list
                alice_va2_4050_3040 = []
                alice_va2_error_4050_3040 = []
                for inds in x_vals_4050_3040:
                    alice_va2_4050_3040.append(np.mean(alice_va2_4050[inds]))
                    alice_va2_error_4050_3040.append(np.mean(alice_va2_error_4050[inds]))
                alice_va2_4050_3040 = np.array(alice_va2_4050_3040)
                alice_va2_error_4050_3040 = np.array(alice_va2_error_4050_3040)


                ax.errorbar(
                    x_vals_3040,
                    (alice_va2_3040+alice_va2_4050_3040)/2,
                    yerr=np.sqrt(alice_va2_error_3040**2+alice_va2_error_4050_3040**2)/2,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-50%"
                )



        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a2}$")
        ax.legend()
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.3)
        ax.set_xlim(0,10)
        fig.suptitle(r"$v_{a2}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[ :, 5]
        va4err = optimal_param_errors[ :, 5]
        ax.errorbar(
            self.pTassocBinCenters,
            va4,
            yerr=va4err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else np.loadtxt("./SemiCentralvn.csv", delimiter=",", skiprows=1)
            x_vals = alice_vn[:, 0]
            delta_x = x_vals-alice_vn[:,1]
            alice_va4 = alice_vn[:,9]
            alice_va4_error = np.sqrt(alice_vn[:,10]**2+alice_vn[:,11]**2)
            ax.errorbar(
                x_vals,
                alice_va4,
                yerr=alice_va4_error,
                xerr=delta_x,
                fmt="o",
                ms=2,
                label="ALICE"
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a4}$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.2)
        fig.suptitle(r"$v_{a4}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}.png")  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_optimal_parameters_for_enhanced_species_in_z_vertex_bins(self, i, species, withALICEData=True):
        # retrieve optimal parameters and put in numpy array
        optimal_params = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        optimal_param_errors = np.zeros(
            (len(self.pTassocBinEdges) - 1, 6)
        )  # type:ignore
        #for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
        for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
            optimal_params[j] = self.RPFObjsForEnhancedSpeciesZV[species][i, j].popt  # type:ignore
            optimal_param_errors[j] = np.sqrt(
                    np.diag(self.RPFObjsForEnhancedSpeciesZV[species][i, j].pcov)
                )  # type:ignore
        # plot optimal parameters as a function of pTassoc for each pTtrig bin
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0)
        # first plot B vs ptassoc
        B = optimal_params[:, 0]
        Berr = optimal_param_errors[:, 0]
        ax.errorbar(
            self.pTassocBinCenters,
            B,
            yerr=Berr,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$B$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        fig.tight_layout()
        fig.suptitle(r"$B$ vs $p_{T,assoc}$")
        fig.savefig(f"{self.base_save_path}B_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        
        # now plot v2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v2 = optimal_params[:, 1]
        v2err = optimal_param_errors[:, 1]
        ax.errorbar(
            self.pTassocBinCenters,
            v2,
            yerr=v2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_2$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.12)
        fig.suptitle(r"$v_2$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot v3 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v3 = optimal_params[:, 2]
        v3err = optimal_param_errors[:, 2]
        ax.errorbar(
                self.pTassocBinCenters,
                v3,
                yerr=v3err,
                xerr=np.array(self.pTassocBinWidths)/2, 
                fmt="o",
                ms=2,
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_3$")
        ax.set_title(
                f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
            )  # type:ignore
        ax.set_ylim(-0.2, 0.1)
        fig.suptitle(r"$v_3$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v3_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot v4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        v4 = optimal_params[:, 3]
        v4err = optimal_param_errors[:, 3]
        ax.errorbar(
            self.pTassocBinCenters,
            v4,
            yerr=v4err,\
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_4$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.2, 0.2)
        fig.suptitle(r"$v_4$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}v4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot va2 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va2 = optimal_params[:, 4]
        va2err = optimal_param_errors[:, 4]
        ax.errorbar(
            self.pTassocBinCenters,
            va2,
            yerr=va2err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
            label="This Analysis"
        )  # type:ignore
        # load the va2 measurement from ALICE at ./Centralvn.csv and SemiCentralvn.csv
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else [np.loadtxt("./30-40.csv", delimiter=",", skiprows=1), np.loadtxt("./40-50.csv", delimiter=",", skiprows=1)]
            if self.analysisType=='central': 
                x_vals = alice_vn[:, 0]
                delta_x = x_vals-alice_vn[:,1]
                alice_va2 = alice_vn[:,3]
                alice_va2_error = np.sqrt(alice_vn[:,4]**2+alice_vn[:,5]**2)
                ax.errorbar(
                    x_vals,
                    alice_va2,
                    yerr=alice_va2_error,
                    xerr=delta_x,
                    fmt="o",
                    ms=2,
                    label="ALICE"
                )  # type:ignore
            if self.analysisType == 'semicentral':
                x_vals_3040 = alice_vn[0][:, 0]
                x_vals_4050 = alice_vn[1][:, 0]
                
                delta_x_3040 = x_vals_3040-alice_vn[0][:,1]
                delta_x_4050 = x_vals_4050-alice_vn[1][:,1]
                alice_va2_3040 = alice_vn[0][:,3]
                alice_va2_4050 = alice_vn[1][:,3]
                alice_va2_error_3040 = np.sqrt(alice_vn[0][:,4]**2+alice_vn[0][:,5]**2)
                alice_va2_error_4050 = np.sqrt(alice_vn[1][:,4]**2+alice_vn[1][:,5]**2)

                
                ax.errorbar(
                    x_vals_3040,
                    alice_va2_3040,
                    yerr=alice_va2_error_3040,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-40%"
                )
                ax.errorbar(
                    x_vals_4050,
                    alice_va2_4050,
                    yerr=alice_va2_error_4050,
                    xerr=delta_x_4050,
                    fmt="o",
                    ms=2,
                    label="ALICE 40-50%"
                )
                # make a list of tuples which are the x_vals in the 40-50% bin that correspond to the x_vals in the 30-40% bin
                x_vals_4050_3040 = []
                for ind,x in enumerate(x_vals_3040):
                    x_vals_4050_3040.append(np.argwhere(np.abs(x_vals_4050-x)<delta_x_3040[ind]))
                # average the va2 values and errors for each tuple in the list
                alice_va2_4050_3040 = []
                alice_va2_error_4050_3040 = []
                for inds in x_vals_4050_3040:
                    alice_va2_4050_3040.append(np.mean(alice_va2_4050[inds]))
                    alice_va2_error_4050_3040.append(np.mean(alice_va2_error_4050[inds]))
                alice_va2_4050_3040 = np.array(alice_va2_4050_3040)
                alice_va2_error_4050_3040 = np.array(alice_va2_error_4050_3040)


                ax.errorbar(
                    x_vals_3040,
                    (alice_va2_3040+alice_va2_4050_3040)/2,
                    yerr=np.sqrt(alice_va2_error_3040**2+alice_va2_error_4050_3040**2)/2,
                    xerr=delta_x_3040,
                    fmt="o",
                    ms=2,
                    label="ALICE 30-50%"
                )



        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a2}$")
        ax.legend()
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.3)
        ax.set_xlim(0,10)
        fig.suptitle(r"$v_{a2}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va2_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore
        plt.close(fig)
        # now plot va4 vs ptassoc
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), )
        fig.subplots_adjust(wspace=0)
        va4 = optimal_params[ :, 5]
        va4err = optimal_param_errors[ :, 5]
        ax.errorbar(
            self.pTassocBinCenters,
            va4,
            yerr=va4err,
            xerr=np.array(self.pTassocBinWidths)/2, 
            fmt="o",
            ms=2,
        )  # type:ignore
        if withALICEData:
            alice_vn = np.loadtxt("./Centralvn.csv", delimiter=",", skiprows=1) if self.analysisType=='central' else np.loadtxt("./SemiCentralvn.csv", delimiter=",", skiprows=1)
            x_vals = alice_vn[:, 0]
            delta_x = x_vals-alice_vn[:,1]
            alice_va4 = alice_vn[:,9]
            alice_va4_error = np.sqrt(alice_vn[:,10]**2+alice_vn[:,11]**2)
            ax.errorbar(
                x_vals,
                alice_va4,
                yerr=alice_va4_error,
                xerr=delta_x,
                fmt="o",
                ms=2,
                label="ALICE"
            )  # type:ignore
        ax.set_xlabel(r"$p_{T,assoc}$ (GeV/c)")
        ax.set_ylabel(r"$v_{a4}$")
        ax.set_title(
            f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c"
        )  # type:ignore
        ax.set_ylim(-0.01, 0.2)
        fig.suptitle(r"$v_{a4}$ vs $p_{T,assoc}$")
        fig.tight_layout()
        fig.savefig(f"{self.base_save_path}va4_vs_pTassoc_{'20-40GeV' if i==0 else '40-60GeV'}_{species}ZV.png")  # type:ignore


    @print_function_name_with_description_on_call(description="")
    def plot_one_bin(self, i, j, k):
        self.epString = (
            "out-of-plane"
            if k == 2
            else ("mid-plane" if k == 1 else ("in-plane" if k == 0 else "inclusive"))
        )
        debug_logger.info(
            f"Plotting bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
        )  # type:ignore
        
        print(
            f"Plotting bin {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
        )  # type:ignore

        correlation_func = plot_TH2(
            self.SEcorrs[i, j, k],
            f"Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
            "$\\Delta \\eta$",
            "$\\Delta \\phi$",
            "Counts",
            cmap="jet",
        )  # type:ignore
        correlation_func.savefig(
            f"{self.base_save_path}{self.epString}/CorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
            dpi=300,
        )  # type:ignore
        plt.close(correlation_func)

        normMEcorrelation_function = plot_TH2(
            self.NormMEcorrs[i, j, k],
            f"Normalized Mixed Event Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
            "\\Delta \\eta",
            "\\Delta \\phi",
            "Counts",
            cmap="jet",
        )  # type:ignore
        normMEcorrelation_function.savefig(
            f"{self.base_save_path}{self.epString}/NormalizedMixedEventCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
            dpi=300,
        )  # type:ignore
        plt.close(normMEcorrelation_function)

        accCorrectedCorrelationFunction = plot_TH2(
            self.AccCorrectedSEcorrs[i, j, k],
            f"Acceptance Corrected Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
            "\\Delta \\eta",
            "\\Delta \\phi",
            "Counts",
            cmap="jet",
        )  # type:ignore
        accCorrectedCorrelationFunction.savefig(
            f"{self.base_save_path}{self.epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
            dpi=300,
        )  # type:ignore
        plt.close(accCorrectedCorrelationFunction)

        normAccCorrectedCorrelationFunction = plot_TH2(
            self.NormAccCorrectedSEcorrs[i, j, k],
            f"Normalized Acceptance Corrected Correlation Function {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
            "\\Delta \\eta",
            "\\Delta \\phi",
            "Counts",
            cmap="jet",
        )  # type:ignore
        normAccCorrectedCorrelationFunction.savefig(
            f"{self.base_save_path}{self.epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
            dpi=300,
        )  # type:ignore
        plt.close(normAccCorrectedCorrelationFunction)

        dPhiBG, dPhiBGax = plt.subplots(1, 1, figsize=(5, 5))
        dPhiBGax = plot_TH1(
            self.dPhiBGcorrs[i, j, k],
            f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
            "$\\Delta \\phi$",
            "Counts",
            ax=dPhiBGax,
        )  # type:ignore
        dPhiBG.savefig(
            f"{self.base_save_path}{self.epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
        )  # type:ignore
        plt.close(dPhiBG)

        dPhiSig, dPhiSigax = plt.subplots(1, 1, figsize=(5, 5))
        dPhiSigax = plot_TH1(
            self.dPhiSigcorrs[i, j, k],
            f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
            "$\\Delta \\phi$",
            "Counts",
            ax=dPhiSigax,
        )  # type:ignore
        dPhiSig.savefig(
            f"{self.base_save_path}{self.epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
        )  # type:ignore
        plt.close(dPhiSig)

    @print_function_name_with_description_on_call(description="")
    def plot_pion_tpc_signal(self, i, j,):
        pionTPCsignal, pionTPCsignalax = plt.subplots(1, 1, figsize=(5, 5))
        pionTPCsignalax = plot_TH1(
            self.pionTPCnSigmaInc[i, j],
            f"Pion TPC signal {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c",
            "$(\\frac{dE}{dx})_{meas}-(\\frac{dE}{dx})_{calc}$",
            "Counts",
            ax=pionTPCsignalax,
        )  # type:ignore
        pionTPCsignal.tight_layout()
        pionTPCsignal.savefig(
            f"{self.base_save_path}pionTPCsignal{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
        )  # type:ignore
        plt.close(pionTPCsignal)

    @print_function_name_with_description_on_call(description="")
    def plot_pion_tpc_nSigma(self, particleType, i, j, k):
        pionTPCsignal, pionTPCsignalax = plt.subplots(1, 1, figsize=(5, 5))
        if particleType == "pion":
            if self.analysisType in ["central", "semicentral"]:
                 pionTPCsignalax = plot_TH1(
                    self.pionTPCnSigma_pionTOFcut[i, j, k],
                    f"Pion TPC nSigma after 2 $\sigma$ pion TOF cut {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                    "$n\sigma$",
                    "Counts",
                    ax=pionTPCsignalax,
                )
            else:
                pionTPCsignalax = plot_TH1(
                    self.pionTPCnSigma_pionTOFcut[i, j],
                    f"Pion TPC nSigma after 2 $\sigma$ pion TOF cut {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c",
                    "$n\sigma$",
                    "Counts",
                    ax=pionTPCsignalax,
                )

        elif particleType == "kaon":
            if self.analysisType in ["central", "semicentral"]:
                pionTPCsignalax = plot_TH1(
                    self.pionTPCnSigma_kaonTOFcut[i, j, k],
                    f"Pion TPC nSigma after 2 $\sigma$ kaon TOF cut {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                    "$n\sigma$",
                    "Counts",
                    ax=pionTPCsignalax,
                )
            else:
                pionTPCsignalax = plot_TH1(
                    self.pionTPCnSigma_kaonTOFcut[i, j],
                    f"Pion TPC nSigma after 2 $\sigma$ kaon TOF cut {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c",
                    "$n\sigma$",
                    "Counts",
                    ax=pionTPCsignalax,
                )

        elif particleType == "proton":
            if self.analysisType in ["central", "semicentral"]:
                pionTPCsignalax = plot_TH1(
                    self.pionTPCnSigma_protonTOFcut[i, j, k],
                    f"Pion TPC nSigma after 2 $\sigma$ proton TOF cut {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                    "$n\sigma$",
                    "Counts",
                    ax=pionTPCsignalax,
                )
            else:
                pionTPCsignalax = plot_TH1(
                    self.pionTPCnSigma_protonTOFcut[i, j],
                    f"Pion TPC nSigma after 2 $\sigma$ proton TOF cut {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c",
                    "$n\sigma$",
                    "Counts",
                    ax=pionTPCsignalax,
                )
        pionTPCsignal.tight_layout()
        pionTPCsignal.savefig(
            f"{self.base_save_path}{self.epString}/pionTPCnSigma_{particleType}TOFcut{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
        )  # type:ignore
        plt.close(pionTPCsignal)

    @print_function_name_with_description_on_call(description="")
    def plot_pionTPCnSigma_vs_dphi(self, i, j, k, species):
        if self.analysisType in ["central", "semicentral"]:
            pionTPCnSigma_vs_dphi, pionTPCnSigma_vs_dphiax = plt.subplots(
                1, 1, figsize=(5, 5)
            )
            pionTPCnSigma_vs_dphi = plot_TH2(
                self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs[
                    species
                ][i, j, k],
                f"Pion TPC nSigma vs dPhi {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                r"$\Delta\phi$",
                r"$n\sigma$",
                r"$\frac{1}{N_{trig} a \epsilon}\frac{d^2(N_{meas}-N_{BG})}{d\Delta\phi dn\sigma}$",
            )
            pionTPCnSigma_vs_dphi.tight_layout()
            pionTPCnSigma_vs_dphi.savefig(
                f"{self.base_save_path}{self.epString}/pionTPCnSigma_vs_dphi_{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(pionTPCnSigma_vs_dphi)

        elif self.analysisType == "pp":
            pionTPCnSigma_vs_dphi, pionTPCnSigma_vs_dphiax = plt.subplots(
                1, 1, figsize=(5, 5)
            )
            pionTPCnSigma_vs_dphi = plot_TH2(
                self.NormalizedBGSubtractedAccCorrectedSEdPhiSigdpionTPCnSigmacorrs[
                    species
                ][i, j],
                f"Pion TPC nSigma vs dPhi {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c",
                r"$\Delta\phi$",
                r"$n\sigma$",
                r"$\frac{1}{N_{trig} a \epsilon}\frac{d^2(N_{meas}-N_{BG})}{d\Delta\phi dn\sigma}$",
            )
            pionTPCnSigma_vs_dphi.tight_layout()
            pionTPCnSigma_vs_dphi.savefig(
                f"{self.base_save_path}pionTPCnSigma_vs_dphi_{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )

    @print_function_name_with_description_on_call(description="")
    def plot_SE_correlation_function(self, i, j, k, normalize=True):
        if normalize:
            if self.analysisType == "pp":
                N_trig = self.N_trigs[i]
            elif self.analysisType in ["central", "semicentral"]:
                N_trig = self.N_trigs[i,k]    
        if self.analysisType in ["central", "semicentral"]:
            # get SEcorrs and rebin
            SEcorr = self.SEcorrs[i, j, k].Clone()
            #SEcorr.Rebin2D(2, 4)
            #SEcorr.Scale(1/8)
            if normalize:
                debug_logger.debug(N_trig)
                SEcorr.Scale(1/N_trig)
            correlation_func = plot_TH2(
                SEcorr,
                f"Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{\epsilon}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            correlation_func.savefig(
                f"{self.base_save_path}{self.epString}/CorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(correlation_func)
        elif self.analysisType == "pp":
            # get SEcorrs and rebin
            SEcorr = self.SEcorrs[i, j].Clone()
            SEcorr.Rebin2D(2, 4)
            SEcorr.Scale(1/8)
            if normalize:
                SEcorr.Scale(1/N_trig)
            correlation_func = plot_TH2(
                SEcorr,
                f"Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{\epsilon}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            correlation_func.savefig(
                f"{self.base_save_path}{self.epString}/CorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(correlation_func)

    @print_function_name_with_description_on_call(description="")
    def plot_ME_correlation_function(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            # get MEcorrs and rebin
            MEcorr = self.NormMEcorrs[i, j, k].Clone()
            #MEcorr.Rebin2D(2, 4)
            #MEcorr.Scale(1/8)
            normMEcorrelation_function = plot_TH2(
                MEcorr,
                f"Normalized Mixed Event Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$a_0\\frac{d^2N_{mixed-event}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            normMEcorrelation_function.savefig(
                f"{self.base_save_path}{self.epString}/NormalizedMixedEventCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(normMEcorrelation_function)
        elif self.analysisType == "pp":
            # get MEcorrs and rebin
            MEcorr = self.NormMEcorrs[i, j].Clone()
            #MEcorr.Rebin2D(2, 4)
            #MEcorr.Scale(1/8)
            normMEcorrelation_function = plot_TH2(
                MEcorr,
                f"Normalized Mixed Event Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$a_0\\frac{d^2N_{mixed-event}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            normMEcorrelation_function.savefig(
                f"{self.base_save_path}{self.epString}/NormalizedMixedEventCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(normMEcorrelation_function)

    @print_function_name_with_description_on_call(description="")
    def plot_acceptance_corrected_SE_correlation_function(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            # get SEcorrs and rebin
            AccCorrSEcorr = self.AccCorrectedSEcorrs[i, j, k].Clone()
            #AccCorrSEcorr.Rebin2D(2, 4)
            #AccCorrSEcorr.Scale(1/8)
            accCorrectedCorrelationFunction = plot_TH2(
                AccCorrSEcorr,
                f"Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            accCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(accCorrectedCorrelationFunction)
        elif self.analysisType == "pp":
            # get SEcorrs and rebin
            AccCorrSEcorr = self.AccCorrectedSEcorrs[i, j].Clone()
            #AccCorrSEcorr.Rebin2D(2, 4)
            #AccCorrSEcorr.Scale(1/8)
            accCorrectedCorrelationFunction = plot_TH2(
                AccCorrSEcorr,
                f"Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore

            accCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(accCorrectedCorrelationFunction)
    
    @print_function_name_with_description_on_call(description="")
    def plot_acceptance_corrected_SE_correlation_function_in_z_vertex_bins(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            # get SEcorrs and rebin
            AccCorrSEcorr = self.AccCorrectedSEcorrsZV[i, j, k].Clone()
            #AccCorrSEcorr.Rebin2D(2, 4)
            #AccCorrSEcorr.Scale(1/8)
            accCorrectedCorrelationFunction = plot_TH2(
                AccCorrSEcorr,
                f"Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            accCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png",
                dpi=300,
            )  # type:ignore
            plt.close(accCorrectedCorrelationFunction)
        elif self.analysisType == "pp":
            # get SEcorrs and rebin
            AccCorrSEcorr = self.AccCorrectedSEcorrsZV[i, j].Clone()
            #AccCorrSEcorr.Rebin2D(2, 4)
            #AccCorrSEcorr.Scale(1/8)
            accCorrectedCorrelationFunction = plot_TH2(
                AccCorrSEcorr,
                f"Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore

            accCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/AcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png",
                dpi=300,
            )  # type:ignore
            plt.close(accCorrectedCorrelationFunction)

    @print_function_name_with_description_on_call(description="")
    def plot_normalized_acceptance_corrected_correlation_function(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            # get SEcorrs and rebin
            NormAccCorrectedSEcorr = self.NormAccCorrectedSEcorrs[i, j, k].Clone()
            #NormAccCorrectedSEcorr.Rebin2D(2, 4)
            #NormAccCorrectedSEcorr.Scale(1/8)
            normAccCorrectedCorrelationFunction = plot_TH2(
                NormAccCorrectedSEcorr,
                f"Normalized Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{N_{trig} \\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            normAccCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(normAccCorrectedCorrelationFunction)
        elif self.analysisType == "pp":
            # get SEcorrs and rebin
            NormAccCorrectedSEcorr = self.NormAccCorrectedSEcorrs[i, j].Clone()
            #NormAccCorrectedSEcorr.Rebin2D(2, 4)
            #NormAccCorrectedSEcorr.Scale(1/8)
            normAccCorrectedCorrelationFunction = plot_TH2(
                NormAccCorrectedSEcorr,
                f"Normalized Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{N_{trig} \\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore

            normAccCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png",
                dpi=300,
            )  # type:ignore
            plt.close(normAccCorrectedCorrelationFunction)
    
    @print_function_name_with_description_on_call(description="")
    def plot_normalized_acceptance_corrected_correlation_function_in_z_vertex_bins(self, i, j, k):
        if self.analysisType in ["central", "semicentral"]:
            # get SEcorrs and rebin
            NormAccCorrectedSEcorr = self.NormAccCorrectedSEcorrsZV[i, j, k].Clone()
            #NormAccCorrectedSEcorr.Rebin2D(2, 4)
            #NormAccCorrectedSEcorr.Scale(1/8)
            normAccCorrectedCorrelationFunction = plot_TH2(
                NormAccCorrectedSEcorr,
                f"Normalized Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{N_{trig} \\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore
            normAccCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png",
                dpi=300,
            )  # type:ignore
            plt.close(normAccCorrectedCorrelationFunction)
        elif self.analysisType == "pp":
            # get SEcorrs and rebin
            NormAccCorrectedSEcorr = self.NormAccCorrectedSEcorrsZV[i, j].Clone()
            #NormAccCorrectedSEcorr.Rebin2D(2, 4)
            #NormAccCorrectedSEcorr.Scale(1/8)
            normAccCorrectedCorrelationFunction = plot_TH2(
                NormAccCorrectedSEcorr,
                f"Normalized Acceptance Corrected Correlation Function\n{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\eta$",
                "$\\Delta \\phi$",
                "$\\frac{1}{N_{trig} \\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d^2N_{meas}}{d\\Delta\\eta d\\Delta\\phi}$",
                cmap="jet",
            )  # type:ignore

            normAccCorrectedCorrelationFunction.savefig(
                f"{self.base_save_path}{self.epString}/NormalizedAcceptanceCorrectedCorrelationFunction{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png",
                dpi=300,
            )  # type:ignore
            plt.close(normAccCorrectedCorrelationFunction)

    @print_function_name_with_description_on_call(description="")
    def plot_dPhi_in_background_region(self, i, j, k, axBG, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiBG, dPhiBGax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiBGax = plot_TH1(
                self.dPhiBGcorrs[i, j, k],
                f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$",
                ax=dPhiBGax,
            )  # type:ignore
            x_binsBG = np.array(
                [
                    self.dPhiBGcorrs[i, j, k].GetXaxis().GetBinCenter(b)
                    for b in range(1, self.dPhiBGcorrs[i, j, k].GetNbinsX() + 1)
                ]
            )
            bin_contentBG = np.array(
                [
                    self.dPhiBGcorrs[i, j, k].GetBinContent(b)
                    for b in range(1, self.dPhiBGcorrs[i, j, k].GetNbinsX() + 1)
                ]
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiBGax.fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j, k],
                    bin_contentBG + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiBG.legend()

            dPhiBG.savefig(
                f"{self.base_save_path}{self.epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiBG)
            axBG[k] = plot_TH1(
                self.dPhiBGcorrs[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axBG[k],
            )  # type:ignore
            if plot_ME_systematic:
                axBG[k].fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j, k],
                    bin_contentBG + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
        elif self.analysisType == "pp":
            dPhiBG, dPhiBGax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiBGax = plot_TH1(
                self.dPhiBGcorrs[i, j],
                f"$\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$",
                ax=dPhiBGax,
            )  # type:ignore
            x_binsBG = np.array(
                [
                    self.dPhiBGcorrs[i, j].GetXaxis().GetBinCenter(k)
                    for k in range(self.dPhiBGcorrs[i, j].GetNbinsX())
                ]
            )  # type:ignore
            bin_contentBG = np.array(
                [
                    self.dPhiBGcorrs[i, j].GetBinContent(k)
                    for k in range(1, self.dPhiBGcorrs[i, j].GetNbinsX() + 1)
                ]
            )  # type:ignore
            # do a fill between with self.ME_norm_systematics
            if plot_ME_systematic:
                dPhiBGax.fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j],
                    bin_contentBG + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiBG.legend()
            dPhiBG.savefig(
                f"{self.base_save_path}{self.epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiBG)
            axBG = plot_TH1(
                self.dPhiBGcorrs[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axBG,
            )  # type:ignore
            if plot_ME_systematic:
                axBG.fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j],
                    bin_contentBG + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_dPhi_in_background_region_in_z_vertex_bins(self, i, j, k, axBG, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiBG, dPhiBGax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiBGax = plot_TH1(
                self.dPhiBGcorrsZV[i, j, k],
                f"dPhiBG {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$",
                ax=dPhiBGax,
            )  # type:ignore
            x_binsBG = np.array(
                [
                    self.dPhiBGcorrsZV[i, j, k].GetXaxis().GetBinCenter(b)
                    for b in range(1, self.dPhiBGcorrsZV[i, j, k].GetNbinsX() + 1)
                ]
            )
            bin_contentBG = np.array(
                [
                    self.dPhiBGcorrsZV[i, j, k].GetBinContent(b)
                    for b in range(1, self.dPhiBGcorrsZV[i, j, k].GetNbinsX() + 1)
                ]
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiBGax.fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j, k],
                    bin_contentBG + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiBG.legend()

            dPhiBG.savefig(
                f"{self.base_save_path}{self.epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiBG)
            axBG[k] = plot_TH1(
                self.dPhiBGcorrsZV[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axBG[k],
            )  # type:ignore
            if plot_ME_systematic:
                axBG[k].fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j, k],
                    bin_contentBG + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
        elif self.analysisType == "pp":
            dPhiBG, dPhiBGax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiBGax = plot_TH1(
                self.dPhiBGcorrsZV[i, j],
                f"$\\Delta \\phi$ {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$",
                ax=dPhiBGax,
            )  # type:ignore
            x_binsBG = np.array(
                [
                    self.dPhiBGcorrsZV[i, j].GetXaxis().GetBinCenter(k)
                    for k in range(self.dPhiBGcorrsZV[i, j].GetNbinsX())
                ]
            )  # type:ignore
            bin_contentBG = np.array(
                [
                    self.dPhiBGcorrsZV[i, j].GetBinContent(k)
                    for k in range(1, self.dPhiBGcorrsZV[i, j].GetNbinsX() + 1)
                ]
            )  # type:ignore
            # do a fill between with self.ME_norm_systematics
            if plot_ME_systematic:
                dPhiBGax.fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j],
                    bin_contentBG + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiBG.legend()
            dPhiBG.savefig(
                f"{self.base_save_path}{self.epString}/dPhiBG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiBG)
            axBG = plot_TH1(
                self.dPhiBGcorrsZV[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{BG}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axBG,
            )  # type:ignore
            if plot_ME_systematic:
                axBG.fill_between(
                    x_binsBG,
                    bin_contentBG - self.ME_norm_systematics[i, j],
                    bin_contentBG + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

    @print_function_name_with_description_on_call(description="")
    def plot_dPhi_in_signal_region(self, i, j, k, axSig, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSig, dPhiSigax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigax = plot_TH1(
                self.dPhiSigcorrs[i, j, k],
                f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$",
                ax=dPhiSigax,
            )  # type:ignore
            x_binsSig = np.array(
                [
                    self.dPhiSigcorrs[i, j, k].GetXaxis().GetBinCenter(b)
                    for b in range(1, self.dPhiSigcorrs[i, j, k].GetNbinsX() + 1)
                ]
            )
            bin_contentSig = np.array(
                [
                    self.dPhiSigcorrs[i, j, k].GetBinContent(b)
                    for b in range(1, self.dPhiSigcorrs[i, j, k].GetNbinsX() + 1)
                ]
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigax.fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j, k],
                    bin_contentSig + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSig.legend()
            dPhiSig.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSig)
            axSig[k] = plot_TH1(
                self.dPhiSigcorrs[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSig[k],
            )  # type:ignore
            if plot_ME_systematic:
                axSig[k].fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j, k],
                    bin_contentSig + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
        elif self.analysisType == "pp":
            dPhiSig, dPhiSigax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigax = plot_TH1(
                self.dPhiSigcorrs[i, j],
                f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$",
                ax=dPhiSigax,
            )  # type:ignore
            x_binsSig = np.array(
                [
                    self.dPhiSigcorrs[i, j].GetXaxis().GetBinCenter(k)
                    for k in range(self.dPhiSigcorrs[i, j].GetNbinsX())
                ]
            )  # type:ignore
            bin_contentSig = np.array(
                [
                    self.dPhiSigcorrs[i, j].GetBinContent(k)
                    for k in range(1, self.dPhiSigcorrs[i, j].GetNbinsX() + 1)
                ]
            )  # type:ignore
            # do a fill between with self.ME_norm_systematics
            if plot_ME_systematic:
                dPhiSigax.fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j],
                    bin_contentSig + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSig.legend()
            dPhiSig.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSig)
            axSig = plot_TH1(
                self.dPhiSigcorrs[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSig,
            )  # type:ignore
            if plot_ME_systematic:
                axSig.fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j],
                    bin_contentSig + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
    
    @print_function_name_with_description_on_call(description="")
    def plot_dPhi_in_signal_region_in_z_vertex_bins(self, i, j, k, axSig, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSig, dPhiSigax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigax = plot_TH1(
                self.dPhiSigcorrsZV[i, j, k],
                f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$",
                ax=dPhiSigax,
            )  # type:ignore
            x_binsSig = np.array(
                [
                    self.dPhiSigcorrsZV[i, j, k].GetXaxis().GetBinCenter(b)
                    for b in range(1, self.dPhiSigcorrsZV[i, j, k].GetNbinsX() + 1)
                ]
            )
            bin_contentSig = np.array(
                [
                    self.dPhiSigcorrsZV[i, j, k].GetBinContent(b)
                    for b in range(1, self.dPhiSigcorrsZV[i, j, k].GetNbinsX() + 1)
                ]
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigax.fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j, k],
                    bin_contentSig + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSig.legend()
            dPhiSig.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSig)
            axSig[k] = plot_TH1(
                self.dPhiSigcorrsZV[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSig[k],
            )  # type:ignore
            if plot_ME_systematic:
                axSig[k].fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j, k],
                    bin_contentSig + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
        elif self.analysisType == "pp":
            dPhiSig, dPhiSigax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigax = plot_TH1(
                self.dPhiSigcorrsZV[i, j],
                f"dPhiSig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$",
                ax=dPhiSigax,
            )  # type:ignore
            x_binsSig = np.array(
                [
                    self.dPhiSigcorrsZV[i, j].GetXaxis().GetBinCenter(k)
                    for k in range(self.dPhiSigcorrsZV[i, j].GetNbinsX())
                ]
            )  # type:ignore
            bin_contentSig = np.array(
                [
                    self.dPhiSigcorrsZV[i, j].GetBinContent(k)
                    for k in range(1, self.dPhiSigcorrsZV[i, j].GetNbinsX() + 1)
                ]
            )  # type:ignore
            # do a fill between with self.ME_norm_systematics
            if plot_ME_systematic:
                dPhiSigax.fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j],
                    bin_contentSig + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSig.legend()
            dPhiSig.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSig)
            axSig = plot_TH1(
                self.dPhiSigcorrsZV[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSig,
            )  # type:ignore
            if plot_ME_systematic:
                axSig.fill_between(
                    x_binsSig,
                    bin_contentSig - self.ME_norm_systematics[i, j],
                    bin_contentSig + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

    @print_function_name_with_description_on_call(description="")
    def plot_background_subtracted_dPhi_NS(self, i, j, k, axSigminusBGNS, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNS, dPhiSigminusBGNSax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGNSax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, k],
                f"Background subtracted $\\Delta \\phi$, near side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGNSax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsNS = (
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNS = np.zeros(n_binsNS)
            for l in range(n_binsNS):
                x_binsNS[l] = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )
            bin_contentNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[
                        i, j, k
                    ].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            bin_errorsNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, k].GetBinError(
                        b
                    )
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            RPFErrorNS = np.array(
                [
                    self.RPFObjs[i, j].simultaneous_fit_err(
                        x_binsNS[l], x_binsNS[1] - x_binsNS[0], *self.RPFObjs[i, j].popt
                    )
                    for l in range(len(x_binsNS))
                ]
            )  # type:ignore

            dPhiSigminusBGNSax.fill_between(
                x_binsNS,
                bin_contentNS - RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS - np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                bin_contentNS + RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS + np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGNSax.fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j, k],
                    bin_contentNS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGNS.legend()
            dPhiSigminusBGNS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNS)
            axSigminusBGNS[k] = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGNS[k],
            )  # type:ignore
            axSigminusBGNS[k].fill_between(
                x_binsNS,
                bin_contentNS - RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS - np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                bin_contentNS + RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS + np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNS[k].fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j, k],
                    bin_contentNS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            return n_binsNS, x_binsNS, bin_contentNS, bin_errorsNS, RPFErrorNS
        elif self.analysisType == "pp":
            dPhiSigminusBGNS, dPhiSigminusBGNSax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGNSax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j],
                f"Background subtracted $\\Delta \\phi$, near side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGNSax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j].GetNbinsX()
            x_binsNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j]
                    .GetXaxis()
                    .GetBinCenter(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )
            bin_contentNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )
            bin_errorsNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j].GetBinError(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )
            norm_errorsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminVals[i, j]
            # set pt ep angle bin

            dPhiSigminusBGNSax.fill_between(
                x_binsNS,
                bin_contentNS - norm_errorsNS,
                bin_contentNS + norm_errorsNS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            # also fill between with me norm systematics
            if plot_ME_systematic:
                dPhiSigminusBGNSax.fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j],
                    bin_contentNS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGNS.legend()
            dPhiSigminusBGNS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNS)

            axSigminusBGNS = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGNS,
            )  # type:ignore
            axSigminusBGNS.fill_between(
                x_binsNS,
                bin_contentNS - norm_errorsNS,
                bin_contentNS + norm_errorsNS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNS.fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j],
                    bin_contentNS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            return n_binsNS, x_binsNS, bin_contentNS, bin_errorsNS, norm_errorsNS
    
    @print_function_name_with_description_on_call(description="")
    def plot_background_subtracted_dPhi_NS_in_z_vertex_bins(self, i, j, k, axSigminusBGNS, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNS, dPhiSigminusBGNSax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGNSax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, k],
                f"Background subtracted $\\Delta \\phi$, near side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGNSax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsNS = (
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNS = np.zeros(n_binsNS)
            for l in range(n_binsNS):
                x_binsNS[l] = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )
            bin_contentNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[
                        i, j, k
                    ].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            bin_errorsNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, k].GetBinError(
                        b
                    )
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            RPFErrorNS = np.array(
                [
                    self.RPFObjsZV[i, j].simultaneous_fit_err(
                        x_binsNS[l], x_binsNS[1] - x_binsNS[0], *self.RPFObjsZV[i, j].popt
                    )
                    for l in range(len(x_binsNS))
                ]
            )  # type:ignore

            dPhiSigminusBGNSax.fill_between(
                x_binsNS,
                bin_contentNS - RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS - np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                bin_contentNS + RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS + np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGNSax.fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j, k],
                    bin_contentNS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGNS.legend()
            dPhiSigminusBGNS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNS)
            axSigminusBGNS[k] = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGNS[k],
            )  # type:ignore
            axSigminusBGNS[k].fill_between(
                x_binsNS,
                bin_contentNS - RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS - np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                bin_contentNS + RPFErrorNS[:, k]
                if k != 3
                else bin_contentNS + np.sqrt(np.sum(RPFErrorNS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNS[k].fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j, k],
                    bin_contentNS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            return n_binsNS, x_binsNS, bin_contentNS, bin_errorsNS, RPFErrorNS
        elif self.analysisType == "pp":
            dPhiSigminusBGNS, dPhiSigminusBGNSax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGNSax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j],
                f"Background subtracted $\\Delta \\phi$, near side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{dN_{meas}-N_{BG}}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGNSax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j].GetNbinsX()
            x_binsNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j]
                    .GetXaxis()
                    .GetBinCenter(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )
            bin_contentNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )
            bin_errorsNS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j].GetBinError(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )
            norm_errorsNS = self.BGSubtractedAccCorrectedSEdPhiSigNScorrsminValsZV[i, j]
            # set pt ep angle bin

            dPhiSigminusBGNSax.fill_between(
                x_binsNS,
                bin_contentNS - norm_errorsNS,
                bin_contentNS + norm_errorsNS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            # also fill between with me norm systematics
            if plot_ME_systematic:
                dPhiSigminusBGNSax.fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j],
                    bin_contentNS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGNS.legend()
            dPhiSigminusBGNS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGNS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNS)

            axSigminusBGNS = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGNS,
            )  # type:ignore
            axSigminusBGNS.fill_between(
                x_binsNS,
                bin_contentNS - norm_errorsNS,
                bin_contentNS + norm_errorsNS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNS.fill_between(
                    x_binsNS,
                    bin_contentNS - self.ME_norm_systematics[i, j],
                    bin_contentNS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            return n_binsNS, x_binsNS, bin_contentNS, bin_errorsNS, norm_errorsNS

    @print_function_name_with_description_on_call(description="")
    def plot_background_subtracted_dPhi_AS(self, i, j, k, axSigminusBGAS, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGAS, dPhiSigminusBGASax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGASax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, k],
                f"Background subtracted $\\Delta \\phi$, away side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGASax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsAS = (
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsAS = np.zeros(n_binsAS)
            for l in range(n_binsAS):
                x_binsAS[l] = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )
            bin_contentAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[
                        i, j, k
                    ].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            bin_errorsAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, k].GetBinError(
                        b
                    )
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            RPFErrorAS = np.array(
                [
                    self.RPFObjs[i, j].simultaneous_fit_err(
                        x_binsAS[l], x_binsAS[1] - x_binsAS[0], *self.RPFObjs[i, j].popt
                    )
                    for l in range(len(x_binsAS))
                ]
            )  # type:ignore
            dPhiSigminusBGASax.fill_between(
                x_binsAS,
                bin_contentAS - RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS - np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                bin_contentAS + RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS + np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGASax.fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j, k],
                    bin_contentAS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGAS.legend()
            dPhiSigminusBGAS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGAS)
            axSigminusBGAS[k] = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGAS[k],
            )  # type:ignore
            axSigminusBGAS[k].fill_between(
                x_binsAS,
                bin_contentAS - RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS - np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                bin_contentAS + RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS + np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGAS[k].fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j, k],
                    bin_contentAS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            return n_binsAS, x_binsAS, bin_contentAS, bin_errorsAS, RPFErrorAS
        elif self.analysisType == "pp":
            dPhiSigminusBGAS, dPhiSigminusBGASax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGASax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j],
                f"Background subtracted $\\Delta \\phi$, away side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGASax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsAS = (
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsAS = np.zeros(n_binsAS)
            for l in range(n_binsAS):
                x_binsAS[l] = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
            bin_contentAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            bin_errorsAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j].GetBinError(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            norm_errorsAS = self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminVals[
                i, j
            ]  # type:ignore
            dPhiSigminusBGASax.fill_between(
                x_binsAS,
                bin_contentAS - norm_errorsAS,
                bin_contentAS + norm_errorsAS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            # also fill between with me norm systematics
            if plot_ME_systematic:
                dPhiSigminusBGASax.fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j],
                    bin_contentAS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGAS.legend()

            dPhiSigminusBGAS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGAS)
            axSigminusBGAS = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGAS,
            )  # type:ignore
            axSigminusBGAS.fill_between(
                x_binsAS,
                bin_contentAS - norm_errorsAS,
                bin_contentAS + norm_errorsAS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGAS.fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j],
                    bin_contentAS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            return n_binsAS, x_binsAS, bin_contentAS, bin_errorsAS, norm_errorsAS
    
    @print_function_name_with_description_on_call(description="")
    def plot_background_subtracted_dPhi_AS_in_z_vertex_bins(self, i, j, k, axSigminusBGAS, plot_ME_systematic=False):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGAS, dPhiSigminusBGASax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGASax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, k],
                f"Background subtracted $\\Delta \\phi$, away side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGASax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsAS = (
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsAS = np.zeros(n_binsAS)
            for l in range(n_binsAS):
                x_binsAS[l] = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )
            bin_contentAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[
                        i, j, k
                    ].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            bin_errorsAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, k].GetBinError(
                        b
                    )
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[
                            i, j, k
                        ].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            RPFErrorAS = np.array(
                [
                    self.RPFObjsZV[i, j].simultaneous_fit_err(
                        x_binsAS[l], x_binsAS[1] - x_binsAS[0], *self.RPFObjsZV[i, j].popt
                    )
                    for l in range(len(x_binsAS))
                ]
            )  # type:ignore
            dPhiSigminusBGASax.fill_between(
                x_binsAS,
                bin_contentAS - RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS - np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                bin_contentAS + RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS + np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGASax.fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j, k],
                    bin_contentAS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGAS.legend()
            dPhiSigminusBGAS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGAS)
            axSigminusBGAS[k] = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, k],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGAS[k],
            )  # type:ignore
            axSigminusBGAS[k].fill_between(
                x_binsAS,
                bin_contentAS - RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS - np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                bin_contentAS + RPFErrorAS[:, k]
                if k != 3
                else bin_contentAS + np.sqrt(np.sum(RPFErrorAS**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGAS[k].fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j, k],
                    bin_contentAS + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            return n_binsAS, x_binsAS, bin_contentAS, bin_errorsAS, RPFErrorAS
        elif self.analysisType == "pp":
            dPhiSigminusBGAS, dPhiSigminusBGASax = plt.subplots(1, 1, figsize=(10, 6))
            dPhiSigminusBGASax = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j],
                f"Background subtracted $\\Delta \\phi$, away side, {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$",
                ax=dPhiSigminusBGASax,
            )  # type:ignore
            # add fit uncertainties as fill_between
            n_binsAS = (
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsAS = np.zeros(n_binsAS)
            for l in range(n_binsAS):
                x_binsAS[l] = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
            bin_contentAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j].GetBinContent(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            bin_errorsAS = np.array(
                [
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j].GetBinError(b)
                    for b in range(
                        1,
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j].GetNbinsX()
                        + 1,
                    )
                ]
            )  # type:ignore
            norm_errorsAS = self.BGSubtractedAccCorrectedSEdPhiSigAScorrsminValsZV[
                i, j
            ]  # type:ignore
            dPhiSigminusBGASax.fill_between(
                x_binsAS,
                bin_contentAS - norm_errorsAS,
                bin_contentAS + norm_errorsAS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            # also fill between with me norm systematics
            if plot_ME_systematic:
                dPhiSigminusBGASax.fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j],
                    bin_contentAS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGAS.legend()

            dPhiSigminusBGAS.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BGAS{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGAS)
            axSigminusBGAS = plot_TH1(
                self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j],
                f"{self.epString}",
                "$\\Delta \\phi$",
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else "",
                ax=axSigminusBGAS,
            )  # type:ignore
            axSigminusBGAS.fill_between(
                x_binsAS,
                bin_contentAS - norm_errorsAS,
                bin_contentAS + norm_errorsAS,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGAS.fill_between(
                    x_binsAS,
                    bin_contentAS - self.ME_norm_systematics[i, j],
                    bin_contentAS + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            return n_binsAS, x_binsAS, bin_contentAS, bin_errorsAS, norm_errorsAS

    @print_function_name_with_description_on_call(description="")
    def plot_background_subtracted_dPhi_INC(
        self,
        i,
        j,
        k,
        axSigminusBGINC,
        n_binsNS,
        x_binsNS,
        bin_contentNS,
        bin_errorsNS,
        BGErrorNS,
        n_binsAS,
        x_binsAS,
        bin_contentAS,
        bin_errorsAS,
        BGErrorAS,
        plot_ME_systematic=False,
    ):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGINC, dPhiSigminusBGINCax = plt.subplots(1, 1, figsize=(10, 6))
            # type:ignore
            # add fit uncertainties as fill_between
            n_binsINC = n_binsNS + n_binsAS
            x_binsINC = np.zeros(n_binsINC)
            for l in range(n_binsINC):
                x_binsINC[l] = x_binsNS[l] if l < n_binsNS else x_binsAS[l - n_binsNS]
            bin_contentINC = np.array(
                [
                    bin_contentNS[l] if l < n_binsNS else bin_contentAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )
            bin_errorsINC = np.array(
                [
                    bin_errorsNS[l] if l < n_binsNS else bin_errorsAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            ) 
            RPFErrorINC = np.array(
                [
                    BGErrorNS[l] if l < n_binsNS else BGErrorAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )  # type:ignore
            dPhiSigminusBGINCax.errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            dPhiSigminusBGINCax.fill_between(
                x_binsINC,
                bin_contentINC - RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC - np.sqrt(np.sum(RPFErrorINC**2, axis=1)),
                bin_contentINC + RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC + np.sqrt(np.sum(RPFErrorINC**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGINCax.fill_between(
                    x_binsINC,
                    bin_contentINC - self.ME_norm_systematics[i, j, k],
                    bin_contentINC + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGINC.legend()
            dPhiSigminusBGINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGINCax.set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            dPhiSigminusBGINCax.set_title(
                f"Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )
            dPhiSigminusBGINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGINC)
            axSigminusBGINC[(k+1)%4].errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            axSigminusBGINC[(k+1)%4].fill_between(
                x_binsINC,
                bin_contentINC - RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC - np.sqrt(np.sum(RPFErrorINC**2, axis=1) ),
                bin_contentINC + RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC + np.sqrt(np.sum(RPFErrorINC**2, axis=1) ),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGINC[(k+1)%4].fill_between(
                    x_binsINC,
                    bin_contentINC - self.ME_norm_systematics[i, j, k],
                    bin_contentINC + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )

            axSigminusBGINC[(k+1)%4].set_xlabel("$\\Delta \\phi$")
            axSigminusBGINC[k].set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            return RPFErrorINC
        elif self.analysisType == "pp":
            dPhiSigminusBGINC, dPhiSigminusBGINCax = plt.subplots(1, 1, figsize=(10, 6))
            # type:ignore
            # add fit uncertainties as fill_between
            n_binsINC = n_binsNS + n_binsAS
            x_binsINC = np.zeros(n_binsINC)
            for l in range(n_binsINC):
                x_binsINC[l] = x_binsNS[l] if l < n_binsNS else x_binsAS[l - n_binsNS]
            bin_contentINC = np.array(
                [
                    bin_contentNS[l] if l < n_binsNS else bin_contentAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )
            bin_errorsINC = np.array(
                [
                    bin_errorsNS[l] if l < n_binsNS else bin_errorsAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )
            norm_errorsINC = (BGErrorNS + BGErrorAS) / 2

            dPhiSigminusBGINCax.errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            dPhiSigminusBGINCax.fill_between(
                x_binsINC,
                bin_contentINC - norm_errorsINC,
                bin_contentINC + norm_errorsINC,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            # also fill between with me norm systematics
            if plot_ME_systematic:
                dPhiSigminusBGINCax.fill_between(
                    x_binsINC,
                    bin_contentINC - self.ME_norm_systematics[i, j],
                    bin_contentINC + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGINC.legend()

            dPhiSigminusBGINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGINCax.set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            dPhiSigminusBGINCax.set_title(
                f"Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )  # type:ignore
            dPhiSigminusBGINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGINC)
            axSigminusBGINC.errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            axSigminusBGINC.fill_between(
                x_binsINC,
                bin_contentINC - norm_errorsINC,
                bin_contentINC + norm_errorsINC,
                alpha=0.3,
                color="red",
            )

            axSigminusBGINC.set_xlabel("$\\Delta \\phi$")
            axSigminusBGINC.set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
    
    @print_function_name_with_description_on_call(description="")
    def plot_background_subtracted_dPhi_INC_in_z_vertex_bins(
        self,
        i,
        j,
        k,
        axSigminusBGINC,
        n_binsNS,
        x_binsNS,
        bin_contentNS,
        bin_errorsNS,
        BGErrorNS,
        n_binsAS,
        x_binsAS,
        bin_contentAS,
        bin_errorsAS,
        BGErrorAS,
        plot_ME_systematic=False,
    ):
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGINC, dPhiSigminusBGINCax = plt.subplots(1, 1, figsize=(10, 6))
            # type:ignore
            # add fit uncertainties as fill_between
            n_binsINC = n_binsNS + n_binsAS
            x_binsINC = np.zeros(n_binsINC)
            for l in range(n_binsINC):
                x_binsINC[l] = x_binsNS[l] if l < n_binsNS else x_binsAS[l - n_binsNS]
            bin_contentINC = np.array(
                [
                    bin_contentNS[l] if l < n_binsNS else bin_contentAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            ) 
            bin_errorsINC = np.array(
                [
                    bin_errorsNS[l] if l < n_binsNS else bin_errorsAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            ) 
            RPFErrorINC = np.array(
                [
                    BGErrorNS[l] if l < n_binsNS else BGErrorAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )   # type:ignore
            dPhiSigminusBGINCax.errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            dPhiSigminusBGINCax.fill_between(
                x_binsINC,
                bin_contentINC - RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC - np.sqrt(np.sum(RPFErrorINC**2, axis=1)),
                bin_contentINC + RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC + np.sqrt(np.sum(RPFErrorINC**2, axis=1)),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGINCax.fill_between(
                    x_binsINC,
                    bin_contentINC - self.ME_norm_systematics[i, j, k],
                    bin_contentINC + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGINC.legend()
            dPhiSigminusBGINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGINCax.set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            dPhiSigminusBGINCax.set_title(
                f"Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )
            dPhiSigminusBGINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGINC)
            axSigminusBGINC[(k+1)%4].errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            axSigminusBGINC[(k+1)%4].fill_between(
                x_binsINC,
                bin_contentINC - RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC - np.sqrt(np.sum(RPFErrorINC**2, axis=1)),
                bin_contentINC + RPFErrorINC[:, k]
                if k != 3
                else bin_contentINC + np.sqrt(np.sum(RPFErrorINC**2, axis=1) ),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGINC[(k+1)%4].fill_between(
                    x_binsINC,
                    bin_contentINC - self.ME_norm_systematics[i, j, k],
                    bin_contentINC + self.ME_norm_systematics[i, j, k],
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )

            axSigminusBGINC[(k+1)%4].set_xlabel("$\\Delta \\phi$")
            axSigminusBGINC[k].set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            return RPFErrorINC
        elif self.analysisType == "pp":
            dPhiSigminusBGINC, dPhiSigminusBGINCax = plt.subplots(1, 1, figsize=(10, 6))
            # type:ignore
            # add fit uncertainties as fill_between
            n_binsINC = n_binsNS + n_binsAS
            x_binsINC = np.zeros(n_binsINC)
            for l in range(n_binsINC):
                x_binsINC[l] = x_binsNS[l] if l < n_binsNS else x_binsAS[l - n_binsNS]
            bin_contentINC = np.array(
                [
                    bin_contentNS[l] if l < n_binsNS else bin_contentAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )
            bin_errorsINC = np.array(
                [
                    bin_errorsNS[l] if l < n_binsNS else bin_errorsAS[l - n_binsNS]
                    for l in range(n_binsINC)
                ]
            )
            norm_errorsINC = (BGErrorNS + BGErrorAS) / 2

            dPhiSigminusBGINCax.errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            dPhiSigminusBGINCax.fill_between(
                x_binsINC,
                bin_contentINC - norm_errorsINC,
                bin_contentINC + norm_errorsINC,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            # also fill between with me norm systematics
            if plot_ME_systematic:
                dPhiSigminusBGINCax.fill_between(
                    x_binsINC,
                    bin_contentINC - self.ME_norm_systematics[i, j],
                    bin_contentINC + self.ME_norm_systematics[i, j],
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGINC.legend()

            dPhiSigminusBGINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGINCax.set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            dPhiSigminusBGINCax.set_title(
                f"Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )  # type:ignore
            dPhiSigminusBGINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGINC)
            axSigminusBGINC.errorbar(
                x_binsINC, bin_contentINC, yerr=bin_errorsINC, fmt="o", color="black"
            )
            axSigminusBGINC.fill_between(
                x_binsINC,
                bin_contentINC - norm_errorsINC,
                bin_contentINC + norm_errorsINC,
                alpha=0.3,
                color="red",
            )

            axSigminusBGINC.set_xlabel("$\\Delta \\phi$")
            axSigminusBGINC.set_ylabel(
                "$\\frac{1}{\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )

    @print_function_name_with_description_on_call(description="")
    def plot_normalized_background_subtracted_dPhi_INC(
        self, i, j, k, axSigminusBGNormINC, BGErrorINC, plot_ME_systematic=False
    ):
        if self.analysisType in ["central", "semicentral"]:
           N_trig = self.N_trigs[i,k]
           print(f"N_trig = {N_trig}, for k={k}")
        elif self.analysisType == "pp":
            N_trig = self.N_trigs[i]

        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = (
                self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINC[
                    l
                ] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINC[
                    l
                ] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore

            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum((BGErrorINC) ** 2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / N_trig,
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig,
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ (rad)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ (rad)$^{-1}$"
                if k == 0
                else ""
            )
            dPhiSigminusBGNormINCax.set_title(
                f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINC[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINC[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / N_trig,
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNormINC[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig,
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )

            axSigminusBGNormINC[(k+1)%4].set_xlabel("$\\Delta \\phi$ (rad)")
            axSigminusBGNormINC[k].set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ (rad)$^{-1}$"
                if k == 0
                else ""
            )

        elif self.analysisType == "pp":
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                i, j
            ].GetNbinsX()  # type:ignore
            x_binsNormINC = np.array(
                [
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                        i, j
                    ].GetBinCenter(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_contentNormINC = np.array(
                [
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                        i, j
                    ].GetBinContent(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_errorsNormINC = np.array(
                [
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                        i, j
                    ].GetBinError(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            norm_errorsNormINC = (
                self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[i, j]
            )  # type:ignore
            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - norm_errorsNormINC / N_trig,
                bin_contentNormINC + norm_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            # dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            dPhiSigminusBGNormINCax.set_title(
                f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINC.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINC.fill_between(
                x_binsNormINC,
                bin_contentNormINC - norm_errorsNormINC / N_trig,
                bin_contentNormINC + norm_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNormINC.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            axSigminusBGNormINC.set_xlabel("$\\Delta \\phi$")
            axSigminusBGNormINC.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
            )
            # axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    @print_function_name_with_description_on_call(description="")
    def plot_normalized_background_subtracted_dPhi_INC_in_z_vertex_bins(
        self, i, j, k, axSigminusBGNormINC, BGErrorINC, plot_ME_systematic=False
    ):
        if self.analysisType in ["central", "semicentral"]:
           N_trig = self.N_trigs[i,k]
        elif self.analysisType == "pp":
            N_trig = self.N_trigs[i]

        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = (
                self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINC = np.zeros(n_binsNormINC)  # type:ignore

            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINC[
                    l
                ] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINC[
                    l
                ] = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore

            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum((BGErrorINC) ** 2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / N_trig,
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig,
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            dPhiSigminusBGNormINCax.set_title(
                f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINC[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINC[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / N_trig,
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNormINC[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig,
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )

            axSigminusBGNormINC[(k+1)%4].set_xlabel("$\\Delta \\phi$")
            axSigminusBGNormINC[k].set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )

        elif self.analysisType == "pp":
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[
                i, j
            ].GetNbinsX()  # type:ignore
            x_binsNormINC = np.array(
                [
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[
                        i, j
                    ].GetBinCenter(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_contentNormINC = np.array(
                [
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[
                        i, j
                    ].GetBinContent(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_errorsNormINC = np.array(
                [
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[
                        i, j
                    ].GetBinError(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            norm_errorsNormINC = (
                self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV[i, j]
            )  # type:ignore
            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - norm_errorsNormINC / N_trig,
                bin_contentNormINC + norm_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
                if k == 0
                else ""
            )
            # dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            dPhiSigminusBGNormINCax.set_title(
                f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINC.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINC.fill_between(
                x_binsNormINC,
                bin_contentNormINC - norm_errorsNormINC / N_trig,
                bin_contentNormINC + norm_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNormINC.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore
            axSigminusBGNormINC.set_xlabel("$\\Delta \\phi$")
            axSigminusBGNormINC.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$"
            )
            # axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    @print_function_name_with_description_on_call(description="")
    def plot_normalized_background_subtracted_dPhi_for_true_species(
        self, i, j, k, species, axSigminusBGNormINCSpecies, BGErrorINC, plot_ME_systematic=True, plot_PID_systematic=True
    ):
        
        if self.analysisType in ["central", "semicentral"]:
            N_trig = self.N_trigs[i,k]
        elif self.analysisType=='pp':
            N_trig = self.N_trigs[i]
            plot_ME_systematic=False
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = (
                self.NormalizedBGSubtracteddPhiForTrueSpecies[species][i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINC = np.zeros(n_binsNormINC)  # type:ignore

            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    self.NormalizedBGSubtracteddPhiForTrueSpecies[species][i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINC[l] = self.NormalizedBGSubtracteddPhiForTrueSpecies[species][
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINC[l] = self.NormalizedBGSubtracteddPhiForTrueSpecies[species][
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore

            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum((BGErrorINC) ** 2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                bin_contentDiff = np.zeros(n_binsNormINC)  # type:ignore

                for l in range(n_binsNormINC):
                    bin_contentDiff[l] = self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                        i, j, k
                    ].GetBinContent(
                        l + 1
                    )  # type:ignore
                bin_contentDiff = np.abs(bin_contentDiff - bin_contentNormINC)
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC +bin_contentDiff,
                    color="red",
                    alpha=0.3,
                    label="ME systematic",
                )
            
            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                elif self.analysisType=='pp':
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    BGLevel = self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])-BGLevel

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )
            dPhiSigminusBGNormINCax.set_title(
                f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC + bin_contentDiff,
                    color="red",
                    alpha=0.3,
                    label="ME systematic",
                )

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )

            axSigminusBGNormINCSpecies[(k+1)%4].set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies[k].set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )

        elif self.analysisType == "pp":
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpecies[species][
                i, j
            ].GetNbinsX()  # type:ignore
            x_binsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForTrueSpecies[species][
                        i, j
                    ].GetBinCenter(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_contentNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForTrueSpecies[species][
                        i, j
                    ].GetBinContent(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_errorsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForTrueSpecies[species][
                        i, j
                    ].GetBinError(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals[
                species
            ][
                i, j
            ]  # type:ignore
            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                bin_contentDiff = np.zeros(n_binsNormINC)  # type:ignore

                for l in range(n_binsNormINC):
                    bin_contentDiff[l] = self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                        i, j
                    ].GetBinContent(
                        l + 1
                    )  # type:ignore
                bin_contentDiff = np.abs(bin_contentDiff - bin_contentNormINC)

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC + bin_contentDiff,
                    alpha=0.3,
                    color="red",
                    label="ME Systematic",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )
            # dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            dPhiSigminusBGNormINCax.set_title(
                f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                
                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC + bin_contentDiff,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            axSigminusBGNormINCSpecies.set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
            )
            # axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    @print_function_name_with_description_on_call(description="")
    def plot_normalized_background_subtracted_dPhi_for_true_species_in_z_vertex_bins(
        self, i, j, k, species, axSigminusBGNormINCSpecies, BGErrorINC, plot_ME_systematic=False, plot_PID_systematic=True
    ):
        if self.analysisType in ["central", "semicentral"]:
            N_trig = self.N_trigs[i,k]
        elif self.analysisType=='pp':
            N_trig = self.N_trigs[i]
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = (
                self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINC = np.zeros(n_binsNormINC)  # type:ignore

            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINC[l] = self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINC[l] = self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore

            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum((BGErrorINC) ** 2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 - self.ME_norm_systematics[i, j, k]/(N_trig),
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 + self.ME_norm_systematics[i, j, k]/(N_trig),
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            
            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )

            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )
            dPhiSigminusBGNormINCax.set_title(
                f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 - self.ME_norm_systematics[i, j, k]/(N_trig),
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 + self.ME_norm_systematics[i, j, k]/(N_trig),
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            axSigminusBGNormINCSpecies[(k+1)%4].set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies[k].set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )

        elif self.analysisType == "pp":
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                i, j
            ].GetNbinsX()  # type:ignore
            x_binsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                        i, j
                    ].GetBinCenter(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_contentNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                        i, j
                    ].GetBinContent(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_errorsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForTrueSpeciesZV[species][
                        i, j
                    ].GetBinError(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminValsZV[
                species
            ][
                i, j
            ]  # type:ignore
            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad{-1}$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad{-1}$)"
                if k == 0
                else ""
            )
            # dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            dPhiSigminusBGNormINCax.set_title(
                f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}ZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            axSigminusBGNormINCSpecies.set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
            )
            # axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    @print_function_name_with_description_on_call(description="")
    def plot_normalized_background_subtracted_dPhi_for_enhanced_species(
        self, i, j, k, species, axSigminusBGNormINCSpecies, BGErrorINC, plot_ME_systematic=True, plot_PID_systematic=False
    ):
        
        if self.analysisType in ["central", "semicentral"]:
            N_trig = self.N_trigs[i,k]
        elif self.analysisType=='pp':
            N_trig = self.N_trigs[i]
            plot_ME_systematic=False
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = (
                self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINC = np.zeros(n_binsNormINC)  # type:ignore

            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINC[l] = self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINC[l] = self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore

            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum((BGErrorINC) ** 2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                bin_contentDiff = np.zeros(n_binsNormINC)  # type:ignore

                for l in range(n_binsNormINC):
                    bin_contentDiff[l] = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                        i, j, k
                    ].GetBinContent(
                        l + 1
                    )  # type:ignore
                bin_contentDiff = np.abs(bin_contentDiff - bin_contentNormINC)
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC +bin_contentDiff,
                    color="red",
                    alpha=0.3,
                    label="ME systematic",
                )
            
            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                elif self.analysisType=='pp':
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )
            dPhiSigminusBGNormINCax.set_title(
                f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}Enh.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC + bin_contentDiff,
                    color="red",
                    alpha=0.3,
                    label="ME systematic",
                )

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )

            axSigminusBGNormINCSpecies[(k+1)%4].set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies[k].set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )

        elif self.analysisType == "pp":
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][
                i, j
            ].GetNbinsX()  # type:ignore
            x_binsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][
                        i, j
                    ].GetBinCenter(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_contentNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][
                        i, j
                    ].GetBinContent(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_errorsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForEnhancedSpecies[species][
                        i, j
                    ].GetBinError(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminVals[
                species
            ][
                i, j
            ]  # type:ignore
            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                bin_contentDiff = np.zeros(n_binsNormINC)  # type:ignore

                for l in range(n_binsNormINC):
                    bin_contentDiff[l] = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                        i, j
                    ].GetBinContent(
                        l + 1
                    )  # type:ignore
                bin_contentDiff = np.abs(bin_contentDiff - bin_contentNormINC)

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC + bin_contentDiff,
                    alpha=0.3,
                    color="red",
                    label="ME Systematic",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )
            # dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            dPhiSigminusBGNormINCax.set_title(
                f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}Enh.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                
                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - bin_contentDiff,
                    bin_contentNormINC + bin_contentDiff,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpecies[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            axSigminusBGNormINCSpecies.set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
            )
            # axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    @print_function_name_with_description_on_call(description="")
    def plot_normalized_background_subtracted_dPhi_for_enhanced_species_in_z_vertex_bins(
        self, i, j, k, species, axSigminusBGNormINCSpecies, BGErrorINC, plot_ME_systematic=False, plot_PID_systematic=False
    ):
        if self.analysisType in ["central", "semicentral"]:
            N_trig = self.N_trigs[i,k]
        elif self.analysisType=='pp':
            N_trig = self.N_trigs[i]
        if self.analysisType in ["central", "semicentral"]:
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = (
                self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][i, j, k]
                .GetXaxis()
                .GetNbins()
            )  # type:ignore
            x_binsNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_contentNormINC = np.zeros(n_binsNormINC)  # type:ignore
            bin_errorsNormINC = np.zeros(n_binsNormINC)  # type:ignore

            for l in range(n_binsNormINC):
                x_binsNormINC[l] = (
                    self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][i, j, k]
                    .GetXaxis()
                    .GetBinCenter(l + 1)
                )  # type:ignore
                bin_contentNormINC[l] = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                    i, j, k
                ].GetBinContent(
                    l + 1
                )  # type:ignore
                bin_errorsNormINC[l] = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                    i, j, k
                ].GetBinError(
                    l + 1
                )  # type:ignore

            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum((BGErrorINC) ** 2, axis=1) ) / N_trig,
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 - self.ME_norm_systematics[i, j, k]/(N_trig),
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 + self.ME_norm_systematics[i, j, k]/(N_trig),
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )
            
            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )

            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )
            dPhiSigminusBGNormINCax.set_title(
                f"Normalized Background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}EnhZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies[(k+1)%4].errorbar(
                x_binsNormINC,
                bin_contentNormINC if k!=3 else bin_contentNormINC*3,
                yerr=bin_errorsNormINC if k!=3 else bin_errorsNormINC*3,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                x_binsNormINC,
                bin_contentNormINC - BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                - np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                bin_contentNormINC + BGErrorINC[:, k] / N_trig
                if k != 3
                else bin_contentNormINC*3
                + np.sqrt(np.sum(BGErrorINC**2, axis=1) ) / (N_trig),
                alpha=0.3,
                color="green",
                label="RPF fit",
            )
            # do fill between for me norm systematic
            if plot_ME_systematic:
                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 - self.ME_norm_systematics[i, j, k]/(N_trig),
                    bin_contentNormINC + self.ME_norm_systematics[i, j, k]/N_trig if k!=3 else bin_contentNormINC*3 + self.ME_norm_systematics[i, j, k]/(N_trig),
                    color="red",
                    alpha=0.3,
                    label="ME normalization",
                )

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies[(k+1)%4].fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig if k!=3 else bin_contentNormINC*3 + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            axSigminusBGNormINCSpecies[(k+1)%4].set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies[k].set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
                if k == 0
                else ""
            )

        elif self.analysisType == "pp":
            dPhiSigminusBGNormINC, dPhiSigminusBGNormINCax = plt.subplots(
                1, 1, figsize=(10, 6)
            )  # type:ignore
            n_binsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                i, j
            ].GetNbinsX()  # type:ignore
            x_binsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                        i, j
                    ].GetBinCenter(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_contentNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                        i, j
                    ].GetBinContent(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            bin_errorsNormINC = np.array(
                [
                    self.NormalizedBGSubtracteddPhiForEnhancedSpeciesZV[species][
                        i, j
                    ].GetBinError(b)
                    for b in range(1, n_binsNormINC + 1)
                ]
            )  # type:ignore
            BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminValsZV[
                species
            ][
                i, j
            ]  # type:ignore
            dPhiSigminusBGNormINCax.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            dPhiSigminusBGNormINCax.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                dPhiSigminusBGNormINCax.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            dPhiSigminusBGNormINC.legend()
            dPhiSigminusBGNormINCax.set_xlabel("$\\Delta \\phi$ ($rad{-1}$)")
            dPhiSigminusBGNormINCax.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad{-1}$)"
                if k == 0
                else ""
            )
            # dPhiSigminusBGNormINCax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            dPhiSigminusBGNormINCax.set_title(
                f"Normalized background subtracted $\\Delta \\phi${self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV/c, {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV/c, {self.epString}, {species}"
            )  # type:ignore
            dPhiSigminusBGNormINC.tight_layout()
            dPhiSigminusBGNormINC.savefig(
                f"{self.base_save_path}{self.epString}/dPhiSig-BG-Norm-{species}{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}_{self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]}EnhZV.png"
            )  # type:ignore
            plt.close(dPhiSigminusBGNormINC)
            axSigminusBGNormINCSpecies.errorbar(
                x_binsNormINC,
                bin_contentNormINC,
                yerr=bin_errorsNormINC,
                fmt="o",
                color="black",
            )
            axSigminusBGNormINCSpecies.fill_between(
                x_binsNormINC,
                bin_contentNormINC - BG_errorsNormINC / N_trig,
                bin_contentNormINC + BG_errorsNormINC / N_trig,
                alpha=0.3,
                color="orange",
                label="Yield normalization",
            )
            if plot_ME_systematic:
                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - self.ME_norm_systematics[i, j]/N_trig,
                    bin_contentNormINC + self.ME_norm_systematics[i, j]/N_trig,
                    alpha=0.3,
                    color="red",
                    label="ME normalization",
                )  # type:ignore

            if plot_PID_systematic:
                if self.analysisType in ["central", "semicentral"]:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j,k]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])
                else:
                    dPhiErr = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,j]
                    dPhiErr = np.array([dPhiErr.GetBinContent(l+1) for l in range(n_binsNormINC)])

                axSigminusBGNormINCSpecies.fill_between(
                    x_binsNormINC,
                    bin_contentNormINC - dPhiErr/N_trig,
                    bin_contentNormINC + dPhiErr/N_trig,
                    color="blue",
                    alpha=0.3,
                    label="PID Error"
                )
            axSigminusBGNormINCSpecies.set_xlabel("$\\Delta \\phi$ ($rad$)")
            axSigminusBGNormINCSpecies.set_ylabel(
                "$\\frac{1}{N_{trig}\\epsilon a(\\Delta \\eta \\Delta\\phi)}\\frac{d(N_{meas}-N_{BG})}{d\\Delta\\phi}$ ($rad^{-1}$)"
            )
            # axSigminusBGNormINC.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    @print_function_name_with_description_on_call(description="")
    def plot_inclusive_yield(self, plot_ME_systematic=True,):

        panel_text1 = "ALICE Work In Progress" + "\n" + f"{'pp' if self.analysisType=='pp' else 'Pb-Pb'}" + r"$\sqrt{s_{NN}}$=5.02TeV"+ f", {'30-50%' if self.analysisType=='semicentral' else '0-10%' if self.analysisType=='central' else ''}"+ "\n"  + "anti-$k_T$ R=0.2"
        panel_text2 =  f"\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
        x_locs = [0.25, 0.4]
        y_locs = [0.75, 0.75]
        font_size = 16

        yieldFig, axYield = plt.subplots(1, 1, figsize=(10, 6))
        axYield.set_title(f"{'Inclusive'} Yield")
        axYield.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYield.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%('Inclusive', 'Inclusive'))
        # set log y
        axYield.set_yscale("log")
        if self.analysisType=='pp':
            plot_ME_systematic=False
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            axYield.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrue['inclusive'][i, :],
                self.YieldErrsTrue['inclusive'][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"Jet p_T {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            N_trigs = N_trigs
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                
                n_bins = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                per_bin_RPF_error = np.zeros((n_bins, len(self.pTassocBinCenters)))
                for l in range(n_bins):
                    x_bins[l] = (
                        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error[l] = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore


                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrue['inclusive'][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrue['inclusive'][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="green",
                    label="RPF systematics" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrue['inclusive'][i, :]-self.YieldsTrueZV['inclusive'][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
               
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrue['inclusive'][i, :]-self.YieldsTrueZV['inclusive'][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 1 else "",
                    )  # type:ignore

                
        axYield.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYield.transAxes)
        axYield.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYield.transAxes)

        axYield.legend()
        yieldFig.tight_layout()
        yieldFig.savefig(f"{self.base_save_path}Yield_{'inclusive'}.png")  # type:ignore
        plt.close(yieldFig)

        yieldFigNS, axYieldNS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldNS.set_title(f"Near side {'Inclusive'} Yield")
        axYieldNS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldNS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%('Inclusive', 'Inclusive'))
        # set log y
        axYieldNS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldNS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueNS['inclusive'][i, :],
                self.YieldErrsTrueNS['inclusive'][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore

                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueNS['inclusive'][i, :]-self.YieldsTrueNSZV['inclusive'][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
               
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueNS['inclusive'][i, :]-self.YieldsTrueNSZV['inclusive'][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

        axYieldNS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldNS.transAxes)
        axYieldNS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldNS.transAxes)
        
        axYieldNS.legend()
        yieldFigNS.tight_layout()
        yieldFigNS.savefig(f"{self.base_save_path}YieldNS_{'inclusive'}.png")  # type:ignore
        plt.close(yieldFigNS)

        yieldFigAS, axYieldAS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldAS.set_title(f"Away Side {'inclusive'} Yield")
        axYieldAS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldAS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)${-1}$"%('inclusive', 'inclusive'))
        # set log y
        axYieldAS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldAS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueAS['inclusive'][i, :],
                self.YieldErrsTrueAS['inclusive'][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore

                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueAS['inclusive'][i, :]-self.YieldsTrueASZV['inclusive'][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                   
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminVals[
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueAS['inclusive'][i, :]-self.YieldsTrueASZV['inclusive'][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

               
        axYieldAS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldAS.transAxes)
        axYieldAS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldAS.transAxes)

        axYieldAS.legend()
        yieldFigAS.tight_layout()
        yieldFigAS.savefig(f"{self.base_save_path}YieldAS_{'inclusive'}.png")  # type:ignore
        plt.close(yieldFigAS)
    
    @print_function_name_with_description_on_call(description="")
    def plot_inclusive_yield_in_z_vertex_bins(self, plot_ME_systematic=False,):

        panel_text1 = "ALICE Work In Progress" + "\n" + f"{'pp' if self.analysisType=='pp' else 'Pb-Pb'}" + r"$\sqrt{s_{NN}}$=5.02TeV"+ f", {'30-50%' if self.analysisType=='semicentral' else '0-10%' if self.analysisType=='central' else ''}"+ "\n"  + "anti-$k_T$ R=0.2"
        panel_text2 =  f"\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
        x_locs = [0.25, 0.4]
        y_locs = [0.75, 0.75]
        font_size = 16

        yieldFig, axYield = plt.subplots(1, 1, figsize=(10, 6))
        axYield.set_title(f"{'Inclusive'} Yield")
        axYield.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYield.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%('Inclusive', 'Inclusive'))
        # set log y
        axYield.set_yscale("log")
        if self.analysisType=='pp':
            plot_ME_systematic=False
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            axYield.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueZV['inclusive'][i, :],
                self.YieldErrsTrueZV['inclusive'][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"Jet p_T {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            N_trigs = N_trigs
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                
                n_bins = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                per_bin_RPF_error = np.zeros((n_bins, len(self.pTassocBinCenters)))
                for l in range(n_bins):
                    x_bins[l] = (
                        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error[l] = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore


                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueZV['inclusive'][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueZV['inclusive'][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="green",
                    label="RPF systematics" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrue['inclusive'][i, :]-self.YieldsTrueZV['inclusive'][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrue['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
               
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV[
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueZV['inclusive'][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueZV['inclusive'][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrue['inclusive'][i, :]-self.YieldsTrueZV['inclusive'][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueZV['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueZV['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 1 else "",
                    )  # type:ignore

                
        axYield.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYield.transAxes)
        axYield.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYield.transAxes)

        axYield.legend()
        yieldFig.tight_layout()
        yieldFig.savefig(f"{self.base_save_path}Yield_{'inclusive'}ZV.png")  # type:ignore
        plt.close(yieldFig)

        yieldFigNS, axYieldNS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldNS.set_title(f"Near side {'Inclusive'} Yield")
        axYieldNS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldNS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%('Inclusive', 'Inclusive'))
        # set log y
        axYieldNS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldNS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueNSZV['inclusive'][i, :],
                self.YieldErrsTrueNSZV['inclusive'][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNSZV['inclusive'][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueNSZV['inclusive'][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore

                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueNS['inclusive'][i, :]-self.YieldsTrueNSZV['inclusive'][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
               
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV[
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNSZV['inclusive'][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueNSZV['inclusive'][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueNS['inclusive'][i, :]-self.YieldsTrueNSZV['inclusive'][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueNS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

        axYieldNS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldNS.transAxes)
        axYieldNS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldNS.transAxes)
        
        axYieldNS.legend()
        yieldFigNS.tight_layout()
        yieldFigNS.savefig(f"{self.base_save_path}YieldNS_{'inclusive'}ZV.png")  # type:ignore
        plt.close(yieldFigNS)

        yieldFigAS, axYieldAS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldAS.set_title(f"Away Side {'inclusive'} Yield")
        axYieldAS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldAS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)${-1}$"%('inclusive', 'inclusive'))
        # set log y
        axYieldAS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldAS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueASZV['inclusive'][i, :],
                self.YieldErrsTrueASZV['inclusive'][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore

                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueASZV['inclusive'][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueASZV['inclusive'][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueAS['inclusive'][i, :]-self.YieldsTrueASZV['inclusive'][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                   
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsminValsZV[
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueASZV['inclusive'][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueASZV['inclusive'][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueAS['inclusive'][i, :]-self.YieldsTrueASZV['inclusive'][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueAS['inclusive'][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

               
        axYieldAS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldAS.transAxes)
        axYieldAS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldAS.transAxes)

        axYieldAS.legend()
        yieldFigAS.tight_layout()
        yieldFigAS.savefig(f"{self.base_save_path}YieldAS_{'inclusive'}ZV.png")  # type:ignore
        plt.close(yieldFigAS)

    @print_function_name_with_description_on_call(description="")
    def plot_yield_for_true_species(self, species, plot_ME_systematic=True, plot_PID_systematic=True):

        panel_text1 = "ALICE Work In Progress" + "\n" + f"{'pp' if self.analysisType=='pp' else 'Pb-Pb'}" + r"$\sqrt{s_{NN}}$=5.02TeV"+ f", {'30-50%' if self.analysisType=='semicentral' else '0-10%' if self.analysisType=='central' else ''}"+ "\n"  + "anti-$k_T$ R=0.2"
        panel_text2 =  f"\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
        x_locs = [0.25, 0.4]
        y_locs = [0.75, 0.75]
        font_size = 16

        yieldFig, axYield = plt.subplots(1, 1, figsize=(10, 6))
        axYield.set_title(f"{species} Yield")
        axYield.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYield.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYield.set_yscale("log")
        if self.analysisType=='pp':
            plot_ME_systematic=False
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            axYield.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrue[species][i, :],
                self.YieldErrsTrue[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"Jet p_T {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            N_trigs = N_trigs + 1e-4
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                
                n_bins = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                per_bin_RPF_error = np.zeros((n_bins, len(self.pTassocBinCenters)))
                for l in range(n_bins):
                    x_bins[l] = (
                        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error[l] = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsForTrueSpecies[species][i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsForTrueSpecies[species][i, j].popt,
                                )
                                ** 2
                            )
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore Here we sum over the range in dphi...should I be normalizing by the range??? 2pi?


                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrue[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrue[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="green",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrue[species][i, :]-self.YieldsTrueZV[species][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrue[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrue[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrue[species][i, :]-self.YieldsTrueZV[species][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrue[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

        axYield.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYield.transAxes)
        axYield.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYield.transAxes)

        axYield.legend()
        yieldFig.tight_layout()
        yieldFig.savefig(f"{self.base_save_path}Yield_{species}.png")  # type:ignore
        plt.close(yieldFig)

        yieldFigNS, axYieldNS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldNS.set_title(f"Near side {species} Yield")
        axYieldNS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldNS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYieldNS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldNS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueNS[species][i, :],
                self.YieldErrsTrueNS[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNS[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueNS[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore

                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueNS[species][i, :]-self.YieldsTrueNSZV[species][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueNS[species][i, :]-self.YieldsTrueNSZV[species][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueNS[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

        axYieldNS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldNS.transAxes)
        axYieldNS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldNS.transAxes)
        
        axYieldNS.legend()
        yieldFigNS.tight_layout()
        yieldFigNS.savefig(f"{self.base_save_path}YieldNS_{species}.png")  # type:ignore
        plt.close(yieldFigNS)

        yieldFigAS, axYieldAS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldAS.set_title(f"Away Side {species} Yield")
        axYieldAS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldAS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)${-1}$"%(species, species))
        # set log y
        axYieldAS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldAS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueAS[species][i, :],
                self.YieldErrsTrueAS[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore

                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueAS[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueAS[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueAS[species][i, :]-self.YieldsTrueASZV[species][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigAS[0]
                        ), self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigAS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigAS[0]
                        ), self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigAS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminVals[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsTrueAS[species][i, :]-self.YieldsTrueASZV[species][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                        if self.analysisType in ["central", "semicentral"]:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForTrueSpecies[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                        else:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForTrueSpecies[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                        axYieldAS.fill_between(
                            self.pTassocBinCenters,
                            np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                            - dPhiPIDErrs,
                            np.array(self.YieldsTrueAS[species][i, :], dtype=float)
                            + dPhiPIDErrs,
                            alpha=0.3,
                            color="blue",
                            label="PID Error" if i==1 else "",
                        )
        
        axYieldAS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldAS.transAxes)
        axYieldAS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldAS.transAxes)

        axYieldAS.legend()
        yieldFigAS.tight_layout()
        yieldFigAS.savefig(f"{self.base_save_path}YieldAS_{species}.png")  # type:ignore
        plt.close(yieldFigAS)
    
    @print_function_name_with_description_on_call(description="")
    def plot_yield_for_true_species_in_z_vertex_bins(self, species, plot_ME_systematic=False, plot_PID_systematic=True):


        panel_text1 = "ALICE Work In Progress" + "\n" + f"{'pp' if self.analysisType=='pp' else 'Pb-Pb'}" + r"$\sqrt{s_{NN}}$=5.02TeV"+ f", {'30-50%' if self.analysisType=='semicentral' else '0-10%' if self.analysisType=='central' else ''}"+ "\n"  + "anti-$k_T$ R=0.2"
        panel_text2 =  f"\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
        x_locs = [0.25, 0.4]
        y_locs = [0.75, 0.75]
        font_size = 16

        yieldFig, axYield = plt.subplots(1, 1, figsize=(10, 6))
        axYield.set_title(f"{species} Yield")
        axYield.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYield.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYield.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            axYield.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueZV[species][i, :],
                self.YieldErrsTrueZV[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"Jet p_T {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            N_trigs = N_trigs + 1e-4
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                
                n_bins = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                per_bin_RPF_error = np.zeros((n_bins, len(self.pTassocBinCenters)))
                for l in range(n_bins):
                    x_bins[l] = (
                        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error[l] = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsForTrueSpeciesZV[species][i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsForTrueSpeciesZV[species][i, j].popt,
                                )
                                ** 2
                            )
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore


                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueZV[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueZV[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="green",
                    label="RPF systematics" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminValsZV[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

        axYield.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYield.transAxes)
        axYield.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYield.transAxes)

        axYield.legend()
        yieldFig.tight_layout()
        yieldFig.savefig(f"{self.base_save_path}Yield_{species}ZV.png")  # type:ignore
        plt.close(yieldFig)

        yieldFigNS, axYieldNS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldNS.set_title(f"Near Side {species} Yield")
        axYieldNS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldNS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYieldNS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldNS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueNSZV[species][i, :],
                self.YieldErrsTrueNSZV[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNSZV[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueNSZV[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore

                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminValsZV[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsTrueNSZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
        
        axYieldNS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldNS.transAxes)
        axYieldNS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldNS.transAxes)
        
        axYieldNS.legend()
        yieldFigNS.tight_layout()
        yieldFigNS.savefig(f"{self.base_save_path}YieldNS_{species}ZV.png")  # type:ignore
        plt.close(yieldFigNS)

        yieldFigAS, axYieldAS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldAS.set_title(f"Away Side{species} Yield")
        axYieldAS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldAS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYieldAS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldAS.errorbar(
                self.pTassocBinCenters,
                self.YieldsTrueASZV[species][i, :],
                self.YieldErrsTrueASZV[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore

                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueASZV[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsTrueASZV[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                    if plot_PID_systematic:
                        if self.analysisType in ["central", "semicentral"]:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                        else:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                        axYieldAS.fill_between(
                            self.pTassocBinCenters,
                            np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                            - dPhiPIDErrs,
                            np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                            + dPhiPIDErrs,
                            alpha=0.3,
                            color="blue",
                            label="PID Error" if i==1 else "",
                        )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForTrueSpeciesminValsZV[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                        if self.analysisType in ["central", "semicentral"]:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                        else:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForTrueSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForTrueSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                        axYieldAS.fill_between(
                            self.pTassocBinCenters,
                            np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                            - dPhiPIDErrs,
                            np.array(self.YieldsTrueASZV[species][i, :], dtype=float)
                            + dPhiPIDErrs,
                            alpha=0.3,
                            color="blue",
                            label="PID Error" if i==1 else "",
                        )
        
        axYieldAS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldAS.transAxes)
        axYieldAS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldAS.transAxes)
        
        axYieldAS.legend()
        yieldFigAS.tight_layout()
        yieldFigAS.savefig(f"{self.base_save_path}YieldAS_{species}ZV.png")  # type:ignore
        plt.close(yieldFigAS)
    
    @print_function_name_with_description_on_call(description="")
    def plot_yield_for_enhanced_species(self, species, plot_ME_systematic=True, plot_PID_systematic=True):

        panel_text1 = "ALICE Work In Progress" + "\n" + f"{'pp' if self.analysisType=='pp' else 'Pb-Pb'}" + r"$\sqrt{s_{NN}}$=5.02TeV"+ f", {'30-50%' if self.analysisType=='semicentral' else '0-10%' if self.analysisType=='central' else ''}"+ "\n"  + "anti-$k_T$ R=0.2"
        panel_text2 =  f"\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
        x_locs = [0.25, 0.4]
        y_locs = [0.75, 0.75]
        font_size = 16

        yieldFig, axYield = plt.subplots(1, 1, figsize=(10, 6))
        axYield.set_title(f"{species} Yield")
        axYield.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYield.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYield.set_yscale("log")
        if self.analysisType=='pp':
            plot_ME_systematic=False
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            axYield.errorbar(
                self.pTassocBinCenters,
                self.YieldsEnhanced[species][i, :],
                self.YieldErrsEnhanced[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"Jet p_T {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            N_trigs = N_trigs + 1e-4
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                
                n_bins = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                per_bin_RPF_error = np.zeros((n_bins, len(self.pTassocBinCenters)))
                for l in range(n_bins):
                    x_bins[l] = (
                        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error[l] = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsForEnhancedSpecies[species][i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsForEnhancedSpecies[species][i, j].popt,
                                )
                                ** 2
                            )
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore Here we sum over the range in dphi...should I be normalizing by the range??? 2pi?


                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhanced[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsEnhanced[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="green",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsEnhanced[species][i, :]-self.YieldsEnhancedZV[species][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminVals[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsEnhanced[species][i, :]-self.YieldsEnhancedZV[species][i,:]) # type:ignore
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhanced[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

        axYield.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYield.transAxes)
        axYield.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYield.transAxes)

        axYield.legend()
        yieldFig.tight_layout()
        yieldFig.savefig(f"{self.base_save_path}Yield_{species}Enh.png")  # type:ignore
        plt.close(yieldFig)

        yieldFigNS, axYieldNS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldNS.set_title(f"Near side {species} Yield")
        axYieldNS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldNS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYieldNS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldNS.errorbar(
                self.pTassocBinCenters,
                self.YieldsEnhancedNS[species][i, :],
                self.YieldErrsEnhancedNS[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedNS[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsEnhancedNS[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore

                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsEnhancedNS[species][i, :]-self.YieldsEnhancedNSZV[species][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore
                
                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminVals[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsEnhancedNS[species][i, :]-self.YieldsEnhancedNSZV[species][i,:]) # type:ignore
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedNS[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

        axYieldNS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldNS.transAxes)
        axYieldNS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldNS.transAxes)
        
        axYieldNS.legend()
        yieldFigNS.tight_layout()
        yieldFigNS.savefig(f"{self.base_save_path}YieldNS_{species}Enh.png")  # type:ignore
        plt.close(yieldFigNS)

        yieldFigAS, axYieldAS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldAS.set_title(f"Away Side {species} Yield")
        axYieldAS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldAS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)${-1}$"%(species, species))
        # set log y
        axYieldAS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldAS.errorbar(
                self.pTassocBinCenters,
                self.YieldsEnhancedAS[species][i, :],
                self.YieldErrsEnhancedAS[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrs[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjs[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjs[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore

                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedAS[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsEnhancedAS[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsEnhancedAS[species][i, :]-self.YieldsEnhancedASZV[species][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigAS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigAS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigAS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigAS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminVals[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    bin_contentDiff = np.zeros(len(self.pTassocBinCenters))  # type:ignore

                    
                    bin_contentDiff = np.abs(self.YieldsEnhancedAS[species][i, :]-self.YieldsEnhancedASZV[species][i,:]) # type:ignore
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                        - bin_contentDiff,
                        np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                        + bin_contentDiff,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                        if self.analysisType in ["central", "semicentral"]:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:,3] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                        else:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForEnhancedSpecies[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpecies[species][i,:] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                        axYieldAS.fill_between(
                            self.pTassocBinCenters,
                            np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                            - dPhiPIDErrs,
                            np.array(self.YieldsEnhancedAS[species][i, :], dtype=float)
                            + dPhiPIDErrs,
                            alpha=0.3,
                            color="blue",
                            label="PID Error" if i==1 else "",
                        )
        
        axYieldAS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldAS.transAxes)
        axYieldAS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldAS.transAxes)

        axYieldAS.legend()
        yieldFigAS.tight_layout()
        yieldFigAS.savefig(f"{self.base_save_path}YieldAS_{species}Enh.png")  # type:ignore
        plt.close(yieldFigAS)
    
    @print_function_name_with_description_on_call(description="")
    def plot_yield_for_enhanced_species_in_z_vertex_bins(self, species, plot_ME_systematic=False, plot_PID_systematic=True):


        panel_text1 = "ALICE Work In Progress" + "\n" + f"{'pp' if self.analysisType=='pp' else 'Pb-Pb'}" + r"$\sqrt{s_{NN}}$=5.02TeV"+ f", {'30-50%' if self.analysisType=='semicentral' else '0-10%' if self.analysisType=='central' else ''}"+ "\n"  + "anti-$k_T$ R=0.2"
        panel_text2 =  f"\n" + r"$p_T^{ch}$ c, $E_T^{clus}>$3.0 GeV" + "\n" + r"$p_T^{lead, ch} >$ 5.0 GeV/c"
        x_locs = [0.25, 0.4]
        y_locs = [0.75, 0.75]
        font_size = 16

        yieldFig, axYield = plt.subplots(1, 1, figsize=(10, 6))
        axYield.set_title(f"{species} Yield")
        axYield.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYield.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYield.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):  # type:ignore
            axYield.errorbar(
                self.pTassocBinCenters,
                self.YieldsEnhancedZV[species][i, :],
                self.YieldErrsEnhancedZV[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"Jet p_T {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            N_trigs = N_trigs + 1e-4
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                
                n_bins = (
                    self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                per_bin_RPF_error = np.zeros((n_bins, len(self.pTassocBinCenters)))
                for l in range(n_bins):
                    x_bins[l] = (
                        self.NormalizedBGSubtractedAccCorrectedSEdPhiSigcorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error[l] = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsForEnhancedSpeciesZV[species][i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsForEnhancedSpeciesZV[species][i, j].popt,
                                )
                                ** 2
                            )
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore


                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedZV[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsEnhancedZV[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="green",
                    label="RPF systematics" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminValsZV[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYield.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(dPhiPIDErrs[j].GetNbinsX())])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYield.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )

        axYield.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYield.transAxes)
        axYield.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYield.transAxes)

        axYield.legend()
        yieldFig.tight_layout()
        yieldFig.savefig(f"{self.base_save_path}Yield_{species}EnhZV.png")  # type:ignore
        plt.close(yieldFig)

        yieldFigNS, axYieldNS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldNS.set_title(f"Near Side {species} Yield")
        axYieldNS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldNS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYieldNS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldNS.errorbar(
                self.pTassocBinCenters,
                self.YieldsEnhancedNSZV[species][i, :],
                self.YieldErrsEnhancedNSZV[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigNScorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore

                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminValsZV[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldNS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                    if self.analysisType in ["central", "semicentral"]:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                    else:
                        (
                            low_bin,
                            high_bin,
                        ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[0]
                        ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                            self.dPhiSigNS[1]
                        )
                        dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                        dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                        dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                    axYieldNS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        - dPhiPIDErrs,
                        np.array(self.YieldsEnhancedNSZV[species][i, :], dtype=float)
                        + dPhiPIDErrs,
                        alpha=0.3,
                        color="blue",
                        label="PID Error" if i==1 else "",
                    )
        
        axYieldNS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldNS.transAxes)
        axYieldNS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldNS.transAxes)
        
        axYieldNS.legend()
        yieldFigNS.tight_layout()
        yieldFigNS.savefig(f"{self.base_save_path}YieldNS_{species}EnhZV.png")  # type:ignore
        plt.close(yieldFigNS)

        yieldFigAS, axYieldAS = plt.subplots(1, 1, figsize=(10, 6))
        axYieldAS.set_title(f"Away Side{species} Yield")
        axYieldAS.set_xlabel("$p_{T}^{assoc.}$ (GeV/c)")
        axYieldAS.set_ylabel("$\\frac{1}{N_{trig.}}\\frac{d(N_{meas.}^{%s}-N_{BG}^{%s})}{dp_T^{assoc.}}$ (GeV/c)$^{-1}$"%(species, species))
        # set log y
        axYieldAS.set_yscale("log")
        for i in range(len(self.pTtrigBinEdges) - 1):
            axYieldAS.errorbar(
                self.pTassocBinCenters,
                self.YieldsEnhancedASZV[species][i, :],
                self.YieldErrsEnhancedASZV[species][i, :],
                np.array(self.pTassocBinWidths) / 2,
                label=f"{self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]}GeV/c",
                fmt="o",
            )  # type:ignore
            # now add the systematics
            N_trigs = []
            for j in range(len(self.pTassocBinEdges) - 1):  # type:ignore
                self.set_pT_epAngle_bin(i, j, 3)  # type:ignore
                N_trigs.append(self.get_N_trig())  # type:ignore
            N_trigs = np.array(N_trigs)
            if self.analysisType in ["central", "semicentral"]:  # type:ignore
                n_bins = (
                    self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, 3]
                    .GetXaxis()
                    .GetNbins()
                )  # type:ignore
                x_bins = np.zeros(n_bins)
                for l in range(n_bins):
                    x_bins[l] = (
                        self.BGSubtractedAccCorrectedSEdPhiSigAScorrsZV[i, j, 3]
                        .GetXaxis()
                        .GetBinCenter(l + 1)
                    )  # type:ignore
                for l in range(n_bins):
                    per_bin_RPF_error = [
                        np.sqrt(
                            np.sum(
                                self.RPFObjsZV[i, j].simultaneous_fit_err(
                                    x_bins[l],
                                    x_bins[1] - x_bins[0],
                                    *self.RPFObjsZV[i, j].popt,
                                )
                                ** 2
                            )
                            
                        )
                        / N_trigs[j]
                        for j in range(len(self.pTassocBinCenters))
                    ]  # type:ignore indexed by j, shape (n_pt_bins, n_bins)
                RPFError = [
                    np.sum(per_bin_RPF_error[j])
                    for j in range(len(self.pTassocBinCenters))
                ]  # type:ignore

                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float) - RPFError,
                    np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float) + RPFError,
                    alpha=0.3,
                    color="orange",
                    label="RPF systematics" if i == 2 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :, 3], dtype=float)
                        / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                    if plot_PID_systematic:
                        if self.analysisType in ["central", "semicentral"]:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                        else:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                        axYieldAS.fill_between(
                            self.pTassocBinCenters,
                            np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                            - dPhiPIDErrs,
                            np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                            + dPhiPIDErrs,
                            alpha=0.3,
                            color="blue",
                            label="PID Error" if i==1 else "",
                        )
            if self.analysisType == "pp":
                BG_errorsNormINC = self.NormalizedBGSubtracteddPhiForEnhancedSpeciesminValsZV[
                    species
                ][
                    i, :
                ]  # type:ignore
                BG_errorsNormINC = np.array(BG_errorsNormINC, dtype=float) / N_trigs
                axYieldAS.fill_between(
                    self.pTassocBinCenters,
                    np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                    - BG_errorsNormINC,
                    np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                    + BG_errorsNormINC,
                    alpha=0.3,
                    color="orange",
                    label="Yield normalization" if i == 1 else "",
                )  # type:ignore
                # now ME_norm_systematics
                if plot_ME_systematic:
                    axYieldAS.fill_between(
                        self.pTassocBinCenters,
                        np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                        - np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                        + np.array(self.ME_norm_systematics[i, :], dtype=float) / N_trigs,
                        alpha=0.3,
                        color="red",
                        label="ME normalization" if i == 2 else "",
                    )  # type:ignore

                if plot_PID_systematic:
                        if self.analysisType in ["central", "semicentral"]:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0,3].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:,3] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin
                        else:
                            (
                                low_bin,
                                high_bin,
                            ) = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[0]
                            ), self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,0].GetXaxis().FindBin(
                                self.dPhiSigAS[1]
                            )
                            dPhiPIDErrs = self.dPhiSigPIDErrForEnhancedSpeciesZV[species][i,:] # a list of histograms; one for each pTassoc bin
                            dPhiPIDErrs = [np.sum([dPhiPIDErrs[j].GetBinContent(k+1) for k in range(low_bin-1, high_bin)])/dPhiPIDErrs[j].GetNbinsX() for j in range(len(dPhiPIDErrs))] # type:ignore
                            dPhiPIDErrs = np.array(dPhiPIDErrs)/N_trigs # now a list of floats for each pTassoc bin

                        axYieldAS.fill_between(
                            self.pTassocBinCenters,
                            np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                            - dPhiPIDErrs,
                            np.array(self.YieldsEnhancedASZV[species][i, :], dtype=float)
                            + dPhiPIDErrs,
                            alpha=0.3,
                            color="blue",
                            label="PID Error" if i==1 else "",
                        )
        
        axYieldAS.text(x_locs[0], y_locs[0], panel_text1, fontsize=font_size, transform=axYieldAS.transAxes)
        axYieldAS.text(x_locs[1], y_locs[1], panel_text2, fontsize=font_size, transform=axYieldAS.transAxes)
        
        axYieldAS.legend()
        yieldFigAS.tight_layout()
        yieldFigAS.savefig(f"{self.base_save_path}YieldAS_{species}EnhZV.png")  # type:ignore
        plt.close(yieldFigAS)

    def get_bin_contents_as_array(self, th1, forFitting=True, forAS=False):
        bin_contents = []
        assert not (forFitting and forAS)
        for i in range(1, th1.GetNbinsX() + 1):
            if (
                th1.GetBinCenter(i) < -np.pi / 2 or th1.GetBinCenter(i) > np.pi / 2
            ) and forFitting:
                continue
            if (
                th1.GetBinCenter(i) < np.pi / 2 or th1.GetBinCenter(i) > 3 * np.pi / 2
            ) and forAS:
                continue
            bin_contents.append(th1.GetBinContent(i))
        return np.array(bin_contents)

    def get_2D_bin_contents_as_array(self, th2, forFitting=True, forAS=False):
        bin_contents = []
        assert not (forFitting and forAS)
        for i in range(1, th2.GetNbinsX() + 1):
            if (
                th2.GetXaxis().GetBinCenter(i) < -np.pi / 2
                or th2.GetXaxis().GetBinCenter(i) > np.pi / 2
            ) and forFitting:
                continue
            if (
                th2.GetXaxis().GetBinCenter(i) < np.pi / 2
                or th2.GetXaxis().GetBinCenter(i) > 3 * np.pi / 2
            ) and forAS:
                continue
            y_contents = []
            for j in range(1, th2.GetNbinsY() + 1):
                y_contents.append(th2.GetBinContent(i, j))

            bin_contents.append(y_contents)
        return np.array(bin_contents)

    def get_bin_centers_as_array(self, th1, forFitting=True, forAS=False):
        bin_centers = []
        for i in range(1, th1.GetNbinsX() + 1):
            if (
                th1.GetBinCenter(i) < -np.pi / 2 or th1.GetBinCenter(i) > np.pi / 2
            ) and forFitting:
                continue
            if (
                th1.GetBinCenter(i) < np.pi / 2 or th1.GetBinCenter(i) > 3 * np.pi / 2
            ) and forAS:
                continue
            bin_centers.append(th1.GetBinCenter(i))
        return np.array(bin_centers)

    def get_2D_bin_centers_as_array(self, th2, forFitting=True, forAS=False):
        x_bin_centers = []
        y_bin_centers = []
        for i in range(1, th2.GetNbinsX() + 1):
            if (
                th2.GetXaxis().GetBinCenter(i) < -np.pi / 2
                or th2.GetXaxis().GetBinCenter(i) > np.pi / 2
            ) and forFitting:
                continue
            if (
                th2.GetXaxis().GetBinCenter(i) < np.pi / 2
                or th2.GetXaxis().GetBinCenter(i) > 3 * np.pi / 2
            ) and forAS:
                continue
            x_bin_centers.append(th2.GetXaxis().GetBinCenter(i))
        for i in range(1, th2.GetNbinsY() + 1):
            y_bin_centers.append(th2.GetYaxis().GetBinCenter(i))
        return np.array(x_bin_centers), np.array(y_bin_centers)

    def get_bin_errors_as_array(self, th1, forFitting=True, forAS=False):
        bin_errors = []
        for i in range(1, th1.GetNbinsX() + 1):
            if (
                th1.GetBinCenter(i) < -np.pi / 2 or th1.GetBinCenter(i) > np.pi / 2
            ) and forFitting:
                continue
            if (
                th1.GetBinCenter(i) < np.pi / 2 or th1.GetBinCenter(i) > 3 * np.pi / 2
            ) and forAS:
                continue
            bin_errors.append(th1.GetBinError(i))
        return np.array(bin_errors)

    def get_2D_bin_errors_as_array(self, th2, forFitting=True, forAS=False):
        bin_errors = []
        for i in range(1, th2.GetNbinsX() + 1):
            if (
                th2.GetXaxis().GetBinCenter(i) < -np.pi / 2
                or th2.GetXaxis().GetBinCenter(i) > np.pi / 2
            ) and forFitting:
                continue
            if (
                th2.GetXaxis().GetBinCenter(i) < np.pi / 2
                or th2.GetXaxis().GetBinCenter(i) > 3 * np.pi / 2
            ) and forAS:
                continue
            y_err = []
            for j in range(1, th2.GetNbinsY() + 1):
                y_err.append(th2.GetBinError(i, j))
            bin_errors.append(y_err)
        return np.array(bin_errors)
