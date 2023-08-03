from itertools import product
import numpy as np
from RPF import RPF
from PionTPCNSigmaFitter import PionTPCNSigmaFitter
import logging
debug_logger = logging.getLogger('debug')
error_logger = logging.getLogger('error')
info_logger = logging.getLogger('info')


class FitMixin:

    def __init__(self):
        # call super init 
        super().__init__()
        


    def fit_RPFs(self):
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            for j in range(len(self.pTassocBinEdges)-1): #type:ignore
                self.fit_RPF(i,j, p0=self.central_p0s[(i,j)])

    def fit_RPFs_for_species(self):
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            for j in range(len(self.pTassocBinEdges)-1): #type:ignore
                for species in ['pion', 'proton', "kaon"]:
                    self.fit_RPF_for_true_species(i,j, species, p0=self.central_p0s[(i,j)])

    def fit_RPF(self, i, j, p0=None):
        # get the RPF object
        best_error = 100000
        best_popt = []
        best_pcov = None
        best_p0 = None

        rpf = RPF(analysisType=self.analysisType) if not p0 else RPF(p0=p0, analysisType=self.analysisType)
        inPlane  = self.dPhiBGcorrs[i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrs[i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrs[i,j,2] #type:ignore
        combo_x = self.get_bin_centers_as_array(inPlane) #type:ignore
        combo_y = np.hstack((self.get_bin_contents_as_array(inPlane), self.get_bin_contents_as_array(midPlane), self.get_bin_contents_as_array(outPlane))) #type:ignore
        combo_yerr = np.hstack((self.get_bin_errors_as_array(inPlane), self.get_bin_errors_as_array(midPlane), self.get_bin_errors_as_array(outPlane))) #type:ignore
        popt, pcov, chi2OverNDF = rpf.fit(combo_x, combo_y, combo_yerr)
        debug_logger.debug("*"*80)
        debug_logger.debug(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        debug_logger.debug("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjs[i,j] = rpf #type:ignore
        
        
        debug_logger.debug(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
        # now let's write a latex table to a txt file for the RPF parameters and their names
        with open(f"RPF_parameters_{self.analysisType}.txt", "a") as f: #type:ignore
            f.write(f"\\begin{{table}}[h!]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{RPF parameters for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV}}\n")
            f.write(f"\\begin{{tabular}}{{|c|c|}}\n")
            f.write(f"\\hline\n")
            f.write(f"Parameter & Value \\\\ \n")
            f.write(f"\\hline\n")
            for name, value in list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4', 'R2', 'R4'], best_popt)):
                f.write(f"{name} & {value} \\\\ \n")
            f.write(f"\\hline\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n")
            f.write(f"\n")
            
        return best_popt, best_pcov, best_error, best_p0
    
    def fit_RPF_in_z_vertex_bins(self, i, j, p0=None):
        # get the RPF object
        best_error = 100000
        best_popt = []
        best_pcov = None
        best_p0 = None

        rpf = RPF(analysisType=self.analysisType) if not p0 else RPF(p0=p0, analysisType=self.analysisType)
        inPlane  = self.dPhiBGcorrsZV[i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrsZV[i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrsZV[i,j,2] #type:ignore
        combo_x = self.get_bin_centers_as_array(inPlane) #type:ignore
        combo_y = np.hstack((self.get_bin_contents_as_array(inPlane), self.get_bin_contents_as_array(midPlane), self.get_bin_contents_as_array(outPlane))) #type:ignore
        combo_yerr = np.hstack((self.get_bin_errors_as_array(inPlane), self.get_bin_errors_as_array(midPlane), self.get_bin_errors_as_array(outPlane))) #type:ignore
        popt, pcov, chi2OverNDF = rpf.fit(combo_x, combo_y, combo_yerr)
        debug_logger.debug("*"*80)
        debug_logger.debug(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        debug_logger.debug("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjsZV[i,j] = rpf #type:ignore
        
        
        debug_logger.debug(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
        # now let's write a latex table to a txt file for the RPF parameters and their names
        with open(f"RPF_parameters_{self.analysisType}.txt", "a") as f: #type:ignore
            f.write(f"\\begin{{table}}[h!]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{RPF parameters for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV}}\n")
            f.write(f"\\begin{{tabular}}{{|c|c|}}\n")
            f.write(f"\\hline\n")
            f.write(f"Parameter & Value \\\\ \n")
            f.write(f"\\hline\n")
            for name, value in list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4', 'R2', 'R4'], best_popt)):
                f.write(f"{name} & {value} \\\\ \n")
            f.write(f"\\hline\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n")
            f.write(f"\n")
            
        return best_popt, best_pcov, best_error, best_p0
    
    def fit_RPF_for_true_species(self, i, j, species, p0=None):
        # get the RPF object
        best_error = 100000
        best_popt = []
        best_pcov = None
        best_p0 = None

        rpf = RPF(analysisType=self.analysisType) if not p0 else RPF(p0=p0, analysisType=self.analysisType)
        inPlane  = self.dPhiBGcorrsForTrueSpecies[species][i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrsForTrueSpecies[species][i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrsForTrueSpecies[species][i,j,2] #type:ignore
        combo_x = self.get_bin_centers_as_array(inPlane) #type:ignore
        combo_y = np.hstack((self.get_bin_contents_as_array(inPlane), self.get_bin_contents_as_array(midPlane), self.get_bin_contents_as_array(outPlane))) #type:ignore
        combo_yerr = np.hstack((self.get_bin_errors_as_array(inPlane), self.get_bin_errors_as_array(midPlane), self.get_bin_errors_as_array(outPlane))) #type:ignore
        popt, pcov, chi2OverNDF = rpf.fit(combo_x, combo_y, combo_yerr)
        debug_logger.debug("*"*80)
        debug_logger.debug(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        debug_logger.debug("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjsForTrueSpecies[species][i,j] = rpf #type:ignore
        
        
        debug_logger.debug(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
        # now let's write a latex table to a txt file for the RPF parameters and their names
        with open(f"RPF_parameters_{species}_{self.analysisType}.txt", "a") as f: #type:ignore
            f.write(f"\\begin{{table}}[h!]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{RPF parameters for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV}}\n")
            f.write(f"\\begin{{tabular}}{{|c|c|}}\n")
            f.write(f"\\hline\n")
            f.write(f"Parameter & Value \\\\ \n")
            f.write(f"\\hline\n")
            for name, value in list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4', 'R2', 'R4'], best_popt)):
                f.write(f"{name} & {value} \\\\ \n")
            f.write(f"\\hline\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n")
            f.write(f"\n")
            
        return best_popt, best_pcov, best_error, best_p0
    
    def fit_RPF_for_true_species_in_z_vertex_bins(self, i, j, species, p0=None):
        # get the RPF object
        best_error = 100000
        best_popt = []
        best_pcov = None
        best_p0 = None

        rpf = RPF(analysisType=self.analysisType) if not p0 else RPF(p0=p0, analysisType=self.analysisType)
        inPlane  = self.dPhiBGcorrsForTrueSpeciesZV[species][i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrsForTrueSpeciesZV[species][i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrsForTrueSpeciesZV[species][i,j,2] #type:ignore
        combo_x = self.get_bin_centers_as_array(inPlane) #type:ignore
        combo_y = np.hstack((self.get_bin_contents_as_array(inPlane), self.get_bin_contents_as_array(midPlane), self.get_bin_contents_as_array(outPlane))) #type:ignore
        combo_yerr = np.hstack((self.get_bin_errors_as_array(inPlane), self.get_bin_errors_as_array(midPlane), self.get_bin_errors_as_array(outPlane))) #type:ignore
        popt, pcov, chi2OverNDF = rpf.fit(combo_x, combo_y, combo_yerr)
        debug_logger.debug("*"*80)
        debug_logger.debug(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        debug_logger.debug("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjsForTrueSpeciesZV[species][i,j] = rpf #type:ignore
        
        
        debug_logger.debug(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
        # now let's write a latex table to a txt file for the RPF parameters and their names
        with open(f"RPF_parameters_{species}_{self.analysisType}.txt", "a") as f: #type:ignore
            f.write(f"\\begin{{table}}[h!]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{RPF parameters for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV}}\n")
            f.write(f"\\begin{{tabular}}{{|c|c|}}\n")
            f.write(f"\\hline\n")
            f.write(f"Parameter & Value \\\\ \n")
            f.write(f"\\hline\n")
            for name, value in list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4', 'R2', 'R4'], best_popt)):
                f.write(f"{name} & {value} \\\\ \n")
            f.write(f"\\hline\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n")
            f.write(f"\n")
            
        return best_popt, best_pcov, best_error, best_p0
    
    def fit_RPF_for_enhanced_species(self, i, j, species, p0=None):
        # get the RPF object
        best_error = 100000
        best_popt = []
        best_pcov = None
        best_p0 = None

        rpf = RPF(analysisType=self.analysisType) if not p0 else RPF(p0=p0, analysisType=self.analysisType)
        inPlane  = self.dPhiBGcorrsForEnhancedSpecies[species][i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrsForEnhancedSpecies[species][i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrsForEnhancedSpecies[species][i,j,2] #type:ignore
        combo_x = self.get_bin_centers_as_array(inPlane) #type:ignore
        combo_y = np.hstack((self.get_bin_contents_as_array(inPlane), self.get_bin_contents_as_array(midPlane), self.get_bin_contents_as_array(outPlane))) #type:ignore
        combo_yerr = np.hstack((self.get_bin_errors_as_array(inPlane), self.get_bin_errors_as_array(midPlane), self.get_bin_errors_as_array(outPlane))) #type:ignore
        popt, pcov, chi2OverNDF = rpf.fit(combo_x, combo_y, combo_yerr)
        debug_logger.debug("*"*80)
        debug_logger.debug(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        debug_logger.debug("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjsForEnhancedSpecies[species][i,j] = rpf #type:ignore
        
        
        debug_logger.debug(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
        # now let's write a latex table to a txt file for the RPF parameters and their names
        with open(f"RPF_parameters_{species}_{self.analysisType}.txt", "a") as f: #type:ignore
            f.write(f"\\begin{{table}}[h!]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{RPF parameters for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV}}\n")
            f.write(f"\\begin{{tabular}}{{|c|c|}}\n")
            f.write(f"\\hline\n")
            f.write(f"Parameter & Value \\\\ \n")
            f.write(f"\\hline\n")
            for name, value in list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4', 'R2', 'R4'], best_popt)):
                f.write(f"{name} & {value} \\\\ \n")
            f.write(f"\\hline\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n")
            f.write(f"\n")
            
        return best_popt, best_pcov, best_error, best_p0
    
    def fit_RPF_for_enhanced_species_in_z_vertex_bins(self, i, j, species, p0=None):
        # get the RPF object
        best_error = 100000
        best_popt = []
        best_pcov = None
        best_p0 = None

        rpf = RPF(analysisType=self.analysisType) if not p0 else RPF(p0=p0, analysisType=self.analysisType)
        inPlane  = self.dPhiBGcorrsForEnhancedSpeciesZV[species][i,j,0] #type:ignore
        midPlane = self.dPhiBGcorrsForEnhancedSpeciesZV[species][i,j,1] #type:ignore
        outPlane = self.dPhiBGcorrsForEnhancedSpeciesZV[species][i,j,2] #type:ignore
        combo_x = self.get_bin_centers_as_array(inPlane) #type:ignore
        combo_y = np.hstack((self.get_bin_contents_as_array(inPlane), self.get_bin_contents_as_array(midPlane), self.get_bin_contents_as_array(outPlane))) #type:ignore
        combo_yerr = np.hstack((self.get_bin_errors_as_array(inPlane), self.get_bin_errors_as_array(midPlane), self.get_bin_errors_as_array(outPlane))) #type:ignore
        popt, pcov, chi2OverNDF = rpf.fit(combo_x, combo_y, combo_yerr)
        debug_logger.debug("*"*80)
        debug_logger.debug(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        debug_logger.debug("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjsForEnhancedSpeciesZV[species][i,j] = rpf #type:ignore
        
        
        debug_logger.debug(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
        # now let's write a latex table to a txt file for the RPF parameters and their names
        with open(f"RPF_parameters_{species}_{self.analysisType}.txt", "a") as f: #type:ignore
            f.write(f"\\begin{{table}}[h!]\n")
            f.write(f"\\centering\n")
            f.write(f"\\caption{{RPF parameters for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV}}\n")
            f.write(f"\\begin{{tabular}}{{|c|c|}}\n")
            f.write(f"\\hline\n")
            f.write(f"Parameter & Value \\\\ \n")
            f.write(f"\\hline\n")
            for name, value in list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4', 'R2', 'R4'], best_popt)):
                f.write(f"{name} & {value} \\\\ \n")
            f.write(f"\\hline\n")
            f.write(f"\\end{{tabular}}\n")
            f.write(f"\\end{{table}}\n")
            f.write(f"\n")
            
        return best_popt, best_pcov, best_error, best_p0

    def fit_PionTPCNSigma(self, i, j, k, region, p0=None, w_inclusive=True, generalized=True):
        debug_logger.info(f"Fitting Pion TPC N Sigma for {i=}, {j=}, {k=}")
        if p0 is None:
            if w_inclusive:
                inclusive_p0 = [80,15,5]
                inclusive_bounds = [[0,0,0],[100000,100000,100000]]
            else:
                inclusive_p0 = []
                inclusive_bounds = [[],[]]
            if generalized:
                generalized_p0 = [0.1, 0.1]
                generalized_bounds = [[-6, -6], [6, 6]]
            else:
                generalized_p0 = []
                generalized_bounds = [[],[]]
            if self.analysisType=='pp':
                if j==0:
                    p0 = [2.5, 0, -.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100]+inclusive_p0 + generalized_p0
                    bounds = [[-6, -0.1, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [6, 0.1, 6, 100.0, 100.0, 100.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1] + generalized_bounds[1]]
                elif j>0 and j<4:
                    p0 = [-1.0, 0, -2.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100] + inclusive_p0+ generalized_p0
                    bounds = [[-6, -0.1, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [0, 0.1, 0, 100.0, 100.0, 100.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0]+ generalized_bounds[0], bounds[1]+inclusive_bounds[1] + generalized_bounds[1]]
                else:
                    p0 = [-2.5, 0, -2.5, 0.5, 0.5, 0.5, 100,100, 0.11, 10,100, 10, 0.11,100, 100] + inclusive_p0+ generalized_p0
                    bounds = [[-6, -0.1, -6, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [0, 0.1, 0, 100.0, 100.0, 100.0, 100000,100000,10,100000,100000,100000,10,100000,100000]]
                   
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]

            elif self.analysisType in ['central', 'semicentral']:
                if j==0:
                    p0 = [-0.5, 0.0, 1.0,  0.5, 0.5, 0.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                    bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0] + generalized_bounds[0], bounds[1]+inclusive_bounds[1]+ generalized_bounds[1]]
                else:
                    p0 = [-1.0, 0.0, 1.0,  0.5, 0.5, 0.5, 100,100, 1.1, 10,100, 10, 1.1,100, 100] + inclusive_p0+ generalized_p0
                    bounds = [[-1.5, -0.5, -1.5, 4e-1,4e-1,4e-1, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0], [1.5, 0.5, 1.5, 10.0, 10.0, 10.0, 100000,100000,100000,100000,100000,100000,100000,100000,100000]]
                    
                    bounds = [bounds[0]+inclusive_bounds[0]+generalized_bounds[0], bounds[1]+inclusive_bounds[1]+generalized_bounds[1]]
        fitter = PionTPCNSigmaFitter(p0=p0, p0_bounds=bounds, w_inclusive=w_inclusive, generalized=generalized)


        if self.analysisType in ['central', 'semicentral']:
            pionEnhanced = self.pionTPCnSigma_pionTOFcut[region][i, j, k]
            protonEnchanced = self.pionTPCnSigma_protonTOFcut[region][i, j, k]
            kaonEnhanced = self.pionTPCnSigma_kaonTOFcut[region][i, j, k]
            inclusive = self.pionTPCnSigmaInc[region][i, j,k]
        else:
            pionEnhanced = self.pionTPCnSigma_pionTOFcut[region][i, j]
            protonEnchanced = self.pionTPCnSigma_protonTOFcut[region][i, j]
            kaonEnhanced = self.pionTPCnSigma_kaonTOFcut[region][i, j]
            inclusive = self.pionTPCnSigmaInc[region][i, j]

        nbins = pionEnhanced.GetNbinsX()
        x = np.array([pionEnhanced.GetBinCenter(i) for i in range(1, nbins+1)])
        y_pi = np.array([pionEnhanced.GetBinContent(i) for i in range(1, nbins+1)])
        y_p = np.array([protonEnchanced.GetBinContent(i) for i in range(1, nbins+1)])
        y_k = np.array([kaonEnhanced.GetBinContent(i) for i in range(1, nbins+1)])
        y = [y_pi, y_p, y_k]
        yerr_pi = np.array([pionEnhanced.GetBinError(i) for i in range(1, nbins+1)])
        yerr_p = np.array([protonEnchanced.GetBinError(i) for i in range(1, nbins+1)])
        yerr_k = np.array([kaonEnhanced.GetBinError(i) for i in range(1, nbins+1)])
        yerr = [yerr_pi, yerr_p, yerr_k]

        if w_inclusive:
            y_inc = np.array([inclusive.GetBinContent(i) for i in range(1, nbins+1)])
            yerr_inc = np.array([inclusive.GetBinError(i) for i in range(1, nbins+1)])
            y = [y_pi, y_p, y_k, y_inc]
            yerr = [yerr_pi, yerr_p, yerr_k, yerr_inc]
        best_popt, best_pcov, chi2perNDF = fitter.fit(x, y, yerr=yerr)

        if self.analysisType in ['central', 'semicentral']:
            self.PionTPCNSigmaFitObjs[region][i,j,k] = fitter
        else:
            self.PionTPCNSigmaFitObjs[region][i,j] = fitter

        return best_popt, best_pcov, chi2perNDF


    
        
    def get_p0s(self, i, j):
        B = [1000*np.exp(-2*self.pTtrigBinEdges[i])+1] #type:ignore
        v1 = [0.005]
        v2 = 0.05 if self.analysisType == "central" else 0.1 #type:ignore
        v2 = [v2]
        v3 = 0.005**2 if self.analysisType == "central" else 0.0075**2 #type:ignore
        v3 = [0.005]
        v4 = 0.01 if self.analysisType == "central" else 0.025 #type:ignore
        v4 = [v4]
        va2 = [0.1]
        va4 = 0.02 if self.analysisType == "central" else 0.07 #type:ignore
        va4 = [va4]
       
        return product(B, v1, v2, v3, v4, va2, va4)


    
