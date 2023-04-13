from itertools import product
import numpy as np
from RPF import RPF


class FitMixin:

    def __init__(self):
        # call super init 
        super().__init__()
        


    def fit_RPFs(self):
        for i in range(len(self.pTtrigBinEdges)-1): #type:ignore
            for j in range(len(self.pTassocBinEdges)-1): #type:ignore
                self.fit_RPF(i,j, p0=self.central_p0s[(i,j)])

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
        print("*"*80)
        print(f"Error: {sum(np.sqrt(np.diag(pcov)))}")
        error = sum(np.sqrt(np.diag(pcov)))
        print("*"*80)
        best_error = error
        best_popt = popt
        best_pcov = pcov
        best_p0 = p0
        self.RPFObjs[i,j] = rpf #type:ignore
        
        print(f"Best RPF fit for pTtrig {self.pTtrigBinEdges[i]}-{self.pTtrigBinEdges[i+1]} GeV, pTassoc {self.pTassocBinEdges[j]}-{self.pTassocBinEdges[j+1]} GeV had Error Sum of {best_error} and p0 of {best_p0} with final parameter values of {list(zip(['B', 'v2', 'v3', 'v4', 'va2', 'va4'], best_popt))}") #type:ignore
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


    
