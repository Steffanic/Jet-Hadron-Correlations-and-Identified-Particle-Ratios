from warnings import warn
import numpy as np
from JetHadron import JetHadron
from ROOT import TFile, THnSparseF 
from array import array
from pytest import fixture, skip

def make_mock_JH_ThNsparse():
    nbins = array("i",[2, 24, 100, 28, 72, 2, 3, 100])
    dim_range_low = array("d",[0, 0, 0, -1.4, -np.pi/2, -0.5, 0, -20])
    dim_range_high = array("d",[20, 200, 10, 1.4, 3*np.pi/2, 1.5, np.pi/2, 20])
    JH = THnSparseF("JH", "JH", 8, nbins, dim_range_low, dim_range_high)
    # now fill in 1000000 random entries 
    for i in range(1000000):
        arr = np.random.rand(8)
        JH.Fill(array("d",arr))
    return JH

@fixture
def centralAnalysisObject():
    return JetHadron(["/home/steffanic/Projects/Thesis/TrainOutputq/AnalysisResults_alihaddcomp04.root"], "central", fill_on_init=False, pickle_on_init=False, plot_on_init=False)

@fixture
def semiCentralAnalysisObject():
    return JetHadron(["/home/steffanic/Projects/Thesis/TrainOutputq/AnalysisResults_alihaddcomp04.root"], "semicentral", fill_on_init=False, pickle_on_init=False, plot_on_init=False)


def test_mixed_event_normalization(centralAnalysisObject, semiCentralAnalysisObject):
    for i in range(len(centralAnalysisObject.pTtrigBinEdges)-1):
        for j in range(len(centralAnalysisObject.pTassocBinEdges)-1):
            for k in range(4):
                centralAnalysisObject.set_pT_epAngle_bin(i,j,k)
                norm_ME, error = centralAnalysisObject.get_normalized_ME_correlation_function()
                assert norm_ME.GetMaximum()-1>0
                if norm_ME.GetMaximum()-1>0.1:
                    warn(f"Trig bin {i}, assoc bin {j}, event plane bin {k} has a max of {norm_ME.GetMaximum()} which is larger than 1.1.")
    
