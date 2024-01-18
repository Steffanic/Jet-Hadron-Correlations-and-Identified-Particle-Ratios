from JetHadronAnalysis.Analysis import Analysis, Region, AssociatedHadronMomentumBin
from JetHadronAnalysis.Types import AnalysisType as at
from JetHadronAnalysis.Types import NormalizationMethod as nm
from JetHadronAnalysis.Types import ParticleType as pt

import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":
    ana_pp = Analysis(at.PP, ["/mnt/d/pp/17p.root"])
    print(f"Starting pp")
    for assoc_bin in AssociatedHadronMomentumBin:
        ana_pp.setAssociatedHadronMomentumBin(assoc_bin)
        ana_pp.setRegion(Region.INCLUSIVE)
        print(f"pp, {assoc_bin}, Inclusive, {ana_pp.getPIDFractions()}")
        ana_pp.setRegion(Region.NEAR_SIDE_SIGNAL)
        print(f"pp, {assoc_bin}, Near-side signal, {ana_pp.getPIDFractions()}")
        ana_pp.setRegion(Region.AWAY_SIDE_SIGNAL)
        print(f"pp, {assoc_bin}, Away-side signal, {ana_pp.getPIDFractions()}")
        ana_pp.setRegion(Region.BACKGROUND)
        print(f"pp, {assoc_bin}, Background, {ana_pp.getPIDFractions()}")
    

    ana_semicentral = Analysis(at.SEMICENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root",])
    print(f"Starting semicentral")
    for assoc_bin in AssociatedHadronMomentumBin:
        ana_semicentral.setAssociatedHadronMomentumBin(assoc_bin)
        ana_semicentral.setRegion(Region.INCLUSIVE)
        print(f"semicentral, {assoc_bin}, Inclusive, {ana_semicentral.getPIDFractions()}")
        ana_semicentral.setRegion(Region.NEAR_SIDE_SIGNAL)
        print(f"semicentral, {assoc_bin}, Near-side signal, {ana_semicentral.getPIDFractions()}")
        ana_semicentral.setRegion(Region.AWAY_SIDE_SIGNAL)
        print(f"semicentral, {assoc_bin}, Away-side signal, {ana_semicentral.getPIDFractions()}")
        ana_semicentral.setRegion(Region.BACKGROUND)
        print(f"semicentral, {assoc_bin}, Background, {ana_semicentral.getPIDFractions()}")
    
    ana_central = Analysis(at.CENTRAL, ["/mnt/d/18q/new_root/296510.root","/mnt/d/18q/new_root/296550.root","/mnt/d/18q/new_root/296551.root","/mnt/d/18q/new_root/295673.root","/mnt/d/18q/new_root/295754.root","/mnt/d/18q/new_root/296065.root","/mnt/d/18q/new_root/296068.root","/mnt/d/18q/new_root/296133.root","/mnt/d/18q/new_root/296191.root","/mnt/d/18q/new_root/296377.root","/mnt/d/18q/new_root/296379.root","/mnt/d/18q/new_root/296423.root","/mnt/d/18q/new_root/296433.root","/mnt/d/18q/new_root/296472.root",])
    print(f"Starting central")
    for assoc_bin in AssociatedHadronMomentumBin:
        ana_central.setAssociatedHadronMomentumBin(assoc_bin)
        ana_central.setRegion(Region.INCLUSIVE)
        print(f"central, {assoc_bin}, Inclusive, {ana_central.getPIDFractions()}")
        ana_central.setRegion(Region.NEAR_SIDE_SIGNAL)
        print(f"central, {assoc_bin}, Near-side signal, {ana_central.getPIDFractions()}")
        ana_central.setRegion(Region.AWAY_SIDE_SIGNAL)
        print(f"central, {assoc_bin}, Away-side signal, {ana_central.getPIDFractions()}")
        ana_central.setRegion(Region.BACKGROUND)
        print(f"central, {assoc_bin}, Background, {ana_central.getPIDFractions()}")
    