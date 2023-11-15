from JetHadronAnalysis.Types import AnalysisType, Region, AssociatedHadronMomentumBin, ParticleType
import sqlite3
import matplotlib.pyplot as plt

from JetHadronAnalysis.PIDDB import getParticleFractionByMomentum, getParameterByMomentum

def plotParticleFractionsByMomentum(analysisType, region, dbCursor):
    plt.figure()
    pionFractionsByMomentum = getParticleFractionByMomentum(analysisType = analysisType, region = region, particleType = ParticleType.PION, dbCursor=dbCursor)
    protonFractionsByMomentum = getParticleFractionByMomentum(analysisType = analysisType, region = region, particleType = ParticleType.PROTON, dbCursor=dbCursor)
    kaonFractionsByMomentum = getParticleFractionByMomentum(analysisType = analysisType, region = region, particleType = ParticleType.KAON, dbCursor=dbCursor)

    plt.errorbar(x=[x[0] for x in pionFractionsByMomentum], y=[x[1] for x in pionFractionsByMomentum], yerr=[x[2] for x in pionFractionsByMomentum], fmt='o', label="Pion")
    plt.errorbar(x=[x[0] for x in protonFractionsByMomentum], y=[x[1] for x in protonFractionsByMomentum], yerr=[x[2] for x in protonFractionsByMomentum], fmt='o', label="Proton")
    plt.errorbar(x=[x[0] for x in kaonFractionsByMomentum], y=[x[1] for x in kaonFractionsByMomentum], yerr=[x[2] for x in kaonFractionsByMomentum], fmt='o', label="Kaon")

    plt.ylim(0, 1)
    plt.title(f"Particle Fractions by Momentum for {analysisType.name} {region.name}")
    plt.legend()
    plt.savefig(f"particleFractionsByMomentum{analysisType.name}_{region.name}.png")
    plt.close()

def plotFitParametersByMomentum(analysisType, parameter, expected_range: list, dbCursor):
    plt.figure()
    parameterByMomentum_ns = getParameterByMomentum(analysisType = analysisType, region = Region.NEAR_SIDE_SIGNAL, parameter = parameter, dbCursor=dbCursor)
    parameterByMomentum_as = getParameterByMomentum(analysisType = analysisType, region = Region.AWAY_SIDE_SIGNAL, parameter = parameter, dbCursor=dbCursor)
    parameterByMomentum_b = getParameterByMomentum(analysisType = analysisType, region = Region.BACKGROUND, parameter = parameter, dbCursor=dbCursor)
    plt.errorbar(x=[x[0] for x in parameterByMomentum_ns], y=[x[1] for x in parameterByMomentum_ns], yerr=[x[2] for x in parameterByMomentum_ns], fmt='o', label=Region.NEAR_SIDE_SIGNAL.name)
    plt.errorbar(x=[x[0]+0.1 for x in parameterByMomentum_as], y=[x[1] for x in parameterByMomentum_as], yerr=[x[2] for x in parameterByMomentum_as], fmt='o', label=Region.AWAY_SIDE_SIGNAL.name)
    plt.errorbar(x=[x[0]+0.2 for x in parameterByMomentum_b], y=[x[1] for x in parameterByMomentum_b], yerr=[x[2] for x in parameterByMomentum_b], fmt='o', label=Region.BACKGROUND.name)
    plt.ylim(*expected_range)
    plt.title(f"{parameter} by Momentum for {analysisType.name}")
    plt.legend()
    plt.savefig(f"{parameter}ByMomentum{analysisType.name}.png")
    plt.close()

    

def main():
    conn = sqlite3.connect("PID.db")
    c = conn.cursor()
    for ana_type in AnalysisType:
        plotParticleFractionsByMomentum(analysisType = ana_type, region = Region.NEAR_SIDE_SIGNAL, dbCursor=c)
        plotParticleFractionsByMomentum(analysisType = ana_type, region = Region.AWAY_SIDE_SIGNAL, dbCursor=c)
        plotParticleFractionsByMomentum(analysisType = ana_type, region = Region.BACKGROUND, dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "mu_pion", expected_range=[-0.2, 0.2], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "mu_proton", expected_range=[-4, 2], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "mu_kaon", expected_range=[-4, 2], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "sigma_pion", expected_range=[0, 2], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "sigma_proton", expected_range=[0, 2], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "sigma_kaon", expected_range=[0, 2], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "alpha_proton", expected_range=[-4, 5], dbCursor=c)
        plotFitParametersByMomentum(analysisType = ana_type,  parameter = "alpha_kaon", expected_range=[-4, 5], dbCursor=c)
    conn.close()

if __name__ == "__main__":
    main()