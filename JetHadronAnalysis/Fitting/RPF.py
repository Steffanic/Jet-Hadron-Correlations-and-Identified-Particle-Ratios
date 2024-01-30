from collections import OrderedDict
import warnings
import numpy as np
from JetHadronAnalysis.Types import AnalysisType, AssociatedHadronMomentumBin, TriggerJetMomentumBin

initial_parameter_defaults = {}
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_1_15)] = [
    1000042.8,
    0.0473,
    -0.000306,
    0.02,
    0.1013,
    0.03,
]  # pTtrig 20-40, pTassoc 1.0-1.5
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_15_2)] = [
    40000.19,
    0.0402,
    -0.0058,
    0.02,
    0.1506,
    0.03,
]  # pTtrig 20-40, pTassoc 1.5-2.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_2_3)] = [
    4006.86,
    0.0414,
    0.0015,
    0.02,
    0.234,
    0.03,
]  # pTtrig 20-40, pTassoc 2.0-3.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_3_4)] = [
    56.84,
    0.0636,
    -0.00766,
    0.02,
    0.237,
    0.03,
]  # pTtrig 20-40, pTassoc 3.0-4.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_4_5)] = [
    8.992,
    0.1721,
    -0.0987,
    0.02,
    0.233,
    0.03,
]  # pTtrig 20-40, pTassoc 4.0-5.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_5_6)] = [
    2.318,
    -0.0508,
    -0.143,
    0.02,
    0.1876,
    0.03,
]  # pTtrig 20-40, pTassoc  5.0-6.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_6_10)] = [
    2.076,
    -0.0886,
    0.12929,
    0.02,
    0.0692,
    0.03,
]  # pTtrig 20-40, pTassoc 6.0-10.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_1_15)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 1.0-1.5
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_15_2)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 1.5-2.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_2_3)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 2.0-3.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_3_4)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 3.0-4.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_4_5)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc    4.0-5.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_5_6)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 5.0-6.0
initial_parameter_defaults[(AnalysisType.SEMICENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_6_10)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 6.0-10.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_1_15)] = [
    1000042.8,
    0.0473,
    -0.000306,
    0.02,
    0.1013,
    0.03,
]  # pTtrig 20-40, pTassoc 1.0-1.5
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_15_2)] = [
    40000.19,
    0.0402,
    -0.0058,
    0.02,
    0.1506,
    0.03,
]  # pTtrig 20-40, pTassoc 1.5-2.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_2_3)] = [
    4006.86,
    0.0414,
    0.0015,
    0.02,
    0.234,
    0.03,
]  # pTtrig 20-40, pTassoc 2.0-3.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_3_4)] = [
    56.84,
    0.0636,
    -0.00766,
    0.02,
    0.237,
    0.03,
]  # pTtrig 20-40, pTassoc 3.0-4.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_4_5)] = [
    8.992,
    0.1721,
    -0.0987,
    0.02,
    0.233,
    0.03,
]  # pTtrig 20-40, pTassoc 4.0-5.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_5_6)] = [
    2.318,
    -0.0508,
    -0.143,
    0.02,
    0.1876,
    0.03,
]  # pTtrig 20-40, pTassoc  5.0-6.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_20_40, AssociatedHadronMomentumBin.PT_6_10)] = [
    2.076,
    -0.0886,
    0.12929,
    0.02,
    0.0692,
    0.03,
]  # pTtrig 20-40, pTassoc 6.0-10.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_1_15)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 1.0-1.5
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_15_2)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 1.5-2.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_2_3)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 2.0-3.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_3_4)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 3.0-4.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_4_5)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc    4.0-5.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_5_6)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 5.0-6.0
initial_parameter_defaults[(AnalysisType.CENTRAL, TriggerJetMomentumBin.PT_40_60, AssociatedHadronMomentumBin.PT_6_10)] = [
    1,
    0.02,
    0.005,
    0.02,
    0.2,
    0.03,
]  # pTtrig 40-60, pTassoc 6.0-10.0

bounds = [[0.0,  -0.2, -0.2, -0.1, -0.005, -0.02], [1e8,  0.2, 0.3, 0.10, 0.80, 0.15]]

resolution_parameters = {}
resolution_parameters[AnalysisType.SEMICENTRAL] = OrderedDict([
    ("R2", 0.7703651242647157),
    ("R4", 0.5046126852106662),
    ("R6", 0.3020062445564112),
    ("R8", 0),
])
resolution_parameters[AnalysisType.CENTRAL] = OrderedDict([
    ("R2", 0.6192508430757114),
    ("R4", 0.34878092755772117),
    ("R6", 0.18777865138044672),
    ("R8", 0),
])

def vtilde( n, phis, c, R2, R4, R6, R8):
    def vtilde_( v2, v3, v4):
        denom = (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)
        if n<=0 or n>=5: return 0
        if n==1:
            return 0 #v1
        if n==2:
            return (v2 + np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + (v4)*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + (v2)*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4 + (v4)*np.cos(6*phis)*np.sin(6*c)/(6*c)*R6) / denom
        if n==3:
            return v3 
        if n==4:
            return (v4 + np.cos(4*phis)*np.sin(4*c)/(4*c)*R4 + (v2)*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + (v2)*np.cos(6*phis)*np.sin(6*c)/(6*c)*R6 + (v4)*np.cos(8*phis)*np.sin(8*c)/(8*c)*R8) / denom
        else:
            return 0
    return vtilde_

def btilde( phis, c, R2, R4, R6, R8):
    def btilde_(B, v2, v4):
        return np.pi*B*(1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)
    return btilde_

def background(phis, c, R2, R4, R6, R8):
    def background_(dPhi, B,  v2, v3, v4,  va2, va4):
        btilde_func = btilde(phis, c, R2, R4, R6, R8)
        return btilde_func(B, v2, v4)*(1+2*vtilde(3, phis, c, R2, R4, R6, R8)( v2, v3, v4)*np.cos(1*dPhi) +2*vtilde(2, phis, c, R2, R4, R6, R8)( v2, v3, v4)*va2*np.cos(2*dPhi) +2*vtilde(3, phis, c, R2, R4, R6, R8)( v2, v3, v4)*np.cos(3*dPhi) + 2*vtilde(4, phis, c, R2, R4, R6, R8)( v2, v3, v4)*va4*np.cos(4*dPhi))
    return background_

def background_err( phis, c, R2, R4, R6, R8):
    
    def background_err_(dPhi, B, v2, v3, v4, va2, va4, pcov):
        bground = background(phis, c, R2, R4, R6, R8)
        btilde_func = btilde(phis, c, R2, R4, R6, R8)

        #dbddPhi = np.pi*btilde_func(B, v2, v4)*(-2*v1*np.sin(1*dPhi) -4*vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.sin(2*dPhi) -6*v3*np.sin(3*dPhi) - 8*vtilde(4, phis, c)(v2, v4)*va4*np.sin(4*dPhi))

        dbdB = bground(dPhi, B, v2, v3, v4, va2, va4)/B

        #dbdv1 = btilde_func(B, v2, v4)*2*vtilde(1, phis, c)(v1, v2, v3, v4)*np.cos(1*dPhi)/v1


        v2_first_term = (-(2*R2*va2*np.cos(2*dPhi)*np.cos(2*phis)*np.sin(2*c)*(v2+R2*np.cos(2*phis)*np.sin(2*c)/(2*c) + R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(c*(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/2*c +2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2))
        v2_second_term = ((2*va2*np.cos(2*dPhi)*(1+R4*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
        v2_third_term = ((2*va4*np.cos(4*dPhi)*(R2*np.cos(2*phis)*np.sin(2*c)/(2*c) + R6*np.cos(6*phis)*np.sin(6*c)/(6*c)))/(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
        v2_fourth_term = -(2*R2*va4*np.cos(4*dPhi)*np.cos(2*phis)*np.sin(2*c)*(v4+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c)))/(c*(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2)
        v2_bonus_term = 1/c*B*R2*np.cos(2*phis)*np.sin(2*c)*(1+2*0*np.cos(dPhi)+2*v3*np.cos(3*dPhi) + (2*va2*np.cos(2*dPhi)*(v2+R2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(1+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))+2*va4*np.cos(4*dPhi)*(v4+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c))/(1+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
        
        dbdv2 = btilde_func(B, v2, v4)*(v2_first_term + v2_second_term + v2_third_term + v2_fourth_term) + v2_bonus_term
        #dbdv2 = btilde_func(B, v2, v4)*2*(va2*np.cos(2*dPhi)*(
        #    (1 + np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)
        #    /
        #    (1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)
        #            -(v2 + np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + (v4)*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + (v2)*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4 )*(2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2))
        #                /
        #                (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)**2 
        #     - va4*np.cos(4*dPhi)*((v4 + np.cos(4*phis)*np.sin(4*c)/(4*c)*R4  )*(2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2))
        #     /
        #     (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)**2) + np.pi*(B*c*2/np.pi*2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2)*(1+2*v1*np.cos(1*dPhi) +2*vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.cos(2*dPhi) +2*v3*np.cos(3*dPhi) + 2*vtilde(4, phis, c)(v1, v2, v3, v4)*va4*np.cos(4*dPhi))

        dbdv3 = btilde_func(B, v2, v4)*vtilde(3, phis, c, R2, R4, R6, R8)(v2, v3, v4)*2*np.cos(3*dPhi)

        v4_first_term = (-(2*R4*va2*np.cos(2*dPhi)*np.cos(4*phis)*np.sin(4*c)*(v2+R2*np.cos(2*phis)*np.sin(2*c)/(2*c) + R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(c*(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/2*c +2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2))
        v4_second_term = ((R2*va2*np.cos(2*dPhi)*np.cos(2*phis)*np.sin(2*c)/(2*c))/(c*(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))))
        v4_third_term = ((2*va4*np.cos(4*dPhi)*(1+R8*np.cos(8*phis)*np.sin(8*c)/(8*c)))/(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
        v4_fourth_term = -(R4*va4*np.cos(4*dPhi)*np.cos(4*phis)*np.sin(4*c)*(v4+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c)))/(c*(1 + 2*R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c) + 2*R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))**2)
        v4_bonus_term = 1/(2*c)*B*R4*np.cos(4*phis)*np.sin(4*c)*(1+2*0*np.cos(dPhi)+2*v3*np.cos(3*dPhi) + (2*va2*np.cos(2*dPhi)*(v2+R2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R2*v4*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v2*np.cos(4*phis)*np.sin(4*c)/(4*c)))/(1+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c))+2*va4*np.cos(4*dPhi)*(v4+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*np.cos(4*phis)*np.sin(4*c)/(4*c)+R6*v2*np.cos(6*phis)*np.sin(6*c)/(6*c)+R8*v4*np.cos(8*phis)*np.sin(8*c)/(8*c))/(1+R2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)+R4*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)))
        
        dbdv4 = btilde_func(B, v2, v4)*(v4_first_term + v4_second_term + v4_third_term + v4_fourth_term) + v4_bonus_term

        #dbdv4 = np.pi*btilde_func(B, v2, v4)*2*(va2*np.cos(2*dPhi)*(
        #    (np.cos(2*phis)*np.sin(2*c)/(2*c)*R2)*(1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4) 
        #    - (v2 +np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 +v4*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 +v2*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)*(2*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4))
        #    /
        #    (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)**2 
        #    + va4*np.cos(4*dPhi)*((1+2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)-(v4+np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)*(2*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4))
        #    /
        #    (1 + 2*v2*np.cos(2*phis)*np.sin(2*c)/(2*c)*R2 + 2*v4*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)**2) + np.pi*(B*c*2/np.pi*2*np.cos(4*phis)*np.sin(4*c)/(4*c)*R4)*(1+2*v1*np.cos(1*dPhi) +2*vtilde(2, phis, c)(v1, v2, v3, v4)*va2*np.cos(2*dPhi) +2*v3*np.cos(3*dPhi) + 2*vtilde(4, phis, c)(v1, v2, v3, v4)*va4*np.cos(4*dPhi))

        dbdva2 = btilde_func(B, v2, v4)*2*vtilde(2, phis, c, R2, R4, R6, R8)( v2, v3, v4)*np.cos(2*dPhi)

        dbdva4 = btilde_func(B, v2, v4)*2*vtilde(4, phis, c, R2, R4, R6, R8)( v2, v3, v4)*np.cos(4*dPhi)

        err = np.sqrt( (dbdB**2 * pcov[0,0]) + (dbdv2**2 * pcov[1,1]) + (dbdv3**2 * pcov[2,2]) + (dbdv4**2 * pcov[3,3]) + (dbdva2**2 * pcov[4,4]) + (dbdva4**2 * pcov[5,5]) +  \
            + 2*(dbdB*dbdv2 * pcov[0,1]) + 2*(dbdB*dbdv3 * pcov[0,2])+2*(dbdB*dbdv4 * pcov[0,3])+2*(dbdB*dbdva2 * pcov[0,4])+2*(dbdB*dbdva4 * pcov[0,5])\
                    + 2*(dbdv2*dbdv3 * pcov[1,2])+2*(dbdv2*dbdv4 * pcov[1,3])+2*(dbdv2*dbdva2 * pcov[1,4])+2*(dbdv2*dbdva4 * pcov[1,5])
                    + 2*(dbdv3*dbdv4 * pcov[2,3])+2*(dbdv3*dbdva2 * pcov[2,4])+2*(dbdv3*dbdva4 * pcov[2,5])
                    + 2*(dbdv4*dbdva2 * pcov[3,4])+2*(dbdv4*dbdva4 * pcov[3,5])
                    + 2*(dbdva2*dbdva4 * pcov[4,5]))
                    
        return err
    return background_err_

def simultaneous_fit(non_zero_masks, R2, R4, R6, R8, dPhi, B, v2, v3, v4, va2, va4):
    in_plane_func = background(0, np.pi/6, R2, R4, R6, R8)
    mid_plane_func = background(np.pi/4, np.pi/6, R2, R4, R6, R8)
    out_plane_func = background(np.pi/2, np.pi/6, R2, R4, R6, R8)
    
    if non_zero_masks is None:
        warnings.warn("non_zero_masks is None, using all data points")
        non_zero_masks = [np.ones(len(dPhi), dtype=bool)] * 3

    return np.hstack([in_plane_func(dPhi[non_zero_masks[0]], B, v2, v3, v4, va2, va4), mid_plane_func(dPhi[non_zero_masks[1]], B, v2, v3, v4, va2, va4), out_plane_func(dPhi[non_zero_masks[2]], B, v2, v3, v4, va2, va4)])

def simultaneous_err(non_zero_masks, R2, R4, R6, R8, dPhi, B, v2, v3, v4, va2, va4,  pcov):
    in_plane_err = background_err(0, np.pi/6, R2, R4, R6, R8)
    mid_plane_err = background_err(np.pi/4, np.pi/6, R2, R4, R6, R8)
    out_plane_err = background_err(np.pi/2, np.pi/6, R2, R4, R6, R8)

    if non_zero_masks is None:
        warnings.warn("non_zero_masks is None, using all data points")
        non_zero_masks = [np.ones(len(dPhi), dtype=bool)] * 3

    return np.hstack([in_plane_err(dPhi[non_zero_masks[0]], B, v2, v3, v4, va2, va4, pcov), mid_plane_err(dPhi[non_zero_masks[1]], B, v2, v3, v4, va2, va4, pcov), out_plane_err(dPhi[non_zero_masks[2]], B, v2, v3, v4, va2, va4, pcov)])