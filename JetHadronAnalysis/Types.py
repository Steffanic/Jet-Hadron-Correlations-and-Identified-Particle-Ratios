from enum import Enum

class AnalysisType(Enum):
    PP = 1
    SEMICENTRAL = 2
    CENTRAL = 3

class ParticleType(Enum):
    PION = 1
    PROTON = 2
    KAON = 3
    OTHER = 4
    INCLUSIVE = 5

class NormalizationMethod(Enum):
    SLIDING_WINDOW = 1
    MAX = 2

class Region(Enum):
    NEAR_SIDE_SIGNAL = 1
    AWAY_SIDE_SIGNAL = 2
    BACKGROUND_ETAPOS = 3
    BACKGROUND_ETANEG = 4
    INCLUSIVE = 5

class AssociatedHadronMomentumBin(Enum):
    PT_1_15 = 1
    PT_15_2 = 2
    PT_2_3 = 3
    PT_3_4 = 4
    PT_4_5 = 5
    PT_5_6 = 6
    PT_6_10 = 7