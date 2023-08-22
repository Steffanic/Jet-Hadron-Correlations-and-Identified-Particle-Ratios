from enum import Enum
from math import pi
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
    BACKGROUND =5
    INCLUSIVE = 6

class AssociatedHadronMomentumBin(Enum):
    PT_1_15 = 1
    PT_15_2 = 2
    PT_2_3 = 3
    PT_3_4 = 4
    PT_4_5 = 5
    PT_5_6 = 6
    PT_6_10 = 7

class OtherTOFRangeTags(Enum):
    P_HI = 1
    P_LO_K_HI = 2
    PI_LO_P_LO_K_LO = 3
    PI_HI_P_LO_K_LO = 4

regionDeltaPhiRangeDictionary = {
    Region.NEAR_SIDE_SIGNAL: [-pi / 2, pi / 2],
    Region.AWAY_SIDE_SIGNAL: [pi / 2, 3 * pi / 2],
    Region.BACKGROUND_ETAPOS: [-pi / 2, pi / 2],
    Region.BACKGROUND_ETANEG: [-pi / 2, pi / 2],
    Region.INCLUSIVE: [-pi / 2, 3 * pi / 2]
}

regionDeltaEtaRangeDictionary = {
    Region.NEAR_SIDE_SIGNAL: [-0.6, 0.6],
    Region.AWAY_SIDE_SIGNAL: [-1.2, 1.2],
    Region.BACKGROUND_ETAPOS: [0.8, 1.2],
    Region.BACKGROUND_ETANEG: [-1.2, -0.8],
    Region.INCLUSIVE: [-1.4, 1.4]
}

regionDeltaPhiBinCountsDictionary = {
    Region.NEAR_SIDE_SIGNAL: 36,
    Region.AWAY_SIDE_SIGNAL: 36,
    Region.BACKGROUND_ETAPOS: 36,
    Region.BACKGROUND_ETANEG: 36,
    Region.INCLUSIVE: 72
}

speciesTOFRangeDictionary = {
    ParticleType.PION: ([-2,2], [-10,-2], [-10,-2]),#  -10 includes the underflow bin
    ParticleType.KAON: ([-10,10], [-2,2], [-10,-2]),
    ParticleType.PROTON: ([-10,10], [-10,10], [-2,2]),
    ParticleType.INCLUSIVE: ([-10,10], [-10,10], [-10,10]),
}

associatedHadronMomentumBinRangeDictionary = {
    AssociatedHadronMomentumBin.PT_1_15: (1, 1.5),
    AssociatedHadronMomentumBin.PT_15_2: (1.5, 2),
    AssociatedHadronMomentumBin.PT_2_3: (2, 3),
    AssociatedHadronMomentumBin.PT_3_4: (3, 4),
    AssociatedHadronMomentumBin.PT_4_5: (4, 5),
    AssociatedHadronMomentumBin.PT_5_6: (5, 6),
    AssociatedHadronMomentumBin.PT_6_10: (6, 10),
}

# Tuples are (min_pi, max_pi), (min_k, max_k), (min_p, max_p)
OtherTOFRangeDictionary = {
    OtherTOFRangeTags.P_HI: [(-10,10), (-10,10), (2,10)],
    OtherTOFRangeTags.P_LO_K_HI: [(-10,10), (2,10), (-10,-2)],
    OtherTOFRangeTags.PI_LO_P_LO_K_LO: [(-10,-2), (-10,-2), (-10,-2)],
    OtherTOFRangeTags.PI_HI_P_LO_K_LO: [(2,10), (-10,-2), (-10,-2)]
}