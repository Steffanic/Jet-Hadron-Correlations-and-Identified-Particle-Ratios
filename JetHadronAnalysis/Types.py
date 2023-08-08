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