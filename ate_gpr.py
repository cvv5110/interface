import sys
import numpy as np
import slepc4py
from mpi4py import MPI
from hypate.CoupledSolver import FSTE
from solverGPR import SolverGPR
sys.path.append("/home/cvv5110/Desktop/APUS/fiml/")
from getConditions import getConditions

slepc4py.init(sys.argv)
comm = MPI.COMM_WORLD

Minf = 7.523
Pinf = 3759.678
Tinf = 466.2
del0 = 0.0009095675977560256
Me0  = 7.228500244963227
Tw   = 273.0

cfg = './models/dw2d_l22.slv' # 
mdl = './models/dw2d.mdl'

solF = SolverGPR(Minf=Minf, Pinf=Pinf, Tinf=Tinf, del0=del0, Me0=Me0, level=-1, Tini=Tw)

solopt = {
    'cfgFile'   : cfg,
    'cfgOpts'   : {'stepS'      : 1500,
                   'dtS'        : 0.001,
                   'coupleMode' : 'fst',
                   'coupleSchm' : 'fh',
                   'solFObj'    : solF
                  },
    'mdlFile'   : mdl,
    'mdlOpts'   : {'caseName' : 'ht2d_l22',
                   'Minf'     : Minf,
                   'Pinf'     : Pinf,
                   'Tinf'     : Tinf,
                   'Tw'       : Tw,
                   'meshF'    : None
                  },
    'ifVerb'    : True
    }

solobj = FSTE(solverCls='jenny')
solobj(ifIni=True, ifRun=True, **solopt)

print()
