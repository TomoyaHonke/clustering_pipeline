from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

@dataclass
class CorrfuncConfig:
    nthreads: int = 56
    mu_max: float = 1.0
    nmu_bins: int = 20
    binfile: np.ndarray = field(
        default_factory=lambda: np.linspace(20, 200, 46)
    ) # s bins [Mpc/h]
    output_savg: bool = False
    mode: str = "radec"　 # "radec" or "xyz"
    
    @property
    def ns(self):
        return len(self.binfile) - 1

@dataclass # NOTE: not used in current pipeline
class PathConfig:
    data_dir: Path = Path("/data/honke/DESI_clu")
    out_dir: Path = Path("/data/honke/corr_clustering")

@dataclass
class AnalysisConfig:
    zbins: Optional[Dict[str, Tuple[float, float]]] = None
    combine_regions: bool = True
    regions: Tuple[str, ...] = ("NGC", "SGC")

    use_jackknife: bool = False
    n_jack: int = 64  # number of jackknife regions

    def get_regions(self):
        return self.regions