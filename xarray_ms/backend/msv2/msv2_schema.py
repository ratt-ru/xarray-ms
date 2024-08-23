# https://casa.nrao.edu/Memos/229.html#SECTION00061000000000000000
MS_SCHEMA = {
  "UVW": {"dims": ("uvw",)},
  "UVW2": {"dims": ("uvw",)},
  "DATA": {"dims": ("freq", "pol")},
  "FLOAT_DATA": {"dims": ("freq", "pol")},
  "SIGMA": {"dims": ("pol",)},
  "SIGMA_SPECTRUM": {"dims": ("freq", "pol")},
  "WEIGHT": {"dims": ("pol",)},
  "WEIGHT_SPECTRUM": {"dims": ("freq", "pol")},
  "FLAG": {"dims": ("freq", "pol")},
  # Extra imaging columns
  "MODEL_DATA": {"dims": ("freq", "pol")},
  "CORRECTED_DATA": {"dims": ("freq", "pol")},
  "IMAGING_WEIGHT": {"dims": ("freq",)},
}
