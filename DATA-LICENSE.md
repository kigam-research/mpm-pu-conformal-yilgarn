# Data Licence and Provenance

The MIT licence in `LICENSE` applies to the source code of this repository only.
The data files follow the terms of their source custodians, as set out below.
Column names refer to `data/Yilgarn_GIS_DB.csv` (70,200 grid cells at 5 km resolution).

## Label column

| Column | Source | Licence | Attribution | Extract date |
|---|---|---|---|---|
| `Ni_mine` | Nickel site records of the Mines and Mineral Deposits Database (MINEDEX), <https://catalogue.data.wa.gov.au/dataset/minedex-dmirs-001>, and the komatiite-hosted nickel layers of the GSWA Mineral Systems Atlas, <https://msamaps.dmp.wa.gov.au/msamaps/> | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** | Geological Survey of Western Australia / Department of Energy, Mines, Industry Regulation and Safety | 2024-07-04 (MINEDEX), 2024-06-07 (Atlas) |

The CC BY-NC 4.0 non-commercial condition applies to the `Ni_mine` column and to any
derivative of it, including the trained-model outputs and zone assignments produced by
this code when run on this dataset.

## Predictor columns

| Columns | Source | Licence | Attribution |
|---|---|---|---|
| `TMI`, `RTP`, `1VD`, `K_pct`, `Th_ppm`, `U_ppm`, `worms_mag_*`, `worms_grav_*` | Western Australian airborne magnetic, gravity, and radiometric grid compilations and multi-scale worm products, via the GSWA MAGIX register, <https://magix.dmirs.wa.gov.au> | Creative Commons Attribution 4.0 International (CC BY 4.0) | © State of Western Australia (Department of Mines, Petroleum and Exploration) |
| `DGIR`, `AuSREM` | Geoscience Australia eCat Product Catalogue, <https://ecat.ga.gov.au> | Creative Commons Attribution 4.0 International (CC BY 4.0) | © Commonwealth of Australia (Geoscience Australia) |
| `Prox_Ultramafic Source`, `Prox_MajorCrustal`, `Prox_Fault`, `Tectonic Bedrock Age (MA)`, `Interpreted Bedrock Age (MA)` | Derived from the GSWA 1:500,000 State Interpreted Bedrock Geology and structural databases | Source custodian terms (GSWA) | © State of Western Australia |

## Boundary data

`data/boundary/gadm41_AUS_0.*` are redistributed from GADM (<https://gadm.org>) and are
subject to the GADM terms of use (academic and non-commercial use).

## Notes

- The source catalogue pages for the label data were checked on 2026-08-10
  (MINEDEX catalogue entry: licence CC BY-NC 4.0, publisher DEMIRS, access Open).
- The extracts bundled here are snapshots; the live databases are updated by their
  custodians and may differ from the 2024 extracts.
