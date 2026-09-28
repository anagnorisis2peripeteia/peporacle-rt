# Data

The nine retention-time datasets are not stored in this repository. Download them with

```bash
python scripts/download_data.py
```

which fetches them from the [PepMNet repository](https://github.com/danielgarzonotero/PepMNet)
(commit `2986c9f`) and checks each file against the SHA-256 below. These are the exact files
the published results were trained on.

## Sources

The datasets were assembled from published studies by Ma et al. for DeepRT+ and later
redistributed as CSVs by PepMNet. Both repositories are MIT-licensed, but DeepRT+ states
that the copyright of the data belongs to the original authors and asks users to cite them.
If you use the data, cite the original study:

| file | peptides | chromatography | original study |
|---|---:|---|---|
| `hela_mod.csv` | 3,413 | RPLC | Sharma et al., *Cell Rep.* 2014, 8, 1583–1594. [10.1016/j.celrep.2014.07.036](https://doi.org/10.1016/j.celrep.2014.07.036) |
| `hela_mod3.csv` | 1,170 | RPLC | as above; the peptides of `hela_mod.csv` without modified residues |
| `yeast_unmod.csv` | 14,361 | RPLC | Nagaraj et al., *Mol. Cell. Proteomics* 2012, 11, M111.013722. [10.1074/mcp.M111.013722](https://doi.org/10.1074/mcp.M111.013722) |
| `misc_dia.csv` | 146,587 | RPLC | Rosenberger et al., *Sci. Data* 2014, 1, 140031. [10.1038/sdata.2014.31](https://doi.org/10.1038/sdata.2014.31) |
| `scx.csv` | 30,482 | SCX | Gussakovsky et al., *Anal. Chem.* 2017, 89, 11795–11802. [10.1021/acs.analchem.7b03436](https://doi.org/10.1021/acs.analchem.7b03436) |
| `luna_hilic.csv`, `luna_silica.csv`, `atlantis_silica.csv`, `xbridge_amide.csv` | 36,271 / 37,110 / 39,091 / 40,290 | HILIC | Spicer and Krokhin, *J. Chromatogr. A* 2018, 1534, 75–84. [10.1016/j.chroma.2017.12.046](https://doi.org/10.1016/j.chroma.2017.12.046) |

Curation: Ma et al., *Anal. Chem.* 2018, 90, 10881–10888,
[10.1021/acs.analchem.8b02386](https://doi.org/10.1021/acs.analchem.8b02386)
([DeepRT+](https://github.com/horsepurve/DeepRTplus)). CSV format: Garzon Otero, Akbari and
Bilodeau, *Mol. Syst. Des. Eng.* 2025, 10, 205–218,
[10.1039/d4me00172a](https://doi.org/10.1039/d4me00172a).

In `hela_mod.csv`, modified residues are written as digits: 1 oxidised methionine,
2 phosphoserine, 3 phosphothreonine, 4 phosphotyrosine (`peporacle.sequences.HELA_DIGIT_MAP`).

## Checksums

| dataset | peptides | sha256 |
|---|---|---|
| `hela_mod3.csv` | 1170 | `602320c45d53d6d8d9c2f6bb941d5030d016634841806f18c90fa44194fd78b6` |
| `hela_mod.csv` | 3413 | `4cb4d2d9df19b2a4d4d28e350aedcf90dfe6cabe143bf49fc05b8a0ae4eaa351` |
| `yeast_unmod.csv` | 14361 | `a92d9a134e59b3f62b6ea3b4901da7f0d331d332ca51ca1c1d4a1eb84c35ed25` |
| `scx.csv` | 30482 | `2f094cd0093b81ebaacbdeec2c531bcb9aa808c9d9f349ab9845491abdc3e939` |
| `luna_hilic.csv` | 36271 | `bfedfd20d5511cd40c178fbf202e21b8a661c5430ad71e3810cf55eb343eeb6a` |
| `luna_silica.csv` | 37110 | `808f7c7c67a4161e9f6da42a0f2a32ac8bd1dbf28159d65652d6d0360818a49e` |
| `atlantis_silica.csv` | 39091 | `74bc27752efe4fa430b9df9bcac16ebd2d48478d6a5f8dfdc64e62e19ee64eac` |
| `xbridge_amide.csv` | 40290 | `3643961d0b4612e5923fa0231881274e3c283581758702f17a49c95ec72b20dc` |
| `misc_dia.csv` | 146587 | `cfed677e8baabb716888476a6ab1c2a7ec269758cdbf4909bf7c510257e0c786` |

Monomers come from pyPept's bundled library (`third_party/pyPept/src/pyPept/data/monomers.sdf`).
The caches were built from a file named `rt_monomers.sdf` that is byte-identical to it.
