# Dataset preparation

This project does not download catalogues or images automatically. Keep source
files under `data/raw/` and generated artifacts under `data/processed/`; data
contents are excluded from version control.

## Photometry and labels

Prepare an SDSS-derived CSV with one row per galaxy, the configured numeric
photometric columns, a spectroscopic redshift target, and an object identifier.
Change `data.target_column`, `data.photometric_feature_columns`, and
`data.photometric_object_id_column` in `configs/config.yaml` to match the
catalogue. Record the release, query, selection cuts, units, and provenance in
your project notes.

## Images and cross-identification

Prepare an image metadata CSV containing the configured image object identifier
and a local `image_path` for each usable image. Galaxy Zoo image identifiers,
SDSS `objid` values, and catalogue identifiers are not universally
interchangeable. Do not join on similarly named identifiers without verifying
their meaning and source provenance. If direct identifiers are not shared,
cross-match by sky coordinates using a documented catalogue/release, coordinate
reference system, angular tolerance, ambiguity policy, and quality cuts.
Validate the match against known objects or an authoritative cross-match table.

Only after that verification, set `data.source_compatibility_verified: true`.
The merger performs an exact, one-to-one identifier join and reports its
matched count and rates; it does not perform astronomical coordinate matching
or certify source compatibility for you. Review the match report before
training. Duplicate identifiers and missing/invalid identifiers are rejected,
not silently joined.

## Reproducibility and licensing

Retain source URLs, release/version, query code, selection criteria, and
applicable usage/licensing terms alongside your research documentation. The
project does not ship data or assert that any particular cross-match is valid.

