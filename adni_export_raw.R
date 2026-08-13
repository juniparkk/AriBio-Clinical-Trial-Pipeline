# ============================================================
# ADNI_EXPORT_RAW -- one-time/rerunnable format bridge.
#
# The ADNIMERGE2 R package (raw/clinical/ADNIMERGE2 under ADNI_ROOT)
# ships its raw eCRF tables as R-serialized .rda files. This project's
# preprocessing pipeline is written in Python (pandas), so this script
# reads the small set of raw tables the pipeline actually needs and
# writes them out as flat CSVs into ADNI_ROOT/interim/ -- nothing more.
#
# This is a pure format conversion, not a transformation: no
# filtering, no recoding, no derived columns are added here. raw/
# stays untouched and immutable, exactly as required. interim/ is
# local-only (ADNI_ROOT is not a git repository), rerunnable at any
# time, and safe to delete and regenerate from raw/ alone.
#
# Usage: Rscript adni_export_raw.R
# ============================================================

adni_root <- path.expand("~/Desktop/ADNI")
data_dir <- file.path(adni_root, "raw", "clinical", "ADNIMERGE2", "data")
out_dir <- file.path(adni_root, "interim")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

tables_to_export <- c(
  "REGISTRY", "ROSTER", "PTDEMOG", "DXSUM", "MMSE", "ADAS", "APOERES", "VISITS",
  "UCBERKELEY_AMY_6MM"
)

for (nm in tables_to_export) {
  e <- new.env()
  load(file.path(data_dir, paste0(nm, ".rda")), envir = e)
  df <- get(nm, envir = e)
  out_path <- file.path(out_dir, paste0(nm, ".csv"))
  write.csv(df, out_path, row.names = FALSE, na = "")
  cat(sprintf("wrote %s  (%d rows, %d cols) -> %s\n", nm, nrow(df), ncol(df), out_path))
}

cat("DONE\n")
