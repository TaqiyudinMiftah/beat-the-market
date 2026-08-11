# Data sources and provenance

Retrieved 11 August 2026 (Asia/Jakarta).

- **Universe:** the IDX30 constituents effective 3 August–30 October 2026. The
  full table is transcribed in `data/universe_idx30_2026-08.csv` from the
  28 July 2026 report by [Katadata](https://katadata.co.id/finansial/bursa/6a6801332b41a/daftar-lengkap-saham-penghuni-idx30-bei-depak-ptba-emiten-bakrie-dewa-masuk).
  The official IDX July 2026 fact sheet is cached at
  `data/raw/idx30_2026-07.pdf`.
- **Index definitions:** [IDX stock index descriptions](https://www.idx.id/en/products/index/)
  and [IDX80/LQ45/IDX30 methodology](https://www.idx.id/media/i2sd4vsk/appendix-index-guide-methodology-idx80-lq45-and-idx30.pdf).
- **Prices and volume:** Yahoo Finance chart endpoint, with one CSV per symbol
  under `data/raw/yahoo/`. Indonesian equities are requested with the `.JK`
  Yahoo suffix; the benchmark is `^JKSE`. The raw request metadata is stored in
  `data/raw/yahoo_metadata.json`.

Yahoo data are convenient for research but are not a substitute for a licensed
exchange feed. The cached panel should be refreshed and independently checked
before any live use.
