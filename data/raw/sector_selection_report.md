# Stock Sector Selection Report

## Objective
This report documents the sector-based stock file organization task completed for the local CS3244 project. The goal was to use the cleaned sector mapping file to group raw Kaggle stock files into sector-specific folders for downstream exploratory analysis and modeling support.

## Input Files and Data Sources
- Mapping file: `stocks_sector_clean_01.csv`
- Raw stock source: local KaggleHub cache for `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`
- Raw stock directory used during organization: `~/.cache/kagglehub/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/versions/3/Stocks`

The mapping file contains four descriptive columns:
- `Ticker`
- `Name`
- `Sector`
- `Description`

## Methodology
The sector selection workflow was carried out as follows:
1. Read `stocks_sector_clean_01.csv` and collect the ticker-to-sector mapping.
2. Identify the five target sectors in the cleaned mapping file.
3. Create a local output directory `sector_stocks/` under `Stock Selection/`.
4. Create one subdirectory per sector using the exact sector names from the mapping file.
5. Scan the raw Kaggle stock directory for ticker files.
6. For each ticker in the mapping file, locate the corresponding stock file in the Kaggle dataset and copy it into the correct sector folder.
7. Log any tickers that exist in the mapping file but do not exist in the local Kaggle dataset into `missing_tickers.log`.

A deliberate design decision was made to keep the original Kaggle raw dataset untouched and outside the project folder. Only the organized sector outputs were stored in the local project. This keeps the local project easier to manage while still preserving reproducibility.

## Sector Coverage Summary
The cleaned mapping file contains 125 tickers in total, evenly split across five sectors.

| Sector | Tickers in mapping | Files found and copied | Missing |
| --- | ---: | ---: | ---: |
| Banks | 25 | 23 | 2 |
| Energy | 25 | 22 | 3 |
| Retail | 25 | 24 | 1 |
| Shipping | 25 | 24 | 1 |
| Tech | 25 | 25 | 0 |
| Total | 125 | 118 | 7 |

## Missing Tickers
The following tickers were listed in the cleaned mapping file but could not be found in the local Kaggle stock dataset:
- `SHEL` (Energy)
- `TTE` (Energy)
- `EQNR` (Energy)
- `MUFG` (Banks)
- `TFC` (Banks)
- `GAP` (Retail)
- `TRMD` (Shipping)

These gaps are likely caused by ticker-name changes, historical renaming, dataset age, or missing coverage in the particular Kaggle stock dump.

## Output Location
The resulting organized files are stored in:
- `Stock Selection/sector_stocks/Banks/`
- `Stock Selection/sector_stocks/Energy/`
- `Stock Selection/sector_stocks/Retail/`
- `Stock Selection/sector_stocks/Shipping/`
- `Stock Selection/sector_stocks/Tech/`

The missing ticker log is stored in:
- `Stock Selection/sector_stocks/missing_tickers.log`

## Notes and Interpretation
This sector organization step is a data management task rather than a modeling step. Its main value is to make later sector-level EDA and comparison easier, because all selected raw stock files are now grouped consistently by sector. The current result is complete enough for downstream analysis, with 118 out of 125 mapped tickers successfully matched and organized.

If needed later, the missing tickers can be investigated by checking whether the Kaggle dataset uses older ticker symbols or alternative naming conventions.
