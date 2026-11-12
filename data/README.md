# Data

The raw tables are not stored in this repository. They come from the Olist
Brazilian e-commerce public dataset (about 99k orders placed between late 2016
and 2018), published on Kaggle under CC BY-NC-SA 4.0:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

`promiseline fetch` pulls the nine CSV files from a public GitHub mirror,
pinned to one commit so the numbers in the README are reproducible:
https://github.com/tunguyenn99/ecommerce-data-modeling (folder `dataset/`,
commit `63401c2739ae42dd9bbfa7e20227fe7b7de725a3`).

They land in `data/raw/`, which is gitignored. Because of the non-commercial
licence I keep the data out of the repo and only redistribute code.
