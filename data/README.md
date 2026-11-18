# Data

The extracted features (~3 GB) are not versioned. Place these files in this folder to run the notebook:

```
data/
├── train_features.csv               # 34,632 visual features per poster + imdbId
└── features_reduced_UMAP_L1.csv     # 2-D embedding used for clustering
```

Source data: MovieLens (`links.csv`), the IMDb-based `MovieGenre.csv` (poster URLs), and the TMDb API for missing posters.
