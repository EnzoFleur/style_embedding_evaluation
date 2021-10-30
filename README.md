# Style Evaluation for Author Embedding
This repository contains a framework to perform author embedding evaluation on a writing style axis.

## Requirements
```bash
spacy >= 3.0.1
nltk >= 3.5
```

## Description
### Feature extraction
It first extracts for a raw text corpus stylistic features. Data directory should be organized as follow (see dataset directory for example) :

```bash
.\data_dir\author1
          \author1\text1.txt
          \author1\text2.txt
          ...
.\data_dir\author2
          \author2\text1.txt
          \author2\text2.txt
          ...
...        
```

To perform extraction run :

```bash
from extractor import build_authorship
data_dir = "dataset\\English"
authorship = build_authorship(data_dir)

from extractor import create_stylometrics
stylo_df = create_stylometrics(authorship)
```

Authorship is a dataframe linking each author to its textual production, while stylo_df contains all stylometric features by text.

### Embedding evaluation
To perform embedding evaluation, run the following code with your custom embeddings :

```bash
import numpy as np
embeddings = np.random.randn(3,3,512)

from regressor import style_embedding_evaluation, multi_style_evaluation

# To evaluate a single embedding method
res_df = style_embedding_evaluation(embeddings[0], stylo_df, n_fold=2, output="agg")

# To evaluate several embedding methods
df_results = multi_style_evaluation(embeddings, names["model1", "model2", "model3"], features=stylo_df, n_fold=2)
```

You can then produce a spyder chart as follow : 
```bash
from regressor import style_spyder_charts
style_spyder_charts(df_results)
```

![Alt text](image/spyder_chart.png?raw=true "Spyder chart example")

