# Dataset

- Downloaded from [kaggle](https://www.kaggle.com/datasets/kartikeychauhan/movie-plots).
- The movie_plots500.csv file has been altered to have fewer rows so that it could be uploaded to github.
- load data run `python3 loadData.py`.

# Setup

- Python version 3.12.8
- Requires numpy (comes with anaconda python environment or use `pip install numpy`).
- Run loadData.py after any changes to dataset or if preprocessed.csv or tfidf_data.pkl are missing.

# Running

- Use `python3 recommender.py` then enter your query.

# Results

- Sample Input: "I love thrilling action movies set in space, with a comedic twist."
- Output:
`Here are my top  3  recommendations for you:
Movie Title:  way... way out
TF-IDF Similarity Score:  0.07298806265080758

Movie Title:  vague stars of ursa...
TF-IDF Similarity Score:  0.050747406170966326

Movie Title:  motorpsycho!
TF-IDF Similarity Score:  0.044036393310579576`
- [demo.md](https://github.com/user-attachments/files/18924064/demo.md)

