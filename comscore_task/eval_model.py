import argparse
import pickle as pkl

import pandas as pd

from comscore_task.settings import MODELS_DIR, reverse_class_mapping
from comscore_task.utils import preprocess_text


def main():
    parser = argparse.ArgumentParser(
        description='Program evaluating classification model'
    )
    parser.add_argument('-i', '--input_filename', type=str,
                        help="Input parquet file with post to classifify")
    parser.add_argument('-o', '--output_filename', type=str,
                        help="Direction of output file")

    args = parser.parse_args()

    input_path = args.input_filename
    output_path = args.output_filename

    print(f'Input file: {input_path}')
    print(f'Output file: {output_path}')

    print('Loading TF-IDF vectorizer... ')
    with open(MODELS_DIR / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pkl.load(f)
    print('Done!')

    print('Loading SVD model... ')
    with open(MODELS_DIR / 'svd.pkl', 'rb') as f:
        svd = pkl.load(f)
    print('Done!')

    print('Loading XGBoost model... ')
    with open(MODELS_DIR / 'xgb_model.pkl', 'rb') as f:
        model = pkl.load(f)
    print('Done!')

    df = pd.read_parquet(input_path)
    df['preprocessed_post'] = df['post_content'].apply(lambda text: preprocess_text(text))

    tfidf_sparse = tfidf_vectorizer.transform(df['preprocessed_post'])
    x_svd_tfidf = svd.transform(tfidf_sparse)
    y_pred = model.predict(x_svd_tfidf)

    y_pred_df = pd.DataFrame(y_pred, columns=['platform'])
    y_pred_df['platform'].replace(reverse_class_mapping, inplace=True)
    y_pred_df.to_parquet(output_path)


if __name__ == '__main__':
    main()
