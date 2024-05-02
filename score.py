import pandas as pd
import evaluate
from pprint import pprint
from argparse import ArgumentParser

squad_metric = evaluate.load("squad_v2")
def score(preds, refs):
    preds = [{'id': str(idx), 'prediction_text': ans, 'no_answer_probability': 0.0} for idx, ans in enumerate(preds)]
    refs = [{'id': str(idx), 'answers': {'answer_start': [0] * len(ans), 'text': ans}} for idx, ans in enumerate(refs)]
    results = squad_metric.compute(predictions=preds, references=refs)
    return dict(f1=results['f1'], exact_match=results['exact'], total=results['total']) # type:ignore

if __name__ == '__main__':
    parser = ArgumentParser('squad scorer', description="scores predicted answers a la Squad 2.0")

    parser.add_argument('--golds', type=str, required=True, help="path to golds.csv (e.g., 'val_questions.csv')")
    parser.add_argument('--preds', type=str, required=True, help="path to preds.csv (must have an 'answer' column)")

    args = parser.parse_args()

    ps = pd.read_csv(args.preds, index_col=0)
    gs = pd.read_csv(args.golds, index_col=0, converters={'answer': eval})

    pprint(score(ps['answer'], gs['answer']))

