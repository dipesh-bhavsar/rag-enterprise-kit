import argparse, json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness
def run_evaluation(dataset_path, output_path=None):
    with open(dataset_path) as f: data=json.load(f)
    result=evaluate(Dataset.from_list(data), metrics=[faithfulness,answer_relevancy,context_precision])
    df=result.to_pandas(); print(df.to_string())
    if output_path: df.to_csv(output_path,index=False)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--output',default=None)
    args=p.parse_args(); run_evaluation(args.dataset,args.output)
