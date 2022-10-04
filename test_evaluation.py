import os
import glob
import zipfile
from tqdm import tqdm
import pandas as pd
import pytrec_eval
import time
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())

# fout = open("test_evaluation.csv", "w")


def evaluation(path, stu_id):
    # if stu_id == 'mlg109':
    #     return
    # Load run
    try:

        with open(path, "r") as f_run:
            tf_run = pytrec_eval.parse_run(f_run)

        run_name = path.split("/")[-1]
        qrels_dict = dict()
        for _, r in test_qrel.iterrows():
            qid, docno, label, iteration = r
            if qid not in qrels_dict:
                qrels_dict[qid] = dict()
            qrels_dict[qid][docno] = int(label)

        metrics = {"map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"}
        metrics_name = ["map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"]
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, metrics)

        # test
        tf_evals = evaluator.evaluate(tf_run)

        tf_metric2vals = {m: [] for m in metrics}
        for q, d in tf_evals.items():
            for m, val in d.items():
                tf_metric2vals[m].append(val)

        # Compute average across topics
        write_list = [stu_id, run_name]

        for m in metrics_name:
            score = pytrec_eval.compute_aggregated_measure(m, tf_metric2vals[m])
            write_list.append(str(score))
            print(m, "\t", score)

        # fout.write(",".join(write_list) + "\n")
        # fout.flush()
    except Exception as e:
        print(e)


def unzip_file():
    zip_file_list = glob.glob("student_assignment/*.zip")
    names = []
    for file in tqdm(zip_file_list):
        dir_name = file.split("_")[-1][:6]
        if dir_name == "assign":
            dir_name = file.split("/")[-1][:6]
        names.append(dir_name)
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(f"./student_assignment_unzip/{dir_name}")
    print(len(names))
    print(len(os.listdir("student_assignment_unzip")))


def get_evaluation(stu_id):

    text_file_list = glob.glob(
        f"student_assignment_unzip/{stu_id}/**/*.txt", recursive=True
    )
    if len(text_file_list) == 0:
        print(stu_id)
    for run_file in text_file_list:
        evaluation(run_file, stu_id)


if __name__ == "__main__":

    dirs = os.listdir("student_assignment_unzip")

    test_qrel = pd.read_csv("ir_course_dataset/test_qrel.csv", dtype=str)

    test_qrel = test_qrel.astype({"label": "int32"})

    start = time.time()
    for file in os.listdir("test_run/runs/"):
        print(file)
        evaluation(f"test_run/runs/{file}", "mlg109")
    # evaluation(f"test_run/dlm.run", "mlg109")
    # print("time",time.time() - start)

    # pool = mp.Pool(mp.cpu_count())

    # pool.map(get_evaluation,dirs)
    # for _ in tqdm(pool.imap_unordered(get_evaluation, dirs), total=len(dirs)):
    #     pass
    # pool.close()
    # pool.join()
