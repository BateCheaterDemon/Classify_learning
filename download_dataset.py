# 在Python中使用
import opendatasets as od
od.download(dataset_id_or_url="https://www.kaggle.com/c/classify-leaves",
            data_dir="./data/leaves")