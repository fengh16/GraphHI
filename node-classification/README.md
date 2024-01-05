# GraphHI

## Environment

Please refer to `README.md` in the root directory of this project.


## Download the inter model data

To compare with baselines, we utilize the teacher knowledge from GraphAKD as the inter model data (more details in https://github.com/MIRALab-USTC/GraphAKD/blob/main/download_teacher_knowledge.py ).

Use the following command to download inter model data from Google Drive into `./inter_model`.

```bash
python download_inter_model.py --data_name=<dataset>
```


## Reproduce the results

Run the following command in the directory containing this `README.md`. Please change the `--data_dir` and `--inter_model_dir` to your own path.

### Cora

``` bash
cd stu-gcn && CUDA_VISIBLE_DEVICES=0 python train.py --dataset cora --dropout=0.1 --lr=0.001 --n-epochs=600  --n-hidden=64 --n-runs=10 --GraphHI_alpha=9 --GraphHI_CL_weight=20 --GraphHI_CL_temp=1 --GraphHI_CL_sample_num=10 --GraphHI_attention_temp_smooth_addition_new_weight=1.0 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### CiteSeer

``` bash
cd stu-gcn && CUDA_VISIBLE_DEVICES=1 python train.py --dataset citeseer --dropout=0.1 --lr=0.01 --n-epochs=1000 --n-hidden=256 --n-runs=10 --GraphHI_alpha=10 --GraphHI_CL_weight=10 --GraphHI_CL_temp=1 --GraphHI_CL_sample_num=10 --GraphHI_attention_temp_smooth_addition_new_weight=0.5 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### PubMed

``` bash
cd stu-gcn && CUDA_VISIBLE_DEVICES=2 python train.py --dataset pubmed --dropout=0.1 --lr=0.001 --n-epochs=600 --n-hidden=256 --n-runs=10 --GraphHI_alpha=5 --GraphHI_CL_weight=50 --GraphHI_CL_temp=1 --GraphHI_CL_sample_num=10 --GraphHI_attention_temp_smooth_addition_new_weight=0.1 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### Arxiv

``` bash
cd stu-cluster-gcn && CUDA_VISIBLE_DEVICES=3 python train.py -d=ogbn-arxiv --n-runs=10 --n-epochs=2000 --use-labels --use-linear --lr=0.01 --n-hidden=256 --dropout=0.1 --num_partitions=200 --batch-size=32 --GraphHI_alpha=1 --GraphHI_CL_weight=1  --GraphHI_CL_temp=0.8 --GraphHI_CL_sample_num=10 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### Flickr

``` bash
cd stu-gcn && CUDA_VISIBLE_DEVICES=4 python train.py --dataset flickr --dropout=0.1 --lr=0.001 --n-epochs=2000 --n-hidden=256 --n-runs=10 --GraphHI_alpha=0.5 --GraphHI_CL_weight=0.5 --GraphHI_CL_temp=0.8 --GraphHI_CL_sample_num=10 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --GraphHI_attention_temp_smooth_addition_new_weight=0.1 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### Reddit

``` bash
cd stu-gcn && CUDA_VISIBLE_DEVICES=5 python train.py --dataset reddit --dropout=0.1 --lr=0.005 --n-epochs=1500 --n-hidden=256 --n-runs=10 --GraphHI_alpha=0.25 --GraphHI_CL_weight=0.25 --GraphHI_CL_temp=0.8 --GraphHI_CL_sample_num=10 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --GraphHI_attention_temp_smooth_addition_new_weight=0.1 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### Yelp

``` bash
cd stu-cluster-gcn && CUDA_VISIBLE_DEVICES=6 python train.py -d=yelp --n-runs=10 --n-epochs=500 --use-linear --lr=0.0004 --n-hidden=512 --dropout=0.05 --num_partitions=120 --batch-size=32 --GraphHI_alpha=0.4 --GraphHI_CL_weight=1 --GraphHI_CL_temp=0.8 --GraphHI_CL_sample_num=10 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --GraphHI_attention_temp_smooth_addition_new_weight=0.1 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```

### Products

``` bash
cd stu-cluster-gcn && CUDA_VISIBLE_DEVICES=7 python train.py -d=ogbn-products --n-runs=10 --n-epochs=400 --lr=0.005 --n-hidden=512 --dropout=0.1 --num_partitions=160 --batch-size=4 --GraphHI_alpha=1 --GraphHI_CL_weight=0.5 --GraphHI_CL_temp=0.8 --GraphHI_CL_sample_num=10 --role=GraphHI --GraphHI_attention_temp_range_min 0.1 --GraphHI_attention_temp_range_max 100 --data_dir /f/code/dataset/ --inter_model_dir /f/code/GraphHI/inter_model/
```
