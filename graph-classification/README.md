## Environment

Please refer to `README.md` in the root directory of this project.



## Note for the metric

The metric is `AP` for dataset `Molpcba`.



## Download the inter model data

To compare with baselines, we utilize the teacher knowledge from GraphAKD as the inter model data (more details in https://github.com/MIRALab-USTC/GraphAKD/blob/main/download_teacher_knowledge.py ).

Please download file `MOLPCBA-knowledge.pth.tar` from https://cloud.189.cn/web/share?code=aA3YZf6F3iqa（访问码：k2lo）. Then put it in the folder specified by the argument `--inter_model_dir` (such as `/f/code/GraphHI/inter_model/`).



## Scripts

Run the following command in the directory containing this `README.md`. Please change the `--data_dir` and `--inter_model_dir` to your own path.

### Use GCN as Model $f$

``` bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbg-molpcba --gnn gcn --emb_dim=1024 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001 --role=GraphHI --data_dir /f/code/dataset/ --kd_dir /f/code/GraphHI/inter_model/ --GraphHI_alpha 0.9
```

### Use GIN as Model $f$

``` bash
CUDA_VISIBLE_DEVICES=1 python main.py --dataset ogbg-molpcba --gnn gin --emb_dim=1024 --epochs=300 --drop_ratio=0.1 --batch_size=512 --lr=0.001 --role=GraphHI --data_dir /f/code/dataset/ --kd_dir /f/code/GraphHI/inter_model/
```
