import argparse
import gdown
import os


urls_fns_dict = {
    "cora": [("https://drive.google.com/u/0/uc?id=1zTKmuRJE8-Gn4FR5M6tQdh1levF5zrBU&export=download", "cora-knowledge.pth.tar")],
    "citeseer": [("https://drive.google.com/u/0/uc?id=17oGVdaX7RNf2vq97p_YBGVqbcWc38Tmd&export=download", "citeseer-knowledge.pth.tar")],
    "pubmed": [("https://drive.google.com/u/0/uc?id=1_KTpxSYTsaF6GgOu9v0sY8bGB_1YYtJS&export=download", "pubmed-knowledge.pth.tar")],
    "flickr": [("https://drive.google.com/u/0/uc?id=1McII0murqYYSZhIdquAPvpFABDHSjYCI&export=download", "flickr-knowledge.pth.tar")],
    "arxiv": [("https://drive.google.com/u/0/uc?id=1QYYWGRDMV6gXBpXoUb73C8qhC5Zls--K&export=download", "arxiv-knowledge.pth.tar")],
    "reddit": [("https://drive.google.com/u/0/uc?id=1g7FMyMaSTZIK6gJr9bqF3qh1F2X4FzUs&export=download", "reddit-knowledge.pth.tar")],
    "yelp": [("https://drive.google.com/u/0/uc?id=1UFfIMG4n7yyjlUbcOqpwlYeDD1W4S-wf&export=download", "yelp-knowledge.pth.tar")],
    "products": [("https://drive.google.com/u/0/uc?id=1P_lPCJ8l1J8B4EpBTmJ6MyrHQXezuBXZ&export=download", "products-knowledge.pth.tar")],
}


def parse_args():
    parser = argparse.ArgumentParser("download_inter_model.py", conflict_handler="resolve")
    parser.add_argument("--data_name", help="data name", type=str, default="",
                        choices=["cora", "citeseer", "pubmed", "flickr", "arxiv", "reddit", "yelp", "products"])

    return parser.parse_args()


def main():
    args = parse_args()
    inter_model_path = "./inter_model"
    os.makedirs(inter_model_path, exist_ok=True)

    for url, fn in urls_fns_dict[args.data_name]:
        ofn = os.path.join(inter_model_path, fn)
        if not os.path.exists(ofn):
            gdown.download(url, ofn, quiet=False)
            assert os.path.exists(ofn)
        else:
            print(f"{ofn} exists, skip downloading")


if __name__ == "__main__":
    main()
