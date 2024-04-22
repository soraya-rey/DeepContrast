import os
import numpy as np
import pandas as pd
import glob
import yaml
import argparse
from time import gmtime, strftime
from datetime import datetime
import timeit
from prediction.data_prepro import data_prepro
from prediction.model_pred import model_pred


if __name__ == '__main__':
    print("retrieve comand arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--body_part',
        type = str,
        choices = ['HeadNeck', 'Chest'],
        help = 'Specify the body part ("HeadNeck" or "Chest") to run inference.'
        )
    parser.add_argument(
        '-v',
        '--save_csv',
        required = False,
        help = 'Specify True or False to save csv for contrast prediction.',
        default = 'False'
        )
    parser.add_argument(
        '--project_dir',
        type=str,
        help='Specify the project directory.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Specify the data directory.'
    )
    args = parser.parse_args()    
    print("\n --- Starting inference --- \n ")
    proj_dir = "./DeepContrast"
    model_dir = "./DeepContrast/models"
    data_dir = "./DeepContrast/datasets/Chest"
    print("model dir ",model_dir)
    print("data dir ",data_dir)

    print('\n--- MODEL INFERENCE ---\n')
    
    # data preprocessing
    print("data preprocessing")
    df_img, img_arr = data_prepro(
        body_part=args.body_part,
        data_dir=args.data_dir,
        )

    # model prediction
    print("model inference")
    model_pred(
        body_part=args.body_part,
        save_csv=args.save_csv,
        model_dir=model_dir,
        out_dir=args.project_dir,
        df_img=df_img,
        img_arr=img_arr
        )
    
    print('Model prediction done!')
