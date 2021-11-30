import re
import os
from os import listdir
from os.path import isfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

feature_set = 'sub_corelated_features'
colors = None


# print_type = 'cell_num on wl 50'
# print_type = 'diff cell_num & wl'
print_type = 'comparison attn'

if print_type == 'comparison attn':
    # data_dir = os.path.join(ROOT_DIR, "..", "classic_rnn", 'results', 'testbed', 'pid_kf')
    data_dir = os.path.join(ROOT_DIR, "..", "classic_rnn", 'results', feature_set)
    history_csv_names = [f for f in listdir(data_dir) if
                         isfile(os.path.join(data_dir, f)) and f[-3:] == "csv" and f[:7] == "history"]

    data = []
    labels = []

    dim = 9
    w = 0.75
    dimw = w / dim
    fig, ax = plt.subplots(figsize=(10, 6))

    for (i, history_csv_name) in enumerate(history_csv_names):
        m = re.search(r"wl(\d+)", history_csv_name)
        win_len = int(m.group(1))
        m = re.search(r"(\d+)cell", history_csv_name)
        cell_num = int(m.group(1))
        m = re.search(r"(\d+)lyrs", history_csv_name)
        if m:
            attn_lyrs = int(m.group(1))
        else:
            attn_lyrs = 0
        m = re.search(r"Bi", history_csv_name)
        bi = True if m else False

        if cell_num in [16,32,48] and win_len in [30,60,90]:
            history_df = pd.read_csv(os.path.join(data_dir, history_csv_name))
            # print(history_df.shape)
            # history_df = history_df.iloc[30:, :]
            # print(history_df.shape)

            row_id_min_eval_loss = history_df['eval_loss'].idxmin()
            train_loss = history_df.iloc[row_id_min_eval_loss]['train_loss']
            # val_loss = history_df.iloc[row_id_min_eval_loss]['val_loss']
            eval_loss = history_df.iloc[row_id_min_eval_loss]['eval_loss']
            # last_10_df_avg = history_df.iloc[-5:].mean()
            # avg_eval_loss = last_10_df_avg['eval_loss']
            smallest_10_df_avg = history_df.nsmallest(10, 'eval_loss').mean()['eval_loss']

            data.append([win_len, cell_num, attn_lyrs, row_id_min_eval_loss + 1, train_loss, eval_loss, smallest_10_df_avg])
            # data.append([f'wl{win_len} {cell_num}cell {"Bi" if bi else "Uni"} {f"{attn_lyrs}lyr" if attn_lyrs>0 else "No"}-Attn', row_id_min_eval_loss + 1, train_loss, eval_loss, smallest_10_df_avg])

            x = int(win_len / 30) - 1 + (int(cell_num / 16) - 1) * 3 * dimw + attn_lyrs * dimw

            smallest_10_df_avg = history_df.nsmallest(5, 'eval_loss').mean()['eval_loss']

            ax.bar(x, eval_loss, dimw, color='r')
            ax.bar(x, smallest_10_df_avg, dimw, color='orange')
            ax.bar(x, train_loss, dimw, color='b')

    # data = sorted(data, key=lambda x: (x[1], x[0], x[2]))
    # data = np.array(data)
    #
    #
    # x = np.arange(3)
    #
    # ax.bar(x + 2 * dimw, data[:, 3], dimw, color='#FFA48E', label='Avg. Eval. Loss')
    #
    # ax.bar(x, data[:, 1], dimw, color='#1F77B4', label='Least Train Loss')
    # # ax.bar(x + dimw, data[:, 2], dimw, color='#C5ACAB', label='Least Val. Loss')
    # ax.bar(x + dimw, data[:, 2], dimw, color='#FF7F0E', label='Least Eval. Loss')
    #
    # ax.set_xticks(x + dimw)
    # ax.set_ylabel('Loss', fontsize=20)
    # ax.set_title('Training, Validation, and Evaluation Loss', fontsize=20)
    # # ax.set_xticklabels(labels, fontsize=20)
    # # ax.set_xscale('log')
    # ax.set_xticklabels(labels)
    # ax.legend(loc='upper left', fontsize=12)
    #
    # rects = ax.patches
    # epo_nums = data[:, 0].astype(int)
    # agv_eval_loss = data[:, -1]
    # length = len(labels)
    # for i in range(length):
    #     rect = rects[i]
    #     rect_1 = rects[length + i]
    #     rect_2 = rects[2 * length + i]
    #     # rect_3 = rects[3 * length + i]
    #     # rect_eval = rects[3*length+i]
    #     ax.text(
    #         rect_2.get_x() + rect.get_width() / 2, max(rect_1.get_height(), rect_2.get_height(), rect.get_height()),
    #         f'{epo_nums[i]} E', ha="center", va="bottom", fontsize=14
    #     )
    # ax.hlines(y=agv_eval_loss[i], xmin=rect_val.get_x(), xmax=rect_val.get_x()+rect.get_width(), colors='r')

    plt.show()

# exit(0)
# data_dir = os.path.join(ROOT_DIR, "..", "classic_rnn", 'results', feature_set)
# data_dir = os.path.join(ROOT_DIR, "..", "classic_rnn", 'results', 'simu2', feature_set)
# history_csv_names = [f for f in listdir(data_dir) if
#                          isfile(os.path.join(data_dir, f)) and f[-3:] == "csv" and f[:7] == "history"]

if print_type == 'diff cell_num & wl':
    data_dir = os.path.join(ROOT_DIR, "..", "classic_rnn", 'results', feature_set)
    history_csv_names = [f for f in listdir(data_dir) if
                         isfile(os.path.join(data_dir, f)) and f[-3:] == "csv" and f[:7] == "history"]
    data = []
    x = np.arange(4)
    dim = 3
    w = 0.75
    dimw = w / dim
    fig, ax = plt.subplots(figsize=(10, 6))
    for (i, history_csv_name) in enumerate(history_csv_names):
        m = re.search(r"wl(\d+)", history_csv_name)
        win_len = int(m.group(1))
        m = re.search(r"(\d+)cell", history_csv_name)
        cell_num = int(m.group(1))
        if cell_num in [16, 32, 48] and win_len in [30,60,90]:
            history_df = pd.read_csv(os.path.join(data_dir, history_csv_name), nrows=30)
            # history_df = pd.read_csv(os.path.join(data_dir, history_csv_name))
            row_id_min_eval_loss = history_df['eval_loss'].idxmin()
            train_loss = history_df.iloc[row_id_min_eval_loss]['train_loss']
            # val_loss = history_df.iloc[row_id_min_eval_loss]['val_loss']
            eval_loss = history_df.iloc[row_id_min_eval_loss]['eval_loss']
            data.append([win_len, cell_num, row_id_min_eval_loss+1, train_loss, eval_loss])

            x = int(win_len / 30) - 1 + (int(cell_num / 16) - 1) * dimw

            if cell_num == 16:
                train_color = '#1F77B4'
                eval_color = '#FFA48E'
                # eval_color = '#FF7F0E'
                avg_eval_color = '#E32D38'
            elif cell_num == 32:
                train_color = '#009AC8'
                eval_color = '#FF8E58'
                avg_eval_color = '#FF5D78'
            else:
                train_color = '#00B9C2'
                eval_color = '#FF7F0E'
                # eval_color = '#FFA48E'
                avg_eval_color = '#FF8DB6'

            # last_10_df_avg = history_df.iloc[-5:].mean()
            # avg_eval_loss = last_10_df_avg['eval_loss']

            smallest_10_df_avg = history_df.nsmallest(5, 'eval_loss').mean()['eval_loss']

            ax.bar(x, smallest_10_df_avg, dimw, color=avg_eval_color)
            ax.bar(x, eval_loss, dimw, color=eval_color)
            ax.bar(x, train_loss, dimw, color=train_color)

            # ax.hlines(y=last_10_df_avg['eval_loss'], xmin=x-dimw/2, xmax=x+dimw/2, colors='r')
            # ax.hlines(y=last_10_df_avg['train_loss'], xmin=x-dimw/2, xmax=x+dimw/2, colors='b')

            ax.text(x, eval_loss, f'{row_id_min_eval_loss+1} E', ha="center", va="bottom")
            ax.text(x, 0, f'{cell_num} D', ha="center", va="bottom", color='w')

            # if cell_num == 64 and win_len == 50:
            #     ax.axhline(y=eval_loss, color='#2ca02c', linestyle='-.')

    data = np.array(sorted(data, key=lambda x: (x[0], x[1])))
    print(data)
    labels = ['30', '60', '90']

    ax.set_xticks(np.arange(3) + dimw)
    ax.set_xticklabels(labels, fontsize=20)

    ax.set_ylabel('Loss', fontsize=20)
    ax.set_title('Training, Validation, and Evaluation Loss', fontsize=20)

    legend_elements = [
        # Patch(facecolor='#E32D38', edgecolor='#E32D38',
        #                      label='Avg. Eval. Loss'),
                       Patch(facecolor='#ff7f0e', edgecolor='#ff7f0e',
                             label='Eval. Loss'),
                       Patch(facecolor='#1f77b4', edgecolor='#1f77b4',
                             label='Train Loss')]
    ax.legend(handles=legend_elements, loc='right', fontsize=12)

    plt.show()

if print_type == 'cell_num on wl 50':
    data_dir = os.path.join(ROOT_DIR, "..", "classic_rnn", 'results', feature_set)
    history_csv_names = [f for f in listdir(data_dir) if
                         isfile(os.path.join(data_dir, f)) and f[-3:] == "csv" and f[:7] == "history"]
    wl = 60
    data = []
    for (i, history_csv_name) in enumerate(history_csv_names):
        # print(history_csv_name)
        m = re.search(r"wl(\d+)", history_csv_name)
        win_len = m.group(1)
        if int(win_len) == wl:
            m = re.search(r"(\d+)cell", history_csv_name)
            cell_num = m.group(1)
            history_df = pd.read_csv(os.path.join(data_dir, history_csv_name))
            # history_df = pd.read_csv(os.path.join(data_dir, history_csv_name), nrows=15)
            row_id_min_eval_loss = history_df['eval_loss'].idxmin()
            # print(row_id_min_eval_loss)
            train_loss = history_df.iloc[row_id_min_eval_loss]['train_loss']
            # val_loss = history_df.iloc[row_id_min_eval_loss]['val_loss']
            eval_loss = history_df.iloc[row_id_min_eval_loss]['eval_loss']
            # last_10_df_avg = history_df.iloc[-10:].mean()
            smallest_10_df_avg = history_df.nsmallest(10, 'eval_loss').mean()['eval_loss']
            # print(smallest_10_df_avg)
            # exit(0)

            data.append([int(cell_num), row_id_min_eval_loss+1, train_loss, eval_loss, smallest_10_df_avg, eval_loss-train_loss])
    data = sorted(data, key=lambda x: x[-2])
    data = np.array(data)
    # print(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = data[:, 0].astype(int)
    x = np.arange(labels.shape[0])

    dim = 2
    w = 0.75
    dimw = w / dim


    ax.bar(x, data[:, 2], dimw, color='#1F77B4', label='Train Loss')
    # ax.bar(x+dimw, data[:, 3], dimw, color='#C5ACAB', label='Least Val. Loss')
    ax.bar(x+dimw, data[:, 4], dimw, color='#FFA48E', label='Avg. Eval. Loss')
    ax.bar(x+dimw, data[:, 3], dimw, color='#FF7F0E', label='Least Eval. Loss')

    ax.set_xticks(x + dimw)
    # ax.set_xticklabels(data[:, 0])
    ax.set_ylabel('Loss', fontsize=20)
    ax.set_title('Training and Evaluation Loss', fontsize=20)
    ax.set_xticklabels(labels, fontsize=20)
    ax.legend(loc='right', fontsize=12)

    rects = ax.patches
    epo_nums = data[:, 1].astype(int)
    # agv_eval_loss = data[:, -1]
    length = labels.shape[0]
    for i in range(length):
        rect = rects[length+i]
        rect_val = rects[2*length+i]
        # rect_eval = rects[3*length+i]
        ax.text(
            rect.get_x() + rect.get_width()/2, max(rect_val.get_height(), rect.get_height()),
            f'{epo_nums[i]} epochs', ha="center", va="bottom", fontsize=14
        )
        # ax.hlines(y=agv_eval_loss[i], xmin=rect_val.get_x(), xmax=rect_val.get_x()+rect.get_width(), colors='r')

    plt.show()