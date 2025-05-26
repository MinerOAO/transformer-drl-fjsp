import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np

def gantt_chart_plt(schedules_batch, ope_num_of_job, ope_index, num_jobs, makespan, ins_name):
    batch = schedules_batch.cpu().numpy()
    # From matplot mcolor example
    # color_list = sorted(
    #         mcolors.CSS4_COLORS, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    color_list = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
        '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
        '#D37295', '#A5A5A5', '#35B779', '#FFA600', '#CD9BCC',
        '#6B9E75', '#C8C8C8', '#FF7F0E', '#1F77B4', '#DB5F57',
        '#66CCFF'
    ]
    fig, ax = plt.subplots()
    legend_elements = []

    for i in range(num_jobs):
        job_color = color_list[i % len(color_list)]
        legend_elements.append(Patch(facecolor=job_color, label=f'Job {i+1}'))
        for j in range(int(ope_num_of_job[i])): 
            data = batch[ope_index[i]+j]
            ax.barh(data[1], data[3] - data[2], 1, left = data[2], color=job_color) #y, width, height, start_x

    ax.legend(handles=legend_elements, title='Jobs', loc='center left', bbox_to_anchor=(1, 0.5))
    # 设置图表标题和标签
    ax.set_title(f'Gantt Chart ({ins_name} Makespan: {makespan})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    plt.show()