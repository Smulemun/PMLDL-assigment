import matplotlib.pyplot as plt

METRICS = ['SIM', 'ACC', 'FLNC', 'J']
METRIC2NAME = {'SIM': 'Semantic Similarity', 'ACC': 'Style Accuracy', 'FLNC': 'Fluency', 'J': 'J Score'}

def plot_metrics(data):
    '''Function to plot metrics'''
    # for each metric plotting a bar chart
    for i, metric in enumerate(METRICS):
        bars = []
        for key in data.keys():
            bars.append(data[key][metric])
        plt.figure(figsize=(10, 8))
        plt.bar(data.keys(), bars)
        plt.title(METRIC2NAME[metric])
        plt.savefig(f'../reports/figures/{METRIC2NAME[metric].lower().replace(" ", "_")}.png')
        plt.show()