import matplotlib.pyplot as plt

def piano_roll(df):
    colormap = 'brcy'
    fig = plt.figure(figsize=(20, 8))
    
    for index, row in df.iterrows():
        plt.plot([row.start, row.start+row.duration], [row.note, row.note], color=colormap[int(row.label)])
        plt.scatter((2*row.start+row.duration)/2, row.note, s=20, c=colormap[int(row.label)])
    plt.show()