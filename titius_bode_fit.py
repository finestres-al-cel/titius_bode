"""
    titius_bode_fit.py

    Prepare plots to check the validity of the Titius-Bode law
"""
# load modules
import readline
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import string

from math import log10

# this is so that input can autocomplete using the tab
def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)


# ask user which file to use
file = input("nom de l'arxiu ('end' per acabar):")

while (file != "end"):
    # get name of the file, without extension
    result_file = file[0:str.find(file,".")]

    # read data
    try:
        read_data = np.genfromtxt(file, names=True)
        print(read_data)
        print(read_data["dist"][1:len(read_data["dist"])])
    except OSError:
        print(f"file not found: {file}")
        file = input("nom de l'arxiu ('end' per acabar):")
        continue

    # prepare data to fit
    # fit log_(dist-dist_0) = log_p + order * log_q
    data = np.array([np.log10(dist-read_data["dist"][0]) for dist in read_data["dist"][1:len(read_data["dist"])]])
    order = np.array(read_data["ordre"][1:len(read_data["ordre"])])
    A = np.array([order, np.ones(order.size)])
    # obtaining the parameters
    w = np.linalg.lstsq(A.T, data)[0]

    # define a function to plot the resulting fit
    def predicted(x, w):
        return w[1]+x*w[0]

    # create an array with the predicitions and compute the residuals
    prediction = np.array([predicted(x, w) for x in read_data["ordre"][1:len(read_data["ordre"])]])
    residuals = [d-p for (p, d) in zip(prediction, data)]

    # compute r2
    mean_value = np.mean(data)
    residual_sum_of_squares = 0.0
    variance = 0.0
    for (p,d) in zip(prediction, data):
        residual_sum_of_squares += (p-mean_value)*(p-mean_value)
        variance += (d-mean_value)*(d-mean_value)
    r2 = residual_sum_of_squares/variance

    # plot figures
    gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    gs.update(left=0.2, bottom=0.2, hspace=0.05)
    fontsize = 32
    labelsize = 18
    figsize = (9, 7)

    fig = (plt.figure(figsize=figsize))
    # first subplot plots the data and prediction
    ax = fig.add_subplot(gs[0])
    title = ax.set_title(result_file, fontsize=fontsize)
    title.set_position([.5, 1.05])
    ax.set_ylabel(r"$\log\left(\frac{a-a_{-{\rm \infty}}}{\rm 1UA}\right)$", fontsize=fontsize)
    ax.set_xlim(-0.5,order[-1]*1.1)
    if data[0] < 0.0 and data[-1] < 0.0:
        ax.set_ylim(1.1*data[0], 0.9*data[-1])
    elif data[0] < 0.0:
        ax.set_ylim(1.1*data[0], 1.1*data[-1])
    elif data[-1] < 0.0:
        ax.set_ylim(0.9*data[0], 0.9*data[-1])
    else:
        ax.set_ylim(0.9*data[0], 1.1*data[-1])
    ax.plot(order, data,'r^', label="data")
    ax.plot(order, prediction, 'b-', label="model")
    ax.text(0.1,0.8, r"$r^{{2}} = {:.4f}$".format(r2), color='k', transform=ax.transAxes, fontsize=fontsize)
    ax.legend(loc=4,numpoints=1,prop={'size':labelsize},frameon=False)
    ax.tick_params(axis="y", labelsize=labelsize, pad=10)
    ax.axes.xaxis.set_ticklabels([])
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    # second subplot plots the residuals
    ax2 = fig.add_subplot(gs[1])
    ax2.set_ylabel(r"$\rm res$", fontsize=fontsize)
    ax2.set_xlabel(r"$\rm n$", fontsize=fontsize)
    ax2.set_xlim(-0.5,order[-1]*1.1)
    ax2.set_ylim(np.amin(residuals)-0.2, np.amax(residuals)+0.2)
    ax2.plot(order, residuals, 'r^')
    ax2.plot(ax2.get_xlim(), [0.0]*2, 'b--')
    ax2.tick_params(axis="both", labelsize=labelsize, pad=10)
    ax2.locator_params(axis='y', nbins=7)
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    yticks2 = ax2.yaxis.get_major_ticks()
    yticks2[0].label1.set_visible(False)
    yticks2[-1].label1.set_visible(False)
    # save figure using the provided name and with extension .png
    fig.savefig("{}.png".format(result_file))

    # save results using the provided name and with extension .res
    save = open(result_file+".res", "w")
    save.write("# Fit parameters (log(a-a_{0}) = log(p) + order * log(q) )\n")
    save.write("log(p) = " + str(w[1]) + "\n")
    save.write("log(q) = " + str(w[0]) + "\n")
    save.write("p = " + str(10**w[1]) + "\n")
    save.write("q = " + str(10**w[0]) + "\n")
    save.write("r^2 = " + str(r2) + " \n")
    save.write("\n")
    save.write("# Data (distances in AU, log in decimal base)\n")
    save.write("# order dist predicted_dist\n")
    save.write("-inf " + format(read_data["dist"][0], ".4f") + " " + format(read_data["dist"][0], ".5f") + "\n")
    for (d,o,p) in zip(data, order, prediction):
        save.write(str(o) + " " + format(10**d, ".4f")+ " " + format(10**p, ".4f")  + "\n")

    save.close()


    # ask for a new file
    file = input("nom de l'arxiu ('end' per acabar):")
