#!/usr/bin/env python

"""
An extension to rootpy for convenience and speed in plotting.

Matplotlib is the backend, because it is more powerful than ROOT.

But, for convenience and familiarity, ROOT classes are used to
hold and analyze data, through rootpy.

This module provides quick setup functions for matplotlib plots
and an interface for good color choices.

"""

import os
import sys
import glob
import json
import types
import shlex
import argparse
import matplotlib
matplotlib.use('PDF')

import retrieve

# Right now this uses an adaptable system:
#   Full latex support
#   Full vector rendering

# I would like to upgrade to pgf based output if they work out bugs


from seaborn.apionly import color_palette

custom_preamble = {
    'font.size'          : 20, # default is too small
    'figure.autolayout'  : True,
    'text.usetex'        : True,
    'axes.edgecolor'     : '#737373',
    'grid.color'         : '#B3B3B3',
    'grid.linestyle'     : '--',
    'grid.linewidth'     : 2,
    'ytick.major.size'   : 8,
    'ytick.minor.size'   : 4,
    'xtick.major.size'   : 8,
    'xtick.minor.size'   : 4,
#    'lines.linewidth'    : 2.0,
#    'lines.markersize'   : 16.0,
    'legend.numpoints'   : 1
    }

matplotlib.rcParams.update(custom_preamble)

"""
fontdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"fonts/")
pgf_update = {
    'font.family'       : 'serif', # actually uses latex main font from below
    'pgf.texsystem'     : 'xelatex',
    'pgf.rcfonts'       : False,   # don't setup fonts from rc parameters
    'pgf.preamble'      : [
        r'\usepackage{mathspec}',
        '\setallmainfonts(Digits,Latin,Greek)[Path = {0}, BoldFont={{HelveticaBold}}, ItalicFont={{HelveticaOblique}}, BoldItalicFont={{HelveticaBoldOblique}}]{{Helvetica}}'.format(fontdir)
    ]
}
"""

import matplotlib.pyplot as plt
import rootpy.plotting.root2matplotlib as rplt

import matplotlib.ticker as ticker
import matplotlib.offsetbox as offsetbox

import rootpy.ROOT as ROOT
from rootpy.plotting.style import set_style
from rootpy.plotting.utils import get_limits, draw
from matplotlib.colors import LogNorm
from rootpy.plotting.style import get_style, set_style
from rootpy.io import root_open
from rootpy.tree import Tree, TreeChain

from itertools import izip, chain


set_style('ATLAS',mpl=True)

# Update for fonts
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] =  [r'\usepackage{tgheros}',    # helvetica font
                                               r'\usepackage{sansmath}',   # math-font matching  helvetica
                                               r'\sansmath'                # actually tell tex to use it!
                                               r'\usepackage{siunitx}',    # micro symbols
                                               r'\sisetup{detect-all}',    # force siunitx to use the fonts
                                              ]
                
          
# ----------------------------------------
# Utility Functions and Variables
# ----------------------------------------

INTERNAL = r'$\textbf{\textit{ATLAS} Internal}$'
PRELIM = r'$\textbf{\textit{ATLAS} Preliminary}$'

def is_listy(x):
    return any(isinstance(x, t) for t in [types.TupleType,types.ListType])

def make_iterable(x):
    if not is_listy(x):
        return [x]
    return x

"""
def set_font(font="helvetica"):
    fonts = {"helvetica": '\setallmainfonts(Digits,Latin,Greek)[Path = {0}, BoldFont={{HelveticaBold}}, ItalicFont={{HelveticaOblique}}, BoldItalicFont={{HelveticaBoldOblique}}]{{Helvetica}}'.format(fontdir),
             "lato": '\setallmainfonts(Digits,Latin,Greek)[Path = {0}, BoldFont={{Lato-Bold}}, ItalicFont={{Lato-LightItalic}}, BoldItalicFont={{Lato-BoldItalic}}]{{Lato-Light}}'.format(fontdir)}
    if font.lower() not in fonts:
        print "Attempting to use unsupported font", font, "; not enabled."
        return
    matplotlib.rcParams.update({'pgf.preamble':[r'\usepackage{mathspec}', fonts[font.lower()]]}) 
""" 

# ----------------------------------------
# Utility Classes
# ----------------------------------------

class PlotError(Exception):
    pass
    
# ----------------------------------------
# Plot Setup
# ----------------------------------------

def normalize(hists, norm='', bin_norm='', index_norm=''):
    for hist in hists:
        if norm: 
            hist.Scale(norm/hist.Integral())
        if index_norm: 
            hist.Scale(hist[index_norm].Integral()//hist.Integral())
        if bin_norm:
            for i in hist.bins_range():
                hist.SetBinContent(i, hist.GetBinContent(i)/hist.GetBinWidth(i))
                hist.SetBinError(i, hist.GetBinError(i)/hist.GetBinWidth(i)) 
                


def setup_axes(*axes, **kwargs):
    """ Setup an axis with standard options. 

    Just a shortcut for calling a bunch of axes member functions.
    
    Keyword Arguments:
    title -- Text to be placed on top center of axes.
    logy -- If true, set yscale to log.
    logx -- If true, set xscale to log.
    xmin -- Set minimum of x-axis.
    xmax -- Set maximum of x-axis.
    ymin -- Set minimum of y-axis.
    ymax -- Set maximum of y-axis.
    xlabel -- Label for x-axis.
    ylabel -- Label for y-axis.
    xticks -- List of tick positions
    xticklabels -- List of tick labels
    xtickrot -- Rotation for tick labels
    """

    if 'hist' in kwargs:
        hist = kwargs['hist']
        if not 'ylabel' in kwargs: kwargs['ylabel'] = hist.GetYaxis().GetTitle()
        if not 'xlabel' in kwargs: kwargs['xlabel'] = hist.GetXaxis().GetTitle() 
    if 'limits' in kwargs:
        if not 'xmin' in kwargs: kwargs['xmin'] = kwargs['limits'][0]
        if not 'xmax' in kwargs: kwargs['xmax'] = kwargs['limits'][1]
        if not 'ymin' in kwargs: kwargs['ymin'] = kwargs['limits'][2]
        if not 'ymax' in kwargs: kwargs['ymax'] = kwargs['limits'][3]        

    for ax in axes:
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        if 'logy' in kwargs and kwargs['logy']:
            ax.set_yscale('log')
        if 'logx' in kwargs and kwargs['logx']:
            ax.set_xscale('log')
        if 'ymin' in kwargs:
            ax.set_ylim(bottom=kwargs['ymin'])
        if 'ymax' in kwargs:
            ax.set_ylim(top=kwargs['ymax'])
        if 'xmin' in kwargs:
            ax.set_xlim(left=kwargs['xmin'])
        if 'xmax' in kwargs:
            ax.set_xlim(right=kwargs['xmax'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], y=1, ha='right')
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'], x=1, ha='right')        
        if 'xticks' in kwargs:
            ax.set_xticks(kwargs['xticks'])
        if 'xticklabels' in kwargs:
            if 'xtickrot' in kwargs:
                ax.set_xticklabels(kwargs['xticklabels'], rotation=kwargs['xtickrot'])
            else:
                ax.set_xticklabels(kwargs['xticklabels'])
        if 'yticks' in kwargs:
            ax.set_yticks(kwargs['yticks'])
        if 'yticklabels' in kwargs:
            ax.set_yticklabels(kwargs['yticklabels'])
        if ax.get_xscale() != 'log':
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if ax.get_yscale() != 'log':
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_label_coords(-0.15,1)

        
def clean_ratio(axes):
    for ax in axes[:-1]:
        for tick in ax.get_xticklabels():
            tick.set_visible(False)
        ax.set_xlabel("")
    axes[-1].set_ylabel(axes[-1].get_ylabel(), y=.5, ha='center')
    axes[-1].yaxis.grid(True) # Show lines on tick marks
    xmin, xmax, ymin, ymax = axes[-1].axis()
    # Custom tick locations for some common ratio plot limits
    if ymin == 0.0 and ymax == 2.0:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.5,1.0,1.5]))
    elif ymin == 0.0 and ymax == 1.25:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.5,1.0]))
    elif ymin == 0.75 and ymax == 1.5:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.8,1.0,1.2,1.4]))
    elif ymin == 0.9 and ymax == 1.6:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([1.0,1.5]))        
    elif ymin == 0.8 and ymax == 1.2:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.9,1.0,1.1]))
    elif ymin == 0.9 and ymax == 1.2:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([1.0,1.1]))
    elif ymin == 0.9 and ymax == 1.1:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([.95,1.0,1.05]))
    elif ymin == 0.85 and ymax == 1.15:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.9,1.0,1.1]))
    elif ymin == 0.75 and ymax == 1.25:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.9,1.0,1.1]))
    elif ymin == 0.8 and ymax == 2.0:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([1.0,1.5]))
    elif ymin == 0.5 and ymax == 1.5:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([0.75,1.0,1.25]))
    elif ymin == 2.0 and ymax == 6.0:
        axes[-1].yaxis.set_major_locator(ticker.FixedLocator([3.0,4.0,5.0]))

        
def legend(ax, loc='best', text=None, ncol=1):
    legend = ax.legend(frameon=False, loc=loc, prop={'size':18}, labelspacing=0.25, ncol=ncol)
    # Add extra text below legend.
    if text is not None:
        for sub in text.split(";")[::-1]:
            txt=offsetbox.TextArea(sub, {'size':18}) 
            box = legend._legend_box 
            box.get_children().insert(0,txt) 
            box.set_figure(box.figure) 
    return legend

    
# ----------------------------------------
# Drawing Functions
# ----------------------------------------

def errorbar(ax, hists, **kwargs):
    """
    Plot histograms or graphs as points with errors bars.

    kwargs are passed to errorbar
    """
    # Override some defaults
    defaults = {"snap":False,
                "emptybins":False,
                "capsize":0}
    defaults.update(kwargs)
    if "logy" in defaults:
        defaults.pop("logy")
    rplt.errorbar(hists, axes=ax, **defaults)

def band(ax, bottoms, tops, **kwargs):
    """
    Fill the region between bottom and top, for each pair.
    #todo Add an option that makes a band between histogram upper and lower errors

    kwargs are passed to fill_between
    """
    # Override some defaults
    defaults = {"linewidth":0.0}
    defaults.update(kwargs)
    for b,t in zip(make_iterable(bottoms), make_iterable(tops)):
        rplt.fill_between(b, t, axes=ax, **defaults)
    
def _step(ax, h, **kwargs):
    # Make horizontal line segments
    x, y = list(h.xedges()), list(h.y())
    x_pairs = [x[i:i+2] for i in xrange(len(x)-1)]
    y_pairs = [[y[i],y[i]] for i in xrange(len(y))]
    segments = [[x,y] for x,y in izip(x_pairs, y_pairs)]
    segments = sum(segments, [])

    # Borrowed style setting from rootpy
    rplt._set_defaults(h, kwargs, ['common', 'line']) 
    if kwargs.get('color') is None:
        kwargs['color'] = h.GetLineColor('mpl')

    # Only have a label for the first one
    ax.plot(*segments[:2], **kwargs)
    kwargs['label'] = None
    ax.plot(*segments[2:], **kwargs)

def step(ax, hists, **kwargs):
    """
    Plot the histogram as horizontal lines at the bin content

    kwargs are passed to Axes.plot
    """
    if is_listy(hists):
        for h in hists:
            _step(ax, h, **kwargs)
    else:
        _step(ax, hists, **kwargs)

def hist(ax, hists, **kwargs):
    defaults = {'fill':None,
                'capsize':0,
                'yerr':None}
    defaults.update(kwargs)
    rplt.hist(hists, axes=ax, **defaults)
        
def stack(ax, hists, **kwargs):
    defaults = {'fill':'solid',
                'capsize':0,
                'yerr':None}
    defaults.update(kwargs)
    rplt.hist(hists, axes=ax, **defaults)

def hist2d(ax, hists, **kwargs):
    if "logz" in kwargs and kwargs.pop("logz"):
        kwargs["norm"] = LogNorm()
    defaults = {"colorbar":True, "cmap":"Oranges"}
    defaults.update(kwargs)
    rplt.hist2d(hists, axes=ax, **defaults)

_plot_functions = {"errorbar":errorbar,
                   "hist":hist,
                   "stack":stack,
                   "band":band,
                   "step":step}

def draw1d(ax, hists):
    stacked = [h for h in hists if h.drawstyle == 'stack']
    unstacked = [h for h in hists if h.drawstyle != 'stack']
    if stacked: stack(ax, stacked)
    for h in unstacked:
        if h.drawstyle == '': h.drawstyle = 'errorbar'
        if h.drawstyle not in _plot_functions:
            raise PlotError("Do not recognize drawstyle: " + h.drawstyle)
        _plot_functions[h.drawstyle](ax, h)


def plot_single(draw, name, hists, leg=True, legend_loc='best', legend_text=INTERNAL, logx=False, logy=False, ymin=None, ymax=None, norm=False, bin_norm=False, **kwargs):
    """
    Performs basic 2d canvas setup and draws hists with draw

    kwargs are passed to setup_axes
    """
    fig, (ax,) = axes_single()
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if norm:
        normalize(hists, norm=norm)
    if bin_norm:
        normalize(hists, bin_norm=bin_norm)
        
    if ymin is None or ymin == "":
        l,h = min(chain(*[list(h.y()) for h in hists])), max(chain(*[list(h.y()) for h in hists]))
        if logy: ymin = 0.8*l
        else: ymin = l - .1*(h-l)
    ax.set_ylim(bottom=ymin)
        
    if ymax is None or ymax == "":
        l,h = min(chain(*[list(h.y()) for h in hists])), max(chain(*[list(h.y()) for h in hists]))        
        if logy: ymax = h/0.8
        else: ymax = h + .1*(h-l)
    ax.set_ylim(top=ymax)
    
    draw(ax, hists)
    
    setup_axes(ax, hist=hists[0], ymin=ymin, ymax=ymax, **kwargs)
    if leg and any(h.title for h in hists):
        legend(ax, loc=legend_loc, text=legend_text, ncol=1 if len(hists) < 5 else 2)

    if any(x in name for x in ["png","jpg"]):
        fig.savefig(name, dpi=500)
    else:
        fig.savefig(name)

def plot2d(name, hists, cmap='OrRd', logz=False, **kwargs):
    kwargs['leg'] = False
    kwargs['title'] = hists[0].title
    plot(lambda ax, hists: hist2d(ax, hists[0], cmap=cmap, logz=logz), name, hists, **kwargs)
    


# ----------------------------------------
# Plot Layouts
# ----------------------------------------

def axes_single():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, [ax]

def axes_ratio(figw=8.0,figh=6.0):
    fig = plt.figure(figsize=(figw, figh))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1], hspace=0.0, wspace=0.0)
    ax = plt.subplot(gs[0])
    return fig, [ax, plt.subplot(gs[1], sharex=ax)]

def axes_double_ratio(figw=8.0, figh=8.0):
    fig = plt.figure(figsize=(figw, figh))
    gs = gridspec.GridSpec(3,1,height_ratios=[2,2,1], hspace=0.0, wspace=0.0)
    ax = plt.subplot(gs[0])
    return fig, [ax, plt.subplot(gs[1], sharex=ax), plt.subplot(gs[2], sharex=ax)]


# ----------------------------------------
# ROOT Helpers
# ----------------------------------------

def fit(hists, functions):
    graphs, pars, errors = [], [], []
    if isinstance(functions, types.StringTypes):
        functions = [functions for h in hists]
    for hist,function in zip(hists,functions):
        hist.Fit(function, "RB+")
        func = hist.GetFunction(function)
        graphs.append(ROOT.TH1D(func.GetHistogram()))
        params = func.GetParameters()
        param_errors = func.GetParErrors()
        pars.append(params)
        errors.append(param_errors)
    return graphs, pars, errors

def efficiency(h1, h2):
    eff = Graph()
    eff.Divide(h1,h2,"cl=0.683 b(1,1) mode")
    return eff

def saveable(selection):
    replacers = [(" ",""),
                 ("&&","_"),
                 (">=","ge"),
                 ("<=","le"),
                 (">","g"),
                 ("<","l"),
                 ("==","e")]
    for s,r in replacers:
        selection = selection.replace(s,r)
    return selection
    
# ----------------------------------------
# Command Line Plotting
# ----------------------------------------

def style(sname, hists):
    if sname == "bands":
        for h in hists:
            for i in h.bins_range():
                h.SetBinError(i,0.0)
            h.drawstyle = 'step'
        for h in hists[1:]:
            hists.append(h.Clone())
            hists[-1].title = ""
            for i in h.bins_range():
                h.SetBinContent(i,hists[0].GetBinContent(i) - h.GetBinContent(i))
                hists[-1].SetBinContent(i,hists[0].GetBinContent(i) + hists[-1].GetBinContent(i))            
    return hists


def main(args):
    parser = argparse.ArgumentParser('Create a plot from ntuples, using variable and sample lookup information from spreadsheets.')
    parser.add_argument('output', nargs="?", help='Name of the plot to save.')
    parser.add_argument('--samples', nargs='+', help='The names of samples to add to the plot. Each sample gets a separate graphic.')
    parser.add_argument('--variables', nargs='+', help='The names of variables to add to the plot. Each variable gets a separate graphic.')
    parser.add_argument('--selection', help='Additional global selection to apply for all loaded data.')
    parser.add_argument('--style', help='Apply special style type to plot.')
    parser.add_argument('--ratio', action='store_true', help='Additional draw ratio plots.')
    parser.add_argument('--batch', help='Specify a file which contains a set of arguments to run on each line.')

    args = parser.parse_args(args)

    if not args.output and not args.batch:
        raise PlotError("Must provide an output name for the saved plot or a batch file to run on.")

    if args.batch:
        with open(args.batch) as f:
            lines = f.readlines()
        # Maybe ship this out in parallel?
        for next_args in lines:
            try:
                main(shlex.split(next_args))
            except BaseException as e:
                print "Plotting failed with ", e
                print "  and arguments", next_args
        return

    hists = retrieve.retrieve_all(args.samples, args.variables, args.selection)
    with open(retrieve.variables_json) as f:
        options = json.load(f)[args.variables[0]]

    # Come back to this!!
    with open(retrieve.samples_json) as f:
        soptions = json.load(f)[args.samples[0]]

    if 'cmap' in soptions and soptions['cmap']:
        options['cmap'] = soptions['cmap']
    if 'title' in soptions and soptions["title"]:
        if options["title"]:
            options["title"] = ", ".join(soptions["title"], options["title"])
        else:
            options["title"] = soptions["title"]

    if args.style:
        hists = style(args.style, hists)
        
    # ----------------------------------------
    # Select a Plot Setup and Draw Setup
    # ----------------------------------------

    # Might support more options in the future
    if args.ratio and args.double:
        plot_function = plot_double_ratio
    elif args.ratio:
        plot_function = plot_single_ratio
    else:
        plot_function = plot_single

    if '2' in hists[0].__class__.__name__:
        draw_function = draw2d
    else:
        draw_function = draw1d

    # Create figure
    plot_function(draw_function, args.output, hists, **options)
    

if __name__ == "__main__":
    main(sys.argv[1:])
