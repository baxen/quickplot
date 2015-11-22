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
from helpers import not_empty
from math import sqrt

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
import matplotlib.gridspec as gridspec
import rootpy.plotting.root2matplotlib as rplt

import matplotlib.ticker as ticker
import matplotlib.offsetbox as offsetbox

import rootpy.ROOT as ROOT
from rootpy.plotting.style import set_style
from rootpy.plotting.utils import get_limits, draw
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.colors import LogNorm
from rootpy.plotting.style import get_style, set_style
from rootpy.io import root_open
from rootpy.tree import Tree, TreeChain
from rootpy.plotting import Hist

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

canvas_json = 'canvas.json'
INTERNAL = r'$\textbf{\textit{ATLAS} Internal}$'
PRELIM = r'$\textbf{\textit{ATLAS} Preliminary}$'

def is_listy(x):
    return any(isinstance(x, t) for t in [types.TupleType,types.ListType])

def make_iterable(x):
    if not is_listy(x):
        return [x]
    return x

def histify(plottable):
    if 'Hist' in plottable.__class__.__name__:
        return plottable
    elif 'Graph' in plottable.__class__.__name__:
        bins = [(x-err[0],x+err[1]) for x,err in izip(plottable.x(), plottable.xerr())]
        bins = [x[0] for x in bins] + [bins[-1][1]]
        hist = Hist(bins)
        for i,y,yerr in izip(hist.bins_range(), plottable.y(), plottable.yerr()):
            hist.SetBinContent(i, y)
            hist.SetBinError(i, 0.5*(yerr[0]+yerr[1]))
        # Transfer all the graphic properties
        hist.decorate(plottable)
        hist.title = plottable.title
        return hist
    else:
        raise TypeError("Don't know how to create a histogram from " + plottable.__class__.__name__)

def sqrt_hist(hist):
    hist = hist.Clone()
    for i in hist.bins_range():
        rel_unc = hist.GetBinError(i)/hist.GetBinContent(i)
        hist.SetBinContent(i,sqrt(hist.GetBinContent(i)))
        hist.SetBinError(i, hist.GetBinContent(i) * rel_unc * .05)
    return hist
        
        

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
# Plot Layouts
# ----------------------------------------

def build_axes(ratio=False, double=False, **kwargs):
    """
    Builds a figure and axes with either one or two major axes
    and an optional ratio axes.

    kwargs are ignored
    """
    # Two axes and ratio
    if ratio and double:
        fig = plt.figure(figsize=(8.0, 8.0))
        gs = gridspec.GridSpec(3,1,height_ratios=[2,2,1], hspace=0.0, wspace=0.0)
        ax = plt.subplot(gs[0])
        axes = [ax, plt.subplot(gs[1], sharex=ax), plt.subplot(gs[2], sharex=ax)]

    # Single axes and ratio
    elif ratio:
        fig = plt.figure(figsize=(8.0, 6.0))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1], hspace=0.0, wspace=0.0)
        ax = plt.subplot(gs[0])
        axes = [ax, plt.subplot(gs[1], sharex=ax)]

    # Single axis
    else:
        fig = plt.figure()
        axes = [fig.add_subplot(111)]

    for ax in axes:
        ax.ratio = False
        ax.primary = False
    axes[0].primary = True
    if ratio: axes[-1].ratio = True

    return fig, axes


def process_hists(hists, ratio=False, double=False, rindex=0, **kwargs):
    """
    Process and group histograms to match the setup created by build_axes.

    kwargs are ignored
    """
    
    # Processing of hists prior to ratio
    for j,hist in enumerate(hists):
        if not_empty(kwargs, 'xefficiency') or not_empty(kwargs,'xinefficiency'):
            index = int(kwargs['xefficiency']) if not_empty(kwargs, 'xefficiency') else int(kwargs['xinefficiency'])
            num = hist.Clone()
            for x in num.bins_range(overflow=True):
                if index < 0:
                    # Integral up to the value
                    y,yerr = hist.integral(xbin2=x, error=True, overflow=True)
                else:
                    # Integral starting at the value
                    y,yerr = hist.integral(xbin1=x, error=True, overflow=True) 
                num.SetBinContent(x,y)
                num.SetBinError(x,yerr)
            hists[j] = num
        if not_empty(kwargs, 'overflow') and kwargs['overflow']:
            # Add overflow to last bin for plotting
            hist[hist.bins_range()[-1]] += hist[hist.bins_range(overflow=True)[-1]]

    # If two axes, half go on each in order
    if double:
        hgroups = [hists[:len(hists)/2], hists[len(hists)/2:]]
    else:
        hgroups = [hists]
    
    # If ratio, create a ratio from each hist by dividing by the histogram at rindex for each group
    if ratio and not_empty(kwargs, "ratio_sb") and kwargs["ratio_sb"]:
        hgroups.append(sum(([histify(h)/sqrt_hist(histify(group[rindex])) for i,h in enumerate(group) if i != rindex] for group in hgroups), []))
    elif ratio:
        hgroups.append(sum(([histify(h)/histify(group[rindex]) for i,h in enumerate(group) if i != rindex] for group in hgroups), []))

    # Processing of hists after ratio
    for i, hgroup in enumerate(hgroups):
        if ratio and i == len(hgroups) - 1:
            continue
        for j,hist in enumerate(hgroup):
            if not_empty(kwargs, 'xefficiency') or not_empty(kwargs,'xinefficiency'):
                norm, nerr = hist.GetBinContent(abs(index)), hist.GetBinError(abs(index)) 
                den = hist.Clone()
                for x in den.bins_range():
                    den.SetBinContent(x, norm)
                    den.SetBinError(x, nerr)
                hgroups[i][j] = hist/den
            if not_empty(kwargs, 'xinefficiency'):
                hgroups[i][j] = hgroups[i][j]*-1 + 1
    return hgroups
    

# ----------------------------------------
# Plot Setup
# ----------------------------------------

def set_scales(axes, logx=False, logy=False, **kwargs):
    """
    Set log scales on all axes.

    Does not set logy on an axis marked ratio
    kwargs are ignored for convenience in passing
    """
    for ax in axes:
        if logx:
            ax.set_xscale('log')
        if logy and (not hasattr(ax,'ratio') or not ax.ratio):
            ax.set_yscale('log')


def normalize(hists, norm='', bin_norm='', index_norm='', overflow=False, **kwargs):
    for hist in hists:
        if norm: 
            hist.Scale(norm/hist.integral(overflow=overflow))
        if index_norm: 
            hist.Scale(hist[index_norm].Integral()//hist.integral(overflow=overflow))
        if bin_norm:
            for i in hist.bins_range():
                hist.SetBinContent(i, hist.GetBinContent(i)/hist.GetBinWidth(i))
                hist.SetBinError(i, hist.GetBinError(i)/hist.GetBinWidth(i)) 


def setup_axes(*axes, **kwargs):
    """ Setup an axis with standard options. 

    Just a shortcut for calling a bunch of axes member functions.
    
    Keyword Arguments:
    title -- Text to be placed on top center of axes.
    xmin -- Set minimum of x-axis.
    xmax -- Set maximum of x-axis.
    ymin -- Set minimum of y-axis.
    ymax -- Set maximum of y-axis.
    xlabel -- Label for x-axis.
    ylabel -- Label for y-axis.
    xticks -- String holding space-separated tick locations
    xticklabels -- String hold space-separated tick labels
    xtickrot -- Rotation for tick labels
    """

    # Handle some shortcut syntaxes
    if not_empty(kwargs, 'xticks') and not not_empty(kwargs, 'xticklabels'):
        kwargs['xticklabels'] = kwargs['xticks']

    if not_empty(kwargs, 'yticks') and not not_empty(kwargs, 'yticklabels'):
        kwargs['yticklabels'] = kwargs['yticks']

    if not_empty(kwargs, 'xticklabels') and kwargs['xticklabels'] == 'jes_cor':
        kwargs['xticklabels'] = ['30', '300\n' +r'$0.8 < |\eta| < 1.1$', '2500', '30', '300\n' +r'$1.1 < |\eta| < 1.4$', '2500']
    if not_empty(kwargs, 'yticklabels') and kwargs['yticklabels'] == 'jes_cor':
        kwargs['yticklabels'] = ['30', '300', '2500', '30', '300', '2500']

    if not_empty(kwargs, 'xticks'):
        try:
            kwargs['xticks'] = [float(x) for x in kwargs['xticks'].split()]
        except (AttributeError):
            pass

    if not_empty(kwargs, 'xticklabels'):
        try:
            kwargs['xticklabels'] = kwargs['xticklabels'].split()
        except (AttributeError, KeyError):
            pass

    if not_empty(kwargs, 'yticks'):
        try:
            kwargs['yticks'] = [float(x) for x in kwargs['yticks'].split()]
        except (AttributeError, KeyError):
            pass
    if not_empty(kwargs, 'yticklabels'):
        try:
            kwargs['yticklabels'] = kwargs['yticklabels'].split()
        except (AttributeError, KeyError):
            pass
                         
    # Setup each axis
    for ax in axes:
        if not_empty(kwargs, 'title') and ax.primary:
            # Only put title on the top set of axes
            ax.set_title(kwargs['title'])

        if ax.ratio:
            # Use ratio y-axis values
            if not_empty(kwargs, 'rmin'):
                ax.set_ylim(bottom=kwargs['rmin'])
            if not_empty(kwargs, 'rmax'):
                ax.set_ylim(top=kwargs['rmax'])
            if not_empty(kwargs, 'rlabel'):
                ax.set_ylabel(kwargs['rlabel'], y=1, ha='right')
        else:
            # Use normal y-axis values
            if not_empty(kwargs, 'ymin'):
                ax.set_ylim(bottom=kwargs['ymin'])
            if not_empty(kwargs, 'ymax'):
                ax.set_ylim(top=kwargs['ymax'])
            if not_empty(kwargs, 'ylabel'):
                ax.set_ylabel(kwargs['ylabel'], y=1, ha='right')
            if not_empty(kwargs, 'yticks'):
                ax.set_yticks(kwargs['yticks'])
            if not_empty(kwargs, 'yticklabels'):
                ax.set_yticklabels(kwargs['yticklabels'])

        
        if not_empty(kwargs, 'xmin'):
            ax.set_xlim(left=kwargs['xmin'])
        if not_empty(kwargs, 'xmax'):
            ax.set_xlim(right=kwargs['xmax'])

        if not_empty(kwargs, 'xlabel'):
            ax.set_xlabel(kwargs['xlabel'], x=1, ha='right')        
        if not_empty(kwargs, 'xticks'):
            ax.set_xticks(kwargs['xticks'])
        if not_empty(kwargs, 'xticklabels'):
            if 'xtickrot' in kwargs:
                ax.set_xticklabels(kwargs['xticklabels'], rotation=kwargs['xtickrot'])
            else:
                ax.set_xticklabels(kwargs['xticklabels'])
        if ax.get_xscale() != 'log':
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if ax.get_yscale() != 'log':
            if 'yticks' not in kwargs or not kwargs['yticks']:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune='upper' if (not ax.ratio and not ax.primary) else None))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_label_coords(-0.15,1)
        for spine in ax.spines.values(): 
            spine.set_zorder(100)

    # Clean up for ratio plots
    if axes[-1].ratio:
        for ax in axes[:-1]:
            for tick in ax.get_xticklabels():
                tick.set_visible(False)
            ax.set_xlabel("")
        axes[-1].yaxis.set_major_locator(ticker.MaxNLocator(4, prune='upper'))
        axes[-1].yaxis.grid(True) # Show lines on tick marks
        if not_empty(kwargs, 'ratio_sb') and kwargs['ratio_sb']:
            axes[-1].yaxis.get_major_formatter().set_powerlimits((-1,1))

        
def text(ax, text=None, text_location="upper left", **kwargs):
    if text:
        locations = {'best': 0,
                     'upper right':1,
                     'upper left':2,
                     'lower left':3,
                     'lower right':4,
                     'right':5,
                     'center left':6,
                     'center right':7,
                     'lower center':8,
                     'upper center':9,
                     'center':10}
        if text_location in locations:
            text_location = locations[text_location]
        text = text.replace("INTERNAL", INTERNAL)
        text = text.replace("PRELIM", PRELIM)
        text = text.replace(";", "\n")
        anchored = AnchoredText(text, prop=dict(size=18), loc=text_location, frameon=False)
        ax.add_artist(anchored)


def legend(ax, hists, legend_loc='best', legend_text=None, **kwargs):
    # Don't create a legend if there are not labels
    if not any(h.title for h in hists):
        return

    # Don't make a legend on the ratio plot
    if ax.ratio:
        return

    # Shortcut for common labels
    if legend_text == "INTERNAL":
        legend_text = INTERNAL
    if legend_text == "PRELIM":
        legend_text = PRELIM
        
    legend = ax.legend(frameon=False, loc=legend_loc, prop={'size':18}, labelspacing=0.25, ncol=1 if len(hists) < 5 else 2)

    # Add extra text above legend.
    if legend_text and legend_text is not "none":
        for sub in legend_text.split(";")[::-1]:
            txt=offsetbox.TextArea(sub, {'size':18}) 
            box = legend._legend_box 
            box.get_children().insert(0,txt) 
            box.set_figure(box.figure) 
    return legend



def plot(draw, name, hists, **kwargs):
    """
    Performs basic canvas setup and draws hists with draw, saving the plot to name.

    kwargs are passed around to all subfunctions
    """
    fig, axes = build_axes(**kwargs)

    # Set log scales first thing, because some functions check for log scales
    set_scales(axes, **kwargs)

    # Normalize input hists if specified
    normalize(hists, **kwargs)
    
    # Split histograms into group for each axes
    hgroups = process_hists(hists, **kwargs)

    # It might be necessary for some plotting functions to set approx ylims first?
    # Come back to this...
    """
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
    """

    for ax, group in izip(axes, hgroups):
        draw(ax, group, **kwargs)
        if draw != draw2d:
            if ax.primary: text(ax, **kwargs)
            legend(ax, group, **kwargs)

    # Setup axes last so that nothing gets overwritten
    setup_axes(*axes, **kwargs)

    if any(x in name for x in ["png","jpg"]):
        fig.savefig(name, dpi=500)
    else:
        fig.savefig(name)


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

def band(ax, hists, **kwargs):
    """
    Fill the region between bottom and top, for each pair.

    kwargs are passed to fill_between
    """
    # Override some defaults
    rplt.hist(hists, axes=ax, **kwargs)
    defaults = {"linewidth":0.0, "alpha":0.8}
    defaults.update(kwargs)
    for h in make_iterable(hists):
        b = h.Clone()
        t = h.Clone()
        b.title = ""
        t.title = ""
        for i in t.bins_range():
            b.SetBinContent(i,b.GetBinContent(i) - b.GetBinError(i))
            t.SetBinContent(i,t.GetBinContent(i) + t.GetBinError(i))
        rplt._set_defaults(h, defaults, ['common', 'line']) 
        if defaults.get('color') is None:
            defaults['color'] = h.GetLineColor('mpl')
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
    for h in make_iterable(hists):
        _step(ax, h, **kwargs)

def steperr(ax, hists, **kwargs):
    step(ax, hists, **kwargs)
    for h in make_iterable(hists):
        eb = rplt.errorbar(hists, axes=ax, fmt='none', capsize=0, xerr=None, label='', elinewidth=h.linewidth)
        eb[-1][0].set_linestyle(h.linestyle) 

def hist(ax, hists, **kwargs):
    defaults = {'fill':None,
                'capsize':0,
                'stacked':False}
    defaults.update(kwargs)
    for hist in make_iterable(hists):
        # Need the histify syntax to handle graphs
        rplt.hist(histify(hist), axes=ax, **defaults)

def herr(ax, hists, **kwargs):
    defaults = {'marker':None,
                'capsize':0,
                'stacked':False}
    defaults.update(kwargs)
    hist(ax, hists, **defaults)
    for h in make_iterable(hists):
        eb = rplt.errorbar(h, axes=ax, fmt='none', xerr=None, label='', capsize=0, elinewidth=h.linewidth)
        eb[-1][0].set_linestyle(h.linestyle) 
        
def stack(ax, hists, **kwargs):
    defaults = {'fill':'solid',
                'capsize':0,
                'yerr':None}
    defaults.update(kwargs)
    rplt.hist(hists, axes=ax, **defaults)

def hist2d(ax, hists, logz=False, **kwargs):
    if logz:
        kwargs["norm"] = LogNorm()
    defaults = {"colorbar":True}
    defaults.update(kwargs)
    if not not_empty(kwargs, 'cmap'):
        kwargs['cmap'] = 'Blues'
    rplt.hist2d(hists, axes=ax, **defaults)

_plot_functions = {"errorbar":errorbar,
                   "hist":hist,
                   "herr":herr,
                   "stack":stack,
                   "band":band,
                   "step":step,
                   "steperr":steperr}

def draw1d(ax, hists, **kwargs):
    stacked = [h for h in hists if h.drawstyle == 'stack']
    unstacked = [h for h in hists if h.drawstyle != 'stack']
    if stacked: stack(ax, stacked)
    for h in unstacked:
        if h.drawstyle == '': h.drawstyle = 'errorbar'
        if h.drawstyle not in _plot_functions:
            raise PlotError("Do not recognize drawstyle: " + h.drawstyle)
        _plot_functions[h.drawstyle](ax, h)


def draw2d(ax, hists, logz=False, **kwargs):
    hist2d(ax, hists[0], logz=logz, cmap=hists[0].cmap)

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

    
# ----------------------------------------
# Command Line Plotting
# ----------------------------------------


def main(args):
    parser = argparse.ArgumentParser('Create a plot from ntuples, using variable and sample lookup information from spreadsheets. Any additional arguments are taken as overrides for canvas options. (e.g. --title "Title")')
    parser.add_argument('output', nargs="?", help='Name of the plot to save.')
    parser.add_argument('--variables', nargs='*', help='The names of variables to add to the plot. Each variable gets a separate graphic.')
    parser.add_argument('--canvas', help="Canvas to use for the plot, if unspecified will start with a blank canvas.")
    parser.add_argument('--selection', help='Additional global selection to apply for all loaded data.')
    parser.add_argument('--batch', help='Specify a file which contains a set of arguments to run on each line.')

    args, extras = parser.parse_known_args(args)

    extras = dict(zip([key.replace('--','') for key in extras[:-1:2]], extras[1::2]))

    if (not args.output or not len(args.variables)) and not args.batch:
        raise PlotError("Must provide an output name and variable for the plot or a batch file to run on.")

    if args.batch:
        with open(args.batch) as f:
            lines = f.readlines()
        # Maybe ship this out in parallel?
        for next_args in (l for l in lines if l.strip() != ""):
            try:
                main(shlex.split(next_args))
            except BaseException as e:
                print "Plotting failed with ", e
                print "  and arguments", next_args
        return

    hists = retrieve.retrieve_all(args.variables, args.selection)

    if args.canvas:
        with open(canvas_json) as f:    
            canvas = json.load(f)[args.canvas]
        canvas.update(extras)
        plot_args = canvas
    else:
        plot_args = extras

    # ----------------------------------------
    # Select Draw Function and Plot
    # ----------------------------------------
    if '2' in hists[0].__class__.__name__:
        draw_function = draw2d
    else:
        draw_function = draw1d

    # Create figure
    plot(draw_function, args.output, hists, **plot_args)
    

if __name__ == "__main__":
    main(sys.argv[1:])
