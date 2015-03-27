#!/usr/bin/env python

"""
A module to produce scientific plots in a number of styles.
Using matplotlib to generate the plots, because it tends to 
cause less headaches than ROOT, and has some excellent features 
like full latex support for all text.
"""

import os
import sys
import argparse
import brewer2mpl
import style
import numpy as np
import uncertainties.unumpy as up
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.offsetbox as offsetbox
import cPickle as pickle
from copy import copy
from matplotlib.colors import LogNorm


def recursive_remove(obj):
    try:
        for item in obj:
            recursive_remove(item)
    except TypeError: #Not iterable
        try:
            obj.remove()
        except ValueError: #Not on the figure
            pass

def replace_above(a, top, val):
    for i in xrange(len(a)):
        if a[i] > top: a[i] = val

def replace_below(a, bot, val):
    for i in xrange(len(a)):
        if a[i] < bot: a[i] = val

################################################################################
# Utility Classes

class PlotError(Exception):
    pass


class Color:
    """ A static class to provide new colors for each new plotted Hist.

    The color scheme can be changed by setting Color.values to a list of
    colors. 
    
    Color.default provides the color for things like data.
    """
    index = -1
    default = '#333333'
    values = style.neutral[1:]
    @classmethod
    def next(cls):
        """ Cycles through the available colors in order. """
        cls.index += 1
        return cls.values[cls.index%len(cls.values)]
    @classmethod
    def reset(cls):
        """ Resets the cycle to the first color. """
        cls.index = -1
    @classmethod
    def by_index(cls, i):
        return cls.values[i]


class LineStyle:
    """ A static class to provide new linestyles for each new plotted Hist.

    The linestyles used can be changed by setting LineStyle.values to a list of matplotlib ls strings.
    """
    index = -1
    values = ['-' , '--']
    @classmethod
    def next(cls):
        """ Cycles through the available linestyles in order. """
        cls.index += 1
        return cls.values[cls.index%len(cls.values)]
    @classmethod
    def reset(cls):
        """ Resets the cycle to the first linestyle. """
        cls.index = -1

################################################################################




################################################################################
# Histogram classes, which store data and draws them on an axis.

class Hist(np.ndarray):
    """Stores an array of values and uncertainties, as well as associated
    bin edges/widths.  It inherits from numpy's ndarray, so operations like 
    (h1-h2)/h3 
    are supported. Complicated indexing too: 
    h1[3:10:-1]
    h1[h1 > 10]


    Arguments:
    lefts: The starting x-values for the bins (np array).
    widths: The widths of the bins (np array).
    values: The 'height' of the bins (np array).
    errors: Errors on the height of the bins (np array).
    
    Keyword Arguments:
    label: The label to place in any legend.
    color: The color used on the plot (defaults to a color cycle)
    style: Specifies the way the histogram plots itself on an axis using "plot"
             valid options are the second words in the plot_ functions of Hist.
    nonzero: Remove bins which have zero content.
    exclude: Remove bins which intersect the specified interval (a,b).
    xscale: Multiply the x-values by the specified constant.
    yscale: Multiply the y-values by the specified constant.
    binscale: Divide each value/yerr by the width of the corresponding bin.
    norm: Multiply the heights by a constant so that the integral is norm.
    peaknorm: Multiply the heights by a constant so that the peak value is peaknorm.

    Notes
    All remaining kwargs are passed to matplotlib functions when plotted.
    """
    def __new__(cls, lefts, widths, values, errors=None, label=None, xlabel=None, ylabel=None, title=None, color=None, **kwargs):
        if errors is None:
            errors = np.zeros_like(values) 
        if len(lefts) != len(widths) or len(widths) != len(values) or len(values) != len(errors):
            raise PlotError("Length of arrays provided to hist do not match.")
        values = up.uarray(values, errors)
        # Handle options which modify the data.
        if 'nonzero' in kwargs:
            if kwargs.pop('nonzero'):
                indices = np.where(values != 0)
                lefts = lefts[indices]
                widths = widths[indices]
                values = values[indices]
        if 'exclude' in kwargs: 
            interval = kwargs.pop('exclude')
            indices = np.where( (lefts + widths <= interval[0]) | (lefts >= interval[1]) )[0]
            lefts = lefts[indices]
            widths = widths[indices]
            values = values[indices]
        if 'binscale' in kwargs:
            kwargs.pop('binscale')
            values /= widths
        if 'yscale' in kwargs:
            values = kwargs.pop('yscale') * values
        if 'xscale' in kwargs:
            factor = kwargs.pop('xscale')
            lefts = lefts * factor
            widths = widths * factor
        if 'peaknorm' in kwargs:
            factor = kwargs.pop('peaknorm')
            if np.max(values) != 0:
                factor /= np.max(values)
                values = factor * values
        if 'norm' in kwargs:
            factor = kwargs.pop('norm')
            if np.sum(values) != 0:
                factor /= np.sum(values)
                values = factor * values

        # Create the actual object: an ndarray with additional properties.
        obj = values.view(cls)
        obj.lefts = lefts
        obj.widths = widths
        if color is None:
            obj.color = Color.next()
        else:
            obj.color = color
        obj.label = label
        obj.xlabel = xlabel
        obj.ylabel = ylabel
        obj.title = title
        if 'style' in kwargs:
            obj.style = kwargs.pop('style')
        else:
            obj.style = 'lines'
        # Remaining options will be passed to mpl when plotting.
        obj.options = kwargs
        obj._collections = None
        return obj

    def __array_finalize__(self, obj):
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. X():
        #    obj is None
        #    (we're in the middle of the X.__new__
        #    constructor, and self.info will be set when we return to
        #    X.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(X):
        #    obj is arr
        #    (type(obj) can be X)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is X
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # X.__new__ constructor, but also with
        # arr.view(X).
        self.lefts        = getattr(obj, 'lefts', np.arange(np.asarray(obj).shape[0]))
        self.widths       = getattr(obj, 'widths', np.ones_like(self.lefts))
        self.color        = getattr(obj, 'color', style.def_grey)
        self.label        = getattr(obj, 'label', None)
        self.xlabel       = getattr(obj, 'xlabel', None)
        self.ylabel       = getattr(obj, 'ylabel', None)
        self.title        = getattr(obj, 'title', None) 
        self.style        = getattr(obj, 'style', 'lines')
        self.options      = getattr(obj, 'options', {})
        self._collections = getattr(obj, '_collection', None)
        


    @classmethod
    def from_bin_edges(cls, bin_edges, values, errors=None, **kwargs):
        bin_edges = np.array(bin_edges)
        return cls(bin_edges[:-1], np.diff(bin_edges), values, errors, **kwargs)

    @classmethod
    def from_centers(cls, centers, values, errors=None, **kwargs):
        centers = np.array(centers)
        diffs = np.diff(centers)
        diffs = np.append(diffs, diffs[-1])
        return cls(centers-(diffs/2.0), diffs, values, errors, **kwargs)

    @classmethod
    def from_rhist(cls, hist, **options):
        """Create a Hist from a ROOT TH1.

        Keyword Arguments:
        rebin -- Use ROOT's Rebin function before loading.
        overflow -- Add overflow content to last bin.
        
        All remaining kwargs are passed to the Hist constructor.
        """
        if 'rebin' in options:
            hist.Rebin(options.pop('rebin'))

        nbins = hist.GetNbinsX()

        if 'overflow' in options and options.pop('overflow'):
            hist.SetBinContent(nbins,hist.GetBinContent(nbins)+hist.GetBinContent(nbins+1))
            hist.SetBinError(nbins,np.sqrt(hist.GetBinError(nbins)**2+hist.GetBinError(nbins+1)**2))            
                
        bin_edges = np.fromiter((hist.GetBinLowEdge(i) for i in xrange(1,nbins+2)), np.float, nbins+1)
        values = np.fromiter((hist.GetBinContent(i) for i in xrange(1,nbins+1)), np.float, nbins)
        errors = np.fromiter((hist.GetBinError(i) for i in xrange(1,nbins+1)), np.float, nbins)
        xlabel = hist.GetXaxis().GetTitle()
        ylabel = hist.GetYaxis().GetTitle()
        title = hist.GetTitle()
        return cls.from_bin_edges(bin_edges,values,errors,xlabel=xlabel, ylabel=ylabel, title=title, **options)

    def __getitem__(self, val):
        """Overload slicing to update lefts/widths in parallel."""
        if isinstance(val, tuple) and isinstance(val[0], np.ndarray): # Array indexing multiple dimensions
            val = val[0]
        if isinstance(val, slice) or isinstance(val, np.ndarray): # Slices and bool indexing
            x = super(Hist, self).__getitem__(val).view(Hist)
            x.lefts        = self.lefts[val]
            x.widths       = self.widths[val]
            x.color        = self.color
            x.label        = self.label
            x.xlabel       = self.xlabel
            x.ylabel       = self.ylabel
            x.title        = self.title
            x.style        = self.style
            x.options      = self.options
            x._collections = self._collections
            return x
        return super(Hist, self).__getitem__(val) # Single elements

    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def plot_bars(self, ax, bottom=None, trim=False):
        """Plot histogram as a filled bar graph. (Like 'HistStack')"""
        log = ax.yaxis.get_scale=='log'
        if bottom is None and log:
            bottom = np.ones_like(up.nominal_values(self)) * min(up.nominal_values(self)[up.nominal_values(self) > 0 ]) * .1

        heights = up.nominal_values(self)
        if trim:
            ymin, ymax = ax.get_ybound()
            replace_below(bottom, ymin, ymin)
            for i,top in enumerate(bottoms+heights):
                if top > ymax:
                    heights[i] = ymax-bottom[i]

        
        self._collections = ax.bar(self.lefts, heights, self.widths, color=self.color, label=self.label,edgecolor=self.color, bottom=bottom, log=log, **self.options)
        return self._collections[0]

    def plot_lines(self, ax, trim=False):
        """Plot histogram as stepped lines with points and errorbars. (Like 'E0')"""
        values =  up.nominal_values(self)
        errors = up.std_devs(self)
        if trim:
            ymin, ymax = ax.get_ybound()
            for i in xrange(len(values)):
                if values[i] > ymax:
                    errors[i] = abs(ymax + (ymax-ymin)*0.01 - (values[i] - errors[i]))
                    values[i] = ymax + (ymax-ymin)*0.01
                if values[i] < ymin:
                    errors[i] = abs(ymin - (ymax-ymin)*0.01 - (values[i] + errors[i]))
                    values[i] = ymin - (ymax-ymin)*0.01
        
        self._collections = ax.errorbar(self.lefts + (self.widths*.5), values, yerr=errors, xerr=.5*self.widths, marker='o', color=self.color, label=self.label, capsize=0, ls='', **self.options)
        return self._collections

    def plot_band(self, ax, trim=True):
        """Plot histogram as a central line with errorbands."""
        if ax.yaxis.get_scale=='log':
            index = up.nominal_values(self) - up.std_devs(self) > 0
            lefts = self.lefts[index]
            widths = self.widths[index]
            errors = np.asarray(up.std_devs(self))[index]
            values = np.asarray(up.nominal_values(self))[index]
        else:
            lefts  = self.lefts[:]
            widths = self.widths[:]
            errors = np.asarray(up.std_devs(self))
            values = np.asarray(up.nominal_values(self))

        lefts = np.empty((2*self.lefts.size,), dtype=self.lefts.dtype)
        lefts[0::2] = self.lefts
        lefts[1::2] = self.lefts + self.widths

        values = np.repeat(values, 2)
        errors = np.repeat(errors, 2)

        if trim:
            ymin, ymax = ax.get_ybound()
            for i in xrange(len(values)):
                if values[i] > ymax:
                    errors[i] = abs(ymax + (ymax-ymin)*0.01 - (values[i] - errors[i]))
                    values[i] = ymax + (ymax-ymin)*0.01
                if values[i] < ymin:
                    errors[i] = abs(ymin - (ymax-ymin)*0.01 - (values[i] + errors[i]))
                    values[i] = ymin - (ymax-ymin)*0.01
        

        options = copy(self.options)
        options['linewidth'] = 0
        options['linestyle'] = '-'
        self._collections = [ax.fill_between(lefts, values-errors, values+errors, color=self.color, label=self.label, antialiased=True, facecolor=self.color, alpha=0.5, **options)]
        ind = len(self._collections)
        self._collections += list(ax.plot(lefts, values, color=self.color, label=self.label, **self.options))
        return self._collections[ind]

    def plot_points(self, ax, trim=False):
        """Plot histogram as points with errorbars."""
        values =  up.nominal_values(self)
        errors = up.std_devs(self)
        if trim:
            ymin, ymax = ax.get_ybound()
            for i in xrange(len(values)):
                if values[i] > ymax:
                    errors[i] = abs(ymax + (ymax-ymin)*0.01 - (values[i] - errors[i]))
                    values[i] = ymax + (ymax-ymin)*0.01
                if values[i] < ymin:
                    errors[i] = abs(ymin - (ymax-ymin)*0.01 - (values[i] + errors[i]))
                    values[i] = ymin - (ymax-ymin)*0.01
        
        self._collections = ax.errorbar(self.lefts + (self.widths*.5), values, yerr=errors, marker='o', color=self.color, label=self.label, ls='', capsize=0, **self.options)
        return self._collections

    def plot_box(self, ax, trim=True):
        heights = 2*up.std_devs(self)
        bottoms = up.nominal_values(self)-up.std_devs(self)
        lefts, widths = copy(self.lefts), copy(self.widths)
        if trim:
            ymin, ymax = ax.get_ybound()
            replace_below(bottoms, ymin, ymin)
            for i,top in enumerate(bottoms+heights):
                if top > ymax:
                    heights[i] = ymax-bottoms[i]
            xmin, xmax = ax.get_xbound()
            for i in xrange(len(self.lefts)):
                if lefts[i] < xmin < lefts[i] + widths[i]:
                    lefts[i] = xmin
                    widths[i] = lefts[i] + widths[i] - xmin
                if lefts[i] < xmax < lefts[i] + widths[i]:
                    widths[i] = xmax - lefts[i]
        middles = bottoms + heights/2
        centers = lefts + .5*widths
        for i in xrange(len(lefts)):
            ax.plot([lefts[i], lefts[i] + widths[i]], [middles[i],middles[i]], color=self.color, **self.options)
        self._collections = ax.bar(centers, heights, bottom=bottoms, width=widths, color=self.color, alpha=0.5, label=self.label, align='center', linewidth=0, **self.options)
        return self._collections

    def plot_curve(self, ax, trim=False):
        """Plot histogram as an interpolated curve."""
        indices = []
        nz = False
        for value in up.nominal_values(self):
            if value != 0:
                nz = True
            indices.append(nz)
        indices = np.array(indices)
        #tmp_lefts = (self.lefts + (self.widths/2.0))[indices]
        #tmp_values = up.nominal_values(self)[indices]
        if trim:
            ymin, ymax = ax.get_ybound()
            replace_above(tmp_values, ymax, ymax + (ymax-ymin)*.01)
            replace_below(tmp_values, ymax, ymin - (ymax-ymin)*.01)
        tmp_lefts = (self.lefts + (self.widths/2.0))
        tmp_values = up.nominal_values(self)
        self._collections = ax.plot(tmp_lefts, tmp_values, color=self.color, label=self.label, **self.options)
        return self._collections[0]

    def plot_noerror(self, ax, trim=False):
        """Plot histogram as stepped lines without errorbars. (Like 'hist')"""
        tmp_lefts = copy(self.lefts)
        tmp_lefts = np.append(tmp_lefts, self.lefts[-1] + self.widths[-1])
        tmp_values = copy(up.nominal_values(self))
        tmp_values = np.append(tmp_values, up.nominal_values(self)[-1])
        if trim:
            ymin, ymax = ax.get_ybound()
            replace_above(tmp_values, ymax, ymax + (ymax-ymin)*.01)
            replace_below(tmp_values, ymax, ymin - (ymax-ymin)*.01)
        self._collections = ax.plot(tmp_lefts, tmp_values, color=self.color, drawstyle='steps-post', label=self.label, **self.options)
        return self._collections[0]

    def plot(self, ax, trim=True):
        """ Plot histogram using function below given by self.style"""
        try:
            return getattr(self, 'plot_'+self.style)(ax, trim)
        except AttributeError:
            raise PlotError('Unknown plotting style {0}'.format(self.style))

    def remove(self):
        recursive_remove(self._collections)



class Hist2D(np.ndarray):
    """WARNING: Still being tested.
    
    The primary concern is that some indexing options from np.ndarray
    will flatten the internal array, which breaks plotting. For now,
    this works as long as you don't try to use those features. Should
    propably return a Hist (1D) when it happens.

    Stores an array of values and uncertainties, as well as associated
    bin edges/widths in 2 dimensions.  It inherits from numpy's ndarray, so 
    operations like 
    (h1-h2)/h3 
    are supported. Complicated indexing too: 
    h1[:,4:6]

    It stores it's bins as 2d arrays (xgrid, ygrid) from the np.meshgrid function.

    Arguments:
    xbins: The values for the bin edges of x-axis (np array).
    ybins: The values for the bin edges of y-axis (np array).
    values: The 'height' of the bins (np array).
    errors: Errors on the height of the bins (np array).
    
    Keyword Arguments:
    label: The label to place in any legend.
    color: The color used on the plot (defaults to a color cycle)
    style: Specifies the way the histogram plots itself on an axis using "plot"
             valid options are the second words in the plot_ functions of Hist.
    xscale: Multiply the x-values by the specified constant.
    yscale: Multiply the y-values by the specified constant.
    norm: Multiply the heights by a constant so that the integral is norm.
    peaknorm: Multiply the heights by a constant so that the peak value is peaknorm.

    Notes
    All remaining kwargs are passed to matplotlib functions when plotted.
    """
    def __new__(cls, xbin_edges, ybin_edges, values, errors=None, label=None, xlabel=None, ylabel=None, title=None, color=None, **kwargs):
        if errors is None:
            errors = np.ones_like(values) 
        if values.shape != errors.shape:
            raise PlotError("Shape of arrays provided to hist do not match.")
        values = up.uarray(values, errors)
        # Handle options which modify the data.
        if 'yscale' in kwargs:
            ybin_edges = kwargs.pop('yscale') * ybin_edges
        if 'xscale' in kwargs:
            xbin_edges = kwargs.pop('xscale') * xbin_edges
        if 'peaknorm' in kwargs:
            factor = kwargs.pop('peaknorm') / np.max(values)
            values = factor * values
        if 'norm' in kwargs:
            factor = kwargs.pop('norm') / np.sum(values)
            values = factor * values

        # Create the actual object: an ndarray with additional properties.
        obj = values.view(cls)
        obj.xgrid, obj.ygrid = np.meshgrid(xbin_edges, ybin_edges)
        if color is None:
            obj.color = brewer2mpl.get_map('Greens', 'sequential', 8).mpl_colormap
        else:
            obj.color = brewer2mpl.get_map(color, 'sequential', 8).mpl_colormap
        obj.label = label
        obj.xlabel = xlabel
        obj.ylabel = ylabel
        obj.title = title
        if 'style' in kwargs:
            obj.style = kwargs.pop('style')
        else:
            obj.style = 'colors'
        # Remaining options will be passed to mpl when plotting.
        obj.options = kwargs
        return obj

    def __array_finalize__(self, obj):
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. X():
        #    obj is None
        #    (we're in the middle of the X.__new__
        #    constructor, and self.info will be set when we return to
        #    X.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(X):
        #    obj is arr
        #    (type(obj) can be X)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is X
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # X.__new__ constructor, but also with
        # arr.view(X).
        self.xgrid   = getattr(obj, 'xgrid', np.ones_like(np.asarray(obj)))
        self.ygrid   = getattr(obj, 'ygrid', np.ones_like(np.asarray(obj)))
        self.color   = getattr(obj, 'color', style.def_grey)
        self.label   = getattr(obj, 'label', None)
        self.xlabel  = getattr(obj, 'xlabel', None)
        self.ylabel  = getattr(obj, 'ylabel', None)
        self.title   = getattr(obj, 'title', None)        
        self.style   = getattr(obj, 'style', 'lines')
        self.options = getattr(obj, 'options', {})
        
    @classmethod
    def from_rhist(cls, hist, **options):
        """Create a Hist2D from a ROOT TH2."""
        if 'rebin' in options:
            hist.Rebin(options.pop('rebin'))
        nbinsx = hist.GetNbinsX()
        nbinsy = hist.GetNbinsY()
        xbin_edges = np.fromiter((hist.GetXaxis().GetBinLowEdge(i) for i in xrange(1,nbinsx+2)), np.float, nbinsx+1)
        ybin_edges = np.fromiter((hist.GetYaxis().GetBinLowEdge(i) for i in xrange(1,nbinsy+2)), np.float, nbinsy+1)
        values = [[hist.GetBinContent(i,j) for i in xrange(1,nbinsx+1)] for j in xrange(1,nbinsy+1)]
        values = np.array(values)
        xlabel = hist.GetXaxis().GetTitle()
        ylabel = hist.GetYaxis().GetTitle()
        title = hist.GetTitle()        
        return cls(xbin_edges, ybin_edges, values, xlabel=xlabel, ylabel=ylabel, title=title, **options)

    def __getitem__(self, val):
        """Overload slicing to update lefts/widths in parallel."""
        if isinstance(val, tuple) and isinstance(val[0], np.ndarray): # Array indexing multiple dimensions
            val = val[0]
        if isinstance(val, slice) or isinstance(val, np.ndarray): # Slices and bool indexing
            x = super(Hist2D, self).__getitem__(val).view(Hist2D)
            x.xgrid   = self.xgrid[val] 
            x.ygrid   = self.ygrid[val]
            x.color   = self.color
            x.label   = self.label
            x.xlabel  = self.xlabel
            x.ylabel  = self.ylabel
            x.title   = self.title
            x.style   = self.style
            x.options = self.options
            return x
        return super(Hist2D, self).__getitem__(val) # Single elements

    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def plot_colors(self, log=False):
        """Plot 2D histogram as a grid of colors."""
        # still needs to be implemented
        return plt.pcolor(self.xgrid, self.ygrid, up.nominal_values(self), cmap=self.color, **self.options)

    def plot(self, ax):
        """ Plot histogram using function below given by self.style"""
        try:
            return getattr(self, 'plot_'+self.style)(ax)
        except AttributeError:
            raise PlotError('Unknown plotting style {0}'.format(self.style))


class HistStack:
    """Container of histograms that plots them as a stacked bar graph."""
    def __init__(self, *hists):
        self.hists = hists
    
    def plot_bars(self, ax, bottom=None, log=False):
        """Plot all histograms as stacked bar plots."""
        handles = []
        for hist in self.hists:
            handles.append(hist.plot_bars(ax, bottom))
            if bottom is None:
                bottom = copy(up.nominal_values(hist))
            else:
                bottom += up.nominal_values(hist)
        return handles

    def sum_hist(self, label=None):
        """Return a histogram which is equal to the sum of all histograms in the stack."""
        hist = sum(hists)
        hist.label = label
        hist.color = style.def_grey
        return hist


# End of histogram classes.
################################################################################




################################################################################
# Plot setup functions.
# These are convenient but should be largely replaced by the template system.

def setup_axes(ax, ratio=False, prune=None, **kwargs):
    """ Setup an axis with standard options. 

    Just a shortcut for calling a bunch of axes member functions.
    
    Keyword Arguments:
    ratio -- If true, set up for a ratio plot.
    prune -- Removes 'upper' or 'lower' tick label from axis.
    title -- Text to be placed on top center of axes.
    logy -- If true, set yscale to log.
    logx -- If true, set xscale to log.
    xmin -- Set minimum of x-axis.
    xmax -- Set maximum of x-axis.
    ymin -- Set minimum of y-axis.
    ymax -- Set maximum of y-axis.
    xlabel -- Label for x-axis.
    ylabel -- Label for y-axis.
    xticks -- A dict of locations, replacements for xtick labels.
    """
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5, prune=prune))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
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
    if 'ylabel' in kwargs and not ratio:
        ax.set_ylabel(kwargs['ylabel'], y=1, ha='right')
    if 'ylabel' in kwargs and ratio:
        ax.set_ylabel(kwargs['ylabel'])        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], x=1, ha='right')        
    if 'xticks' in kwargs:
        tlocs, tlabels = kwargs['xticks'].keys(), kwargs['xticks'].values()
        ax.set_xticks(tlocs)
        ax.set_xticklabels(tlabels)
    if not ratio:
        ax.yaxis.set_label_coords(-0.12,1)
    else:
        ax.yaxis.set_label_coords(-0.12,.5) # centered label for ratio y-axis
        ax.yaxis.grid(True) # Show lines on tick marks
        xmin, xmax, ymin, ymax = ax.axis()
        # Custom tick locations for some common ratio plot limits
        if ymin == 0.0 and ymax == 2.0:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.5,1.0,1.5]))
        if ymin == 0.0 and ymax == 1.25:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.5,1.0]))
        elif ymin == 0.75 and ymax == 1.5:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.8,1.0,1.2,1.4]))
        elif ymin == 0.9 and ymax == 1.6:
            ax.yaxis.set_major_locator(ticker.FixedLocator([1.0,1.5]))        
        elif ymin == 0.8 and ymax == 1.2:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.9,1.0,1.1]))
        elif ymin == 0.9 and ymax == 1.2:
            ax.yaxis.set_major_locator(ticker.FixedLocator([1.0,1.1]))
        elif ymin == 0.9 and ymax == 1.1:
            ax.yaxis.set_major_locator(ticker.FixedLocator([.95,1.0,1.05]))
        elif ymin == 0.85 and ymax == 1.15:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.9,1.0,1.1]))
        elif ymin == 0.75 and ymax == 1.25:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.9,1.0,1.1]))
        elif ymin == 0.8 and ymax == 2.0:
            ax.yaxis.set_major_locator(ticker.FixedLocator([1.0,1.5]))
        elif ymin == 0.5 and ymax == 1.5:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.75,1.0,1.25]))


# End of plot setup functions.
######################################################################




######################################################################
# Data input functions.

def get_rhist(hname, fname, **kwargs):
    """ Return a histogram from a ROOT TH1 stored in a .root file.
    
    kwargs are passed to the call to Hist constructor. 
    """
    from ROOT import TFile
    root_file = TFile.Open(fname) 
    rhist = root_file.Get(hname)
    if not rhist:
        print "Could not find {0} in {1}.".format(hname, fname)
    if "TH2" in rhist.ClassName():
        h = Hist2D.from_rhist(rhist, **kwargs)
    else:
        h = Hist.from_rhist(rhist, **kwargs)
    root_file.Close()
    return h


def get_pdd(hname, fname, **kwargs):
    """ Return a histogram from a pickle of a dictionary of dictionaries.
    
    This assumes that the keys of the internal dictionary should be
    made into evenly spaced tick labels for the plot.
    The values of the dictionary should be tuples of (val,err).
    """
    with open(fname, "rb") as f:
        dd = pickle.load(f)
    keys = sorted(dd[hname].keys())
    values = [dd[hname][key][0] for key in keys]
    errors = [dd[hname][key][1] for key in keys]
    centers = range(len(keys))
    return Hist.from_centers(centers, values, errors=errors, **kwargs)

# In the future there should be more of these, but these are the only
# ones I regularly use right now. Ideas: np.save(), pickle of hist

input_types = {"root" : get_rhist,
               "pdd"  : get_pdd}


def expand_targets(files, hnames):
    """ Expands list of target arguments into two lists: file names and hist names. 

    Notes:
    Supports a number of syntaxes which are described in help.

    targets -- A list which uses shorthand to cover the files/histograms.
        Targets can be a single file and a list of histograms or a list of files and a single histogram.
        For example:
            file1.root hist1 hist2 hist3
            hist1 file1.root file2.root file3.root

            For the more general case, targets can be specified as:
            file1.root hist1 hist2 file2.root hist3 hist4 ...
    """
    if len(files) == 1:
        files = [files[0] for i in range(len(hnames))]
    elif len(hnames) == 1:
        hnames = [hnames[0] for i in range(len(files))]
    elif len(files) != len(hnames):
        raise PlotError("Invalid set of targets. Check the help documentation.")
    return files,hnames

def get_hists(fnames, hnames, **kwargs):
    """ Returns a list of Hist or Hist2D given the input arguments.

    Allows shorthand to initialize the histograms from multiple file types.
    The supported file extensions are those in input_types.

    Arguments:
    fnames, hnames
    Lists which use shorthand to cover the files/histograms.
        Targets can be a single file and a list of histograms or a list of files and a single histogram.
        For example:
            [file1.root], [hist1, hist2, hist3]
            [file1.root, file2.root, file3.root], [hist1]

            For the more general case, targets can be specified as:
            [file1.root, file2.root, file3.root], [hist1, hist2, hist3]

    This supports all file extensions in input_types (which should be
    expanded eventually.)
    
    Keyword Arguments:
    Any valid keyword argument for a Hist constructor can be provided,
    and it should be a list with one value for each hist in targets or
    a single value which will be applied to all histograms. Additionally,
    the following special values can be used:

    color:"datavsmc" -- Uses black for any hist from a file with data in
        the name and alternating colors for all other hists.

    style:"datavsmc" -- Uses point styles in black for any
        hist from a file with data in the name and alternating line styles
        with errorbands for all other hists. 
    """
    fnames, hnames = expand_targets(fnames, hnames)

    Color.reset()
    LineStyle.reset()

    hist_args = {}
    for key,val in kwargs.iteritems():
        if key=="color" and val=="datavsmc":
            hist_args[key] = [Color.default if "data" in fname else Color.next() for fname in fnames]
        elif key=="style" and val=="datavsmc":
            hist_args[key] = ["lines" if "data" in fname else "box" for fname in fnames]
            #if 'linestyle' not in kwargs:
            #    hist_args['linestyle'] = ["solid" if "data" in fname else LineStyle.next() for fname in fnames]
            if 'zorder' not in kwargs:
                hist_args['zorder'] = [3 if "data" in fname else 2 for fname in fnames]
        else:
            if type(val) in [str,int,float,bool]:
                hist_args[key] = [val for h in hnames]
            elif len(val) != len(hnames):
                # Provided a list with the wrong length.
                raise PlotError("The length of the list provided for {0} does not match the number of histograms.".format(key))
            else:
                hist_args[key] = val

    hists = []
    for i in range(len(hnames)):
        hargs = dict((key,vals[i]) for key,vals in hist_args.iteritems())
        hists.append(input_types[fnames[i].split('.')[-1]](hnames[i], fnames[i], **hargs))
    return hists

# End of data input functions.
######################################################################




######################################################################
# Plotting functions. Each represents a single plot style.

def trim_hist(hist, ax):
    """ Return a copy of hists without any points that wouldn't display on ax. 

    These can cause visual problems with some styles and (even worse)
    cause pgf to print errors if its extremely far out of range.
    """
    ymin, ymax = ax.get_ybound()
    return hist[(up.nominal_values(hist) > ymin) & (up.nominal_values(hist) < ymax)] 


def add_legend(ax, refs, labels, legend_loc='best', legend_text=None):
    legend = ax.legend(refs, labels, frameon=False, loc=legend_loc, prop={'size':18}, labelspacing=0.25)

    # Add extra text below legend.
    if legend_text is not None:
        for sub in legend_text.split(";")[::-1]:
            txt=offsetbox.TextArea(sub, {'size':18}) 
            box = legend._legend_box 
            box.get_children().insert(0,txt) 
            box.set_figure(box.figure) 
    return legend

def compare(ax, hists, draw_legend=True, legend_loc="best", legend_text=None, trim=True):
    """ Plot hists side-by-side. 
    
    Keyword Arguments:
    draw_legend -- If true, draw a legend for all histograms. 
                     Uses their internal labels.
    legend_text -- If true, add text above the legend. 
                     Useful for putting flags like ATLAS internal.
    """
    refs, labels = [], []
    refs = [hist.plot(ax,trim) for hist in hists]
    labels = [hist.label for hist in hists]

    if draw_legend:
        legend = add_legend(ax, refs, labels, legend_loc=legend_loc, legend_text=legend_text)



def stack(ax, hists, draw_legend=True, legend_loc="best", legend_text=None):
    """ Plot histograms as a stack. """
    hs = HistStack(*hists)
    refs = hs.plot_bars(ax)
    labels = [hist.label for hist in hists]
    if draw_legend:
        legend = add_legend(ax, refs, labels, legend_loc=legend_loc, legend_text=legend_text)


def color2D(hist, logz=False, zmin=None, zmax=None, zlabel=None):
    """ Plot a 2D histogram as a colorbar graph. 

    This could use an update to work better with more than one subplot.
    """
    if logz:
        hist.options['norm'] = LogNorm(vmin=zmin, vmax=zmax)
    if zmax is not None:
        hist.options['vmax'] = zmax
    if zmin is not None:
        hist.options['vmin'] = zmin

    hist.plot_colors()
    cb = plt.colorbar()
    if zlabel is not None:
        cb.set_label(zlabel)
    return


def clear(ax, hists):
    """Remove the histograms and legend from the plot.
    
    Used for saving a figure setup without the content.
    """
    for h in hists:
        h.remove()
    ax.legend_ = None

# End of plotting functions.
################################################################################




################################################################################
# A few figure/axes generators for standard styles.
def axes_single():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, [ax]

def axes_ratio(figw=8.0,figh=6.0):
    fig = plt.figure(figsize=(figw, figh))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1], hspace=0.0, wspace=0.0)
    ax = plt.subplot(gs[0])
    return fig, [ax, plt.subplot(gs[1], sharex=ax)]

def axes_double_pad_ratio(figw=8.0, figh=8.0):
    fig = plt.figure(figsize=(figw, figh))
    gs = gridspec.GridSpec(3,1,height_ratios=[2,2,1], hspace=0.0, wspace=0.0)
    ax = plt.subplot(gs[0])
    return fig, [ax, plt.subplot(gs[1], sharex=ax), plt.subplot(gs[2], sharex=ax)]

################################################################################




################################################################################
# Plot formatting: save and load current figure setup (without the data).

# Need to accomplish: save a way to get the current figure/axes setup back. 
# This did not work with pickling the figure. (It saves too much, such as not 
# letting you recover an floating plot bound, and is too likely to break.)

class Formatter:
    def __init__(self, fig_properties={}, axes_properties=[], axes_calls=[]):
        self.fig_properties = fig_properties
        self.axes_properties = axes_properties
        self.axes_calls = axes_calls

    @classmethod
    def from_figure(cls, fig, **kwargs):
        """ Grab the format of the current figure and create a formatter
        that can recreate it. 

        Keyword Arguments:
        minimal: If true, do not record anything related to the the plot axes.
                 (Overall arrangement and titles are saved.)
        float_bounds: If True, do not record the axis ranges (so that 
                      matplotlib can determine them from the data.)
        """
        fig_properties = {'figsize':(fig.get_figwidth(),fig.get_figheight())}
        axes_properties = []
        axes_calls = []
        # What do we care about? Ticks, labels, bounds, text instances, titles, sizes
        # Grab all the axes and their properties.
        axes = fig.get_axes()
        for ax in axes:
            ax_prop = {}
            ax_prop['subplotspec'] = ax.get_subplotspec()
            ax_prop['sharex'] = axes.index(ax._sharex) if ax._sharex in axes else None
            ax_prop['sharey'] = axes.index(ax._sharey) if ax._sharey in axes else None

            ax_call = []

            if not kwargs.get('minimal'):
                xlabel = ax.xaxis.label
                ax_call.append(('set_xlabel', [xlabel.get_text()],{'ha':xlabel.get_ha(),'va':xlabel.get_va(),'x':xlabel.get_position()[0],'y':xlabel.get_position()[1],'visible':xlabel.get_visible()}))
                ylabel = ax.yaxis.label
                ax_call.append(('set_ylabel', [ylabel.get_text()],{'ha':ylabel.get_ha(),'va':ylabel.get_va(),'x':ylabel.get_position()[0],'y':ylabel.get_position()[1],'visible':xlabel.get_visible()}))
                ax_call.append(('set_xscale', [ax.get_xscale()],{}))
                ax_call.append(('set_yscale', [ax.get_yscale()],{}))
                ax_call.append(('set_title',  [ax.get_title()],{'y':1.04}))
                

            if not kwargs.get('minimal') and not kwargs.get('float_bounds'):
                ax_call.append(('set_xticks', [ax.get_xticks(minor=False)],{'minor':False}))
                ax_call.append(('set_yticks', [ax.get_yticks(minor=False)],{'minor':False}))
                ax_call.append(('set_xticklabels', [[tick.get_text() for tick in ax.get_xticklabels()]],{'visible':ax.get_xticklabels()[0].get_visible()}))
                ax_call.append(('set_yticklabels', [[tick.get_text() for tick in ax.get_yticklabels()]],{'visible':ax.get_yticklabels()[0].get_visible()}))
                ax_call.append(('set_xticks', [ax.get_xticks(minor=True)],{'minor':True}))
                ax_call.append(('set_yticks', [ax.get_yticks(minor=True)],{'minor':True}))
                ax_call.append(('set_xlim', [ax.get_xlim()], {}))
                ax_call.append(('set_ylim', [ax.get_ylim()], {}))

            axes_properties.append(ax_prop)
            axes_calls.append(ax_call)
        return cls(fig_properties, axes_properties, axes_calls)

    @classmethod
    def list(cls, filename="formats.p"):
        with open(filename, "rb") as f:
            formatters = pickle.load(f)
        return formatters.keys()

    @classmethod
    def load(cls, name, filename="formats.p"):
        with open(filename, "rb") as f:
            formatters = pickle.load(f)
        if name not in formatters:
            raise PlotError("Did not find saved format: {0} in file: {1}".format(name, filename))
        return cls(*formatters[name])

    def save(self, name, filename="formats.p"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                formatters = pickle.load(f)
        else:
            formatters = {}
        formatters[name] = self.fig_properties, self.axes_properties, self.axes_calls
        with open(filename, "wb") as f:
            pickle.dump(formatters, f)

    def create(self):
        """Returns a figure setup according to the Formatter's properties.
        
        Can use fig.get_axes() to get references to the axes as necessary.
        """
        fig = plt.figure(**self.fig_properties)
        axes = []
        for ax_prop in self.axes_properties:
            prop = copy(ax_prop)
            subplotspec = prop.pop('subplotspec')
            if prop['sharex'] is not None:
                prop['sharex'] = fig.get_axes()[prop['sharex']]
            if prop['sharey'] is not None:
                prop['sharey'] = fig.get_axes()[prop['sharey']]
            axes.append(fig.add_subplot(subplotspec, **prop))
            
        for ax,ax_call in zip(axes, self.axes_calls):
            for func, args, kwargs in ax_call:
                getattr(ax, func)(*args,**kwargs)
                if func == 'set_ylabel': ax.yaxis.set_label_coords(-0.12,kwargs['y'])
        return fig

        


def format_ax(ax, *args, **kwargs):
    """Format all relevant string on an axis using
    the string.format(*args, **kwargs) method.

    This will format the title, axis labels, any
    text instances, and the legend if present. 
    """
    if ax.legend_ is not None:
        for i in xrange(len(ax.legend_.texts)):
            ax.legend_.texts[i]._text = ax.legend_.texts[i]._text.format(*args, **kwargs)
    for i in xrange(len(ax.texts)):
        try:
            ax.texts[i]._text = ax.texts[i]._text.format(*args, **kwargs)
        except KeyError:
            continue
    ax.title._text = ax.title._text.format(*args, **kwargs)
    ax.xaxis.label._text = ax.xaxis.label._text.format(*args, **kwargs)
    ax.xaxis.label._text = ax.xaxis.label._text.format(*args, **kwargs)


def format_fig(fig, *args, **kwargs):
    for ax in fig.get_axes():
        format_ax(ax, *args, **kwargs)

        


# End of plot formatting.
################################################################################    

