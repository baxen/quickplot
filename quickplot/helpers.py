from rootpy.plotting import Hist, Graph
from math import sqrt
from itertools import izip

def not_empty(d, key):
    if key not in d:
        return False
    if d[key] == '':
        return False
    if d[key] == None:
        return False
    return True

def is_true(d, key):
    if key not in d:
        return False
    return d[key]


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

def xscale(hist, scale):
    bins = list(hist.xedges())
    h = Hist([b*scale for b in bins])
    for b1,b2 in izip(h.bins(),hist.bins()):
        b1.value = b2.value
        b1.error = b2.error
    h.decorate(hist)
    h.title = hist.title
    h.drawstyle = hist.drawstyle
    return h

def efficiency_divide(h1, h2):
    ''' Efficiency with correct errors from numerator and denominator hists '''
    eff = Graph()
    eff.Divide(h1,h2,"cl=0.683 b(1,1) mode")
    eff.decorate(h1)
    eff.title = h1.title
    return eff

def sqrt_hist(hist):
    hist = hist.Clone()
    for i in hist.bins_range():
        rel_unc = hist.GetBinError(i)/hist.GetBinContent(i)
        hist.SetBinContent(i,sqrt(hist.GetBinContent(i)))
        hist.SetBinError(i, hist.GetBinContent(i) * rel_unc * .05)
    return hist
        
def running_integral(hist, neg=False):
    result = histify(hist.Clone())
    for x in result.bins_range(overflow=True):
        if neg:
            # Integral up to the value
            y,yerr = hist.integral(xbin2=x, error=True, overflow=True)
        else:
            # Integral starting at the value
            y,yerr = hist.integral(xbin1=x, error=True, overflow=True) 
        result.SetBinContent(x,y)
        result.SetBinError(x,yerr)
    return result

def flat_hist(hist, index):
    val, err = hist.GetBinContent(abs(index)), hist.GetBinError(abs(index))
    result = hist.Clone()
    for x in result.bins_range(overflow=True):
        result.SetBinContent(x, val)
        result.SetBinError(x, err)
    return result

def flat_integral(hist):
    val, err = hist.integral(overflow=True, error=True)
    result = hist.Clone()
    for x in result.bins_range(overflow=True):
        result.SetBinContent(x, val)
        result.SetBinError(x, err)
    return result
