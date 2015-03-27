#!/usr/bin/env python
"""
Configures matplolib's rc params and defines a color cycle.

For this to work correctly, you need to have a working xelatex installed.
"""

import matplotlib
import __main__ as main
from subprocess import Popen, PIPE


def_grey = '#737373'
neutral  = ['#737373', '#FF3F5D', '#4999D8', '#5BCB61', '#FFA44E', '#A95AAE', '#DC6753', '#E670B6']
bold     = ['#020202', '#FF0025', '#0253AD', '#009441', '#FF7300', '#710095', '#B0011B', '#C50097']
light    = ['#CCCCCC', '#FDA9AC', '#B3D1EE', '#D5E8A5', '#F9D1AD', '#DDACD6', '#E3B7A7', '#F4BADB']


custom_preamble = {
    'axes.color_cycle'   : neutral,
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
    'lines.linewidth'    : 2.0,
    'lines.markersize'   : 8.0,
    'legend.numpoints'   : 1
}


if hasattr(main, '__file__') and Popen(['which','xelatex'],stdout=PIPE).wait() == 0:
    # Not interactive: use full latex for better style
    matplotlib.use("pgf")
    custom_preamble.update({
        'font.family'        : 'serif', # actually uses latex main font from below
        'pgf.texsystem'     : 'xelatex', 
        'pgf.rcfonts'       : False,   # don't setup fonts from rc parameters
        'pgf.preamble'      : [
            r'\usepackage{mathspec}'
            r'\setallmainfonts(Digits,Latin,Greek){Lato Light}'
            r'\setallmonofonts[Scale=MatchLowercase]{Monaco}'
        ]
    })


matplotlib.rcParams.update(custom_preamble)


