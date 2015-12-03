#!/user/bin/env python

"""
A tool to retrieve histograms from ntuples using TTree::Draw
(and rootpy as the backend). It also styles, weights, and combines
histograms using information saved in configuration json.

Retrieval is cached, so any call to create a histogram that has
already been loaded will skip the ntuple reading step. This works
even between consecutive executions, but it means you have to delete
the cache.root file if you change the ntuple files (those are assumed static).
"""

import os
import re
import sys
import json
import glob
import types
import shutil
import atexit
import argparse
import rootpy.io.pickler as pickle

from helpers import *

from rootpy import asrootpy
from rootpy.tree import Tree, TreeChain
from itertools import izip, izip_longest
from seaborn.apionly import color_palette, husl_palette
from rootpy.plotting import Hist, Hist2D, Profile, Graph
from rootpy.io import root_open

cache_fname = "cache.root"
samples_json = 'samples.json'
variables_json = 'variables.json'

class ParameterError(Exception):
    pass

# We go through a lot of work here to handle arguments that might be single items
# or lists of items. But its necessary to deal with samples being one dataset
# or a sum of multiple datasets (e.g. Wplus and Wminus)

def is_listy(x):
    return any(isinstance(x, t) for t in [types.TupleType,types.ListType])


def make_iterable(x):
    if not is_listy(x):
        return [x]
    return x


def _join_selection(s1,s2):
    if s1 == '' or s1 is None:
        return s2
    if s2 == '' or s2 is None:
        return s1
    return '(' + s1 + ') && (' + s2 + ')'


def join_selections(s1, s2):
    ''' Join selections with logical and (or lists of selections)
    
    Correctly handles the case of either being none or empty
    '''
    if is_listy(s1) and not is_listy(s2):
        s2 = [s2 for s in s1]
    if not is_listy(s1) and is_listy(s2):
        s1 = [s1 for s in s2]
    if is_listy(s1) and is_listy(s2):
        if len(s1) != len(s2):
            print "Joining selection lists of different lengths will truncate to shorter list! Can cause a bug."
        return [_join_selection(a,b) for a,b in izip(s1,s2)]
    # Neither is a list, just join the two strings
    return _join_selection(s1,s2)



def join_labels(s1, s2):
    ''' Join labels with comma

    Correctly handles the case of either being none or empty
    '''
    if s1 == '' or s1 is None:
        return s2
    if s2 == '' or s2 is None:
        return s1
    return ', '.join((s1,s2))
    


def collect_files(path, extension=".root", exclude=".part"):
    files = []
    for p in glob.glob("*"+path+"*"):
        if os.path.isdir(p):
            for d, dnames, fnames in os.walk(p):
                files += [os.path.join(d,f) for f in fnames if ".root" in f]
        else:
            files += [p]

    files = [os.path.abspath(f) for f in files if ".part" not in f]
    files = [os.path.abspath(f) for f in files if ".part" not in f]
    eosmount = os.path.join(os.environ["HOME"],"eos")
    return [f.replace(eosmount, "root://eosatlas//eos") for f in files]


def load(hname, path):
    """
    Load a histogram from each root file in path and add them together
    """
    files = collect_files(path)
    hists = []
    for fname in files:
        with root_open(fname) as f:
            hists.append(f.Get(hname).Clone())
            hists[-1].SetDirectory(0)
    if len(hists) > 1:
        map(hists[0].Add, hists[1:])
    return hists[0]


def fill(hist, path, variable, selection="", options="", weight=1.0):
    """
    Fill hist with variable from the ntuple stored in all files found from expanding path.

    Attempts to handle files found on an eosmount, but you have to use ~/eos as the mount
    location. Also skips ".part" files used by rucio download.
    
    Selection is passed to TTree::Draw
    Weight is prepended to the selection string
    """
    if weight == "":
        weight = 1.0
    selection = str(weight) + "*(" + selection + ")" if selection else ""

    print variable, selection
    
    files = collect_files(path)
    # Test first file to find the tree name
    possible_trees = ['outputTree', 'ntupOutput', 'CollectionTree']
    with root_open(files[0]) as tmp:
        for tree in possible_trees:
            if tmp.find_key(tree):
                break
        else:
            raise IOError("Could not find a valid tree in the specified root files.")
    chain = TreeChain(tree, collect_files(path))
    chain.Draw(variable, selection=selection, options=options, hist=hist)
    return hist


def hashed(d):
    return tuple(sorted((a,tuple(b)) if is_listy(b) else (a,b) for a,b in d.iteritems()))


def save_cache():
    ''' Pickle the cache and close any open root files. '''
    with root_open(cache_fname+".tmp",'recreate') as output:
        pickle.dump(retrieve.cache, output)
    if retrieve.cache_file: 
        retrieve.cache_file.close()
    shutil.move(cache_fname+".tmp", cache_fname)


def _retrieve_hist(variable, subsamples, weights=1.0, efficiency=None):
    """
    Internal function to retrieve histograms directly from root files.
    
    Does weighted sums over subsamples.
    """
    subsamples = make_iterable(subsamples)
    weights = make_iterable(weights)

    # ----------------------------------------
    # Grab Histograms and Weight
    # ----------------------------------------
    hists = []
    for s,w in izip(subsamples, weights):
        hists.append(load(variable, s))
        hists[-1].Scale(w)
    if len(hists) > 1:
        map(hists[0].Add, hists[1:])
    hist = hists[0]

    # ----------------------------------------
    # Special Cases
    # ----------------------------------------
    if efficiency:
        nums = []
        for s,w in izip(subsamples, weights):
            nums.append(load(efficiency, s))
            nums[-1].Scale(w)
        if len(nums) > 1:
            map(nums[0].Add, nums[1:])
        hist = efficiency_divide(nums[0],hist)
    return hist


def _retrieve_ntuple(variable, bins, subsamples, selections='', weights=1.0, profile=False, efficiency=None):
    """
    Internal function to retrieve histogram from tree after variable information is expanded.
    
    This one performs the actual caching, which is only needed in the ntuple case.
    """
    # ------------------------------------------------------------
    # This needs to be done only once to get serious speed benefits
    # Load the cache and register saving it at program completion
    if "cache" not in retrieve.__dict__: 
        retrieve.cache_file = root_open(cache_fname) if os.path.exists(cache_fname) else None
        retrieve.cache = pickle.load(retrieve.cache_file) if retrieve.cache_file else {}
        atexit.register(save_cache)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Cached Lookup if possible
    # ------------------------------------------------------------
    context = locals()

    # Context contains all the spreadsheet information and arguments
    # If a histogram is cached with the same context, load it instead
    context_hash = hashed(context)
    if context_hash in retrieve.cache:
        return retrieve.cache[context_hash].Clone()
    
    # ------------------------------------------------------------
    # Create Histograms and Fill
    # ------------------------------------------------------------

    # Make the list argument expandable as *args later
    if isinstance(bins, types.ListType):
        bins = [bins]

    subsamples = make_iterable(subsamples)
    weights = make_iterable(weights)
    selections = make_iterable(selections)

    # Filling options based on variable type
    if profile:
        hists = [Profile(*bins) for s in subsamples]
        options = "prof"
    elif variable.count(":") == 1:
        hists = [Hist2D(*bins) for s in subsamples]
        options = ""
    else:
        hists = [Hist(*bins) for s in subsamples]
        options = ""

    # Fill for each sub sample
    for h,s,sel,weight in izip(hists, subsamples, selections, weights):
        fill(h, s, variable, selection=sel, options=options, weight=weight)


    # Combine if more than one sub sample, weights already applied
    if len(hists) > 1:
        map(hists[0].Add, hists[1:])
    hist = hists[0]

    # ----------------------------------------
    # Special Cases for Histograms
    # ----------------------------------------
    
    if efficiency:
        # Efficiency string specifies numerator selection if provided
        if variable.count(":") == 1:
            nums = [Hist2D(*bins) for s in subsamples]
        else:
            nums = [Hist(*bins) for s in subsamples]
        for n,s,sel,weight in izip(nums, subsamples, selections, weights):
            fill(n, s, variable, selection=join_selections(sel,efficiency), weight=weight)
        if len(nums) > 1:
            map(nums[0].Add, nums[1:])
        num = nums[0]
        hist = efficiency_divide(num,hist)

    if profile: # Convert profile to hist
        hist = asrootpy(hist.ProjectionX())

    # ------------------------------------------------------------
    # Cache and Return
    # ------------------------------------------------------------

    retrieve.cache[context_hash] = hist
    return hist



def retrieve(sample_args, variable_args, selection='', name=''):
    """
    Get a histogram filled with variable from sample, both stored in the configuration jsons.
    The histogram is filled, selected, styled, and weighted according to json information.

    Histograms are locally cached so that the slow ntuple reading is only performed
    as necessary.
    """

    # Combine argument dictionaries
    args = sample_args.copy()
    args.update(variable_args)
    
    # ------------------------------------------------------------
    # Join Arguments from Sample+Variable Args
    # ------------------------------------------------------------
    args['selection'] = join_selections(args['selection'], selection)

    bins = args["bins"]
    if bins: # Bins provided, means we are reading an ntuple
        if "(" in bins:
            bins = bins.strip("()").split(",")
            if len(bins) == 3:
                bins = int(bins[0]), float(bins[1]), float(bins[2])
            elif len(bins) == 6:
                bins = int(bins[0]), float(bins[1]), float(bins[2]), int(bins[3]), float(bins[4]), float(bins[5])
            else:
                raise ParameterError("Tuple bin argument must have three (1D) or six (2D) values.")
        elif "[" in bins:
            bins = [float(b) for b in bins.strip("[]").split(",")]
        else:
            raise ParameterError("Could not interpret bins provided by spreadsheet. They should have the form (nbins, low, high) or [edge1,edge2,edge3]")
        
        hist = _retrieve_ntuple(args['variable'], bins, args['subsample'], selections=args['selection'], weights=args['weight'], profile=args['profile'], efficiency=args['efficiency'])
    else:
        hist = _retrieve_hist(args['variable'], args['subsample'], weights=args['weight'], efficiency=args['efficiency'])

    # If any histogram member variable is a dictionary key,
    # set the histogram member to that value.
    for key in args:
        if hasattr(hist, key) and args[key] != '':
            setattr(hist, key, args[key])

    # Special cases for arguments
    hist.title = args['label']
    hist.cmap = args['cmap']
    
    return hist


def retrieve_all(variables, selection=""):
    hists = []
    for variable in variables:
        # Grab variable dictionary, some are lists over the samples
        if ":" in variable:
            variable, sample = variable.split(":")
        else:
            sample = None
        with open(variables_json) as f:
            variable_args = json.load(f)[variable]

        # Variable can look like "varname:sname" to specifiy one sample
        # or "varname" to loop over all subsamples
        if sample:
            indices = [variable_args['sample'].index(sample)]
            if not_empty(variable_args, 'reduction'):
                indices.append(indices[0] + 1)
        else:
            indices = range(len(make_iterable(variable_args['sample'])))

        # For each plottable (a combination of variable, sample) get the arguments
        for i in indices:
            tmp_args = variable_args.copy()
            # Flatten the arguments to use just the value at the current index
            for key,arg in tmp_args.iteritems():
                if is_listy(arg): tmp_args[key] = arg[i]
            sample = tmp_args['sample']
            with open(samples_json) as f:
                sample_args = json.load(f)[sample]
            name = "_".join((sample, variable))
            # ----------------------------------------
            # Special Cases
            # ----------------------------------------
            match = re.search('\[:(\d+)\]', tmp_args['variable'])
            if match:
                for i in xrange(int(match.group(1))):
                    tmp_args['variable'] = re.sub('\[(:?)\d+\]', '[{0}]'.format(i), tmp_args['variable'])
                    hists.append(retrieve(sample_args, tmp_args.copy(), selection))
    
            # ----------------------------------------
            # Default Case
            # ----------------------------------------
            else:
                hists.append(retrieve(sample_args, tmp_args, selection))

        if not_empty(variable_args,'reduction'):
            for i in xrange(-1*len(indices),0,2):
                if variable_args['reduction'] == 'sbp':
                    srootb = running_integral(hists[i])/sqrt_hist(running_integral(hists[i+1]))
                if variable_args['reduction'] == 'sbn':
                    srootb = running_integral(hists[i],neg=True)/sqrt_hist(running_integral(hists[i+1],neg=True))
                if variable_args['reduction'] == 'sbp' or variable_args['reduction'] == 'sbn':
                    seff = efficiency_divide(running_integral(hists[i]), flat_integral(hists[i]))
                    g = Graph(len(list(seff.y())))
                    for j, (x,xerr,y,yerr) in enumerate(izip(seff.y(), seff.yerr(), srootb.y(), srootb.yerr())):
                        g.set_point(j,1-x,y)
                        g.set_point_error(j,0,0,yerr[0],yerr[1])
                    g.decorate(hists[i])
                    g.title = hists[i].title
                    hists[i] = g
            del hists[-1*len(indices)+1::2]
    return hists

    
def main():
    parser = argparse.ArgumentParser("Saves ROOT histograms from ntuples using TTree::Draw style syntax.")
    parser.add_argument('output', help='Name of the plot to save.')
    parser.add_argument('--samples', nargs='+', help='The names of samples to add to the plot. Each sample gets a separate graphic.')
    parser.add_argument('--variables', nargs='+', help='The names of variables to add to the plot. Each variable gets a separate graphic.')
    parser.add_argument('--selection', help='Additional global selection to apply for all loaded data.')

    args = parser.parse_args()

    hists = retrieve_all(args.samples, args.variables, args.selection)
    with root_open(args.output, "recreate") as f:
        for h in hists:
            h.Write(h.name)
            
    

if __name__ == "__main__":
    main()


