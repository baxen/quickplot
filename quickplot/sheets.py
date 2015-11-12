#!/user/bin/env python

"""
A tool to read information from a google spreadsheet, 
generate some types of missing content, and save to local format.
"""

import os
import json
import glob
import types
import argparse

from itertools import izip
from seaborn.apionly import color_palette, husl_palette

credentials_json = os.path.join(os.environ['HOME'],'.gspread.json')

def is_listy(x):
    return any(isinstance(x, t) for t in [types.TupleType,types.ListType])

def hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*[int(rgb[i]*256) if rgb[i] < 1.0 else 255 for i in xrange(3)])

def husl_hex(n, **kwargs):
    return [hex(c) for c in husl_palette(n, **kwargs)]

def color_hex(name, n, **kwargs):
    return [hex(c) for c in color_palette(name, n, **kwargs)]


def first_min(l, values):
    counts = [values.count(x) for x in l]
    lowest = [x for i,x in enumerate(l) if counts[i] == min(counts)]
    return lowest[0]

def default_value(key, values):
    if key == "markerstyle":
        return first_min(['o','d','^','v','<','>'], values)
    elif key == "markersize":
        return 1.2
    elif key == "color":
        return first_min(husl_hex(8,h=.5,l=.55), values)
    elif key == "drawstyle":
        return "errorbar"
    else:
        return ''

def load_sheet(sheet_name):
    import gspread
    from oauth2client.client import SignedJwtAssertionCredentials

    json_key = json.load(open(credentials_json))
    scope = ['https://spreadsheets.google.com/feeds']

    credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'], scope)
    gc = gspread.authorize(credentials)
    return gc.open(sheet_name).sheet1
    
def save_sheet(sheet, name):
    # Save to json, keeping one entry per row, with lists of subrow info
    content = sheet.get_all_records()
    keyname = sheet.cell(1,1).value
    context = [(row.pop(keyname),row) for row in content]
    for i,(s,d) in enumerate(context):
        if s: 
            for key in d:
                d[key] = [d[key]]
            si = i
        else:
            for key in d:
                if d[key]: context[si][1][key] += [d[key]]
    context = dict(x for x in context if x[0])
        
    # If the list has only one entry, convert it to just that value
    for d in context.values():
        for key in d:
            if len(d[key]) == 1:
                d[key] = d[key][0]
    with open(name,'w') as f:
        json.dump(context,f)
    

def update(args):
    """
    Updates spreadsheet, filling blank spaces where possible and downloads to the local json.
    """
    for sheet in args.sheet:
        ss = load_sheet(sheet)
        content = ss.get_all_values()
        
        # Generate missing content where possible
        header = content[0]
        for i,row in enumerate(content[1:]):
            # Only generate content for the sample, not subsamples
            if row[0] == "": continue
            for j,val in enumerate(row):
                if val == '':
                    # indexing starts at one, first row has labels
                    updated = default_value(header[j], [content[x][j] for x in xrange(1,len(content))])
                    if updated:
                        content[i+1][j] = updated
                        ss.update_cell(i+2, j+1, updated)
        save_sheet(ss, sheet.split("_")[-1]+".json")


def color(args):
    sheet = load_sheet(args.sheet)
    cindex = sheet.find("color").col
    slist = sheet.col_values(1)
    colors = color_hex(args.color, len(args.samples))
    for color, sample in izip(colors, args.samples):
        sheet.update_cell(slist.index(sample)+1, cindex, color)
    save_sheet(sheet, args.sheet.split("_")[-1]+".json")

def main():
    parser = argparse.ArgumentParser("Reads a google spreadsheet, updates content where possible, and saves a copy to local json.")
    subparsers = parser.add_subparsers(help='')
    
    parser_update = subparsers.add_parser('update', help='Update the spreadsheet and save local copy.')
    parser_update.set_defaults(func=update)
    parser_update.add_argument("sheet", nargs="+", help="Name of the spreadsheet to update.")

    parser_color = subparsers.add_parser('color', help='Update a subset of colors using specified options and save local.')
    parser_color.add_argument("sheet", help="Name of the spreadsheet to update.")
    parser_color.add_argument('color', help="Name of color palette to use (from seaborn color_palette).")
    parser_color.add_argument('samples', nargs="+", help="The samples to apply the colors to, in order.")
    parser_color.set_defaults(func=color)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()


