# quickplot

Rootpy based plotting for ATLAS publications.

## Quick Example

Download information from the spreadsheets to use to build plots. You need a samples, variables, and canvas spreadsheet. These get saved to json files.
```
python quickplot/sheets.py update example_samples example_variables example_canvas
```

Then use them to make a specific plot:
```
python ~/atlas-code/quickplot/quickplot.py mass_dedx.pdf --variable mass --canvas mass
```

Or a batch from a list of command line arguments in a file:
```
python ~/atlas-code/quickplot/quickplot.py --batch example.args 
```






