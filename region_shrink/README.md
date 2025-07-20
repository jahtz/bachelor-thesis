# regionshrink

Shrinks the region polygons of PageXML files to its content.

## Usage

```txt
$ regionshrink --help
Usage: regionshrink [OPTIONS] FILES...

  Shrinks the region polygons of PageXML files.

  FILES: List of PageXML file paths to process. Accepts individual files or
  glob wildcards.

Options:
  --help                          Show this message and exit.
  --version                       Show the version and exit.
  -i, --images TEXT               Suffix of the image files to search for. If
                                  not provided, the imageFilename attribute is
                                  used.
  -o, --output DIRECTORY          Specify output directory for processed
                                  files. If not set, overwrite input files.
  -p, --padding INTEGER           Padding between region borders and its
                                  content in pixels.  [default: 5]
  -s, --smoothing FLOAT           Smoothing, calculated as the factor of the
                                  average glyph size. Prevents regions eating
                                  between text.  [default: 1.0]
  -m, --mode [merge|largest]      Shrinking mode to use for regions. 'merge'
                                  merges all resulting polygons of each region
                                  after shrinking. 'largest' keeps only the
                                  largest resulting polygon of each region
                                  after shrinking.  [default: merge]
  --logging [ERROR|WARNING|INFO|DEBUG]
                                  Set logging level.  [default: ERROR]
```
