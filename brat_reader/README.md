`brat_reader` is a simple package for reading brat-formatted text annotations (https://brat.nlplab.org/).

# Installation
`brat_reader` has no external dependencies. Install it with

```
python setup.py develop
```

Using `develop` should update your installed version when you pull changes from the github.

## Uninstallation

```
python setup.py develop --uninstall
```

# Usage
Given an .ann file, parse the annotations with 

```python
from brat_reader import BratAnnotations
anns = BratAnnotations.from_file("/path/to/file.ann")
```

`BratAnnotations` automatically links events to their associated text spans and attributes. `Event` instances, sorted by trigger span indices, can be accessed with `BratAnnotations.events`.

`Span`s, `Attribute`s, and `Event`s are `collections.namedtuple` instances with the following structures:

## `Span`
* `id : str`
* `start_index : int`
* `end_index : int`
* `text : str`

## `Attribute`
* `id : str`
* `type : str`
* `value : {str,int}`

## `Event`
* `id : str`
* `type : str`
* `span : Span`
* `attributes : dict({str: Attribute})`
