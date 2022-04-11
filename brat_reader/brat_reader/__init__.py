import re
import os
import html
from collections import defaultdict
from pathlib import Path
import numpy as np


class Annotation(object):

    def __init__(self, _id, _source_file):
        self.id = _id
        self._source_file = _source_file
        if _source_file is not None:
            self._source_file = os.path.basename(_source_file)

    def update(self, key, value):
        self.__dict__[key] = value

    def __eq__(self, other):
        raise NotImplementedError()

    def __repr__(self):
        field_strings = []
        for (k, v) in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, Annotation):
                v_rep = v.short_repr()
            elif isinstance(v, dict):
                repr_dict = {}
                for (sub_k, sub_v) in v.items():
                    if isinstance(sub_v, Annotation):
                        sub_v_rep = sub_v.short_repr()
                    else:
                        sub_v_rep = repr(sub_v)
                    repr_dict[sub_k] = sub_v_rep
                v_rep = repr(repr_dict)
            else:
                v_rep = repr(v)
            field_strings.append(f"{k}: {v_rep}")
        fields_str = ', '.join(field_strings)
        class_name = str(self.__class__).split('.')[-1][:-2]
        rep = f"{class_name}({fields_str})"
        return rep

    def short_repr(self):
        class_name = str(self.__class__).split('.')[-1][:-2]
        return f"{class_name}(id: {self.id})"

    def copy(self):
        return self.__class__(**self.__dict__)

    @staticmethod
    def _resolve_file_path(path):
        try:
            here = Path(path).resolve()
            abspath = str(here.absolute())
        except TypeError:
            abspath = path
        return abspath

    def to_brat_str(self):
        raise NotImplementedError()


class Span(Annotation):
    def __init__(self, _id, _type, start_index, end_index,
                 text, _source_file=None):
        super().__init__(_id=_id, _source_file=_source_file)
        self.type = _type
        self.start_index = start_index
        self.end_index = end_index
        self.text = text

    def __eq__(self, other):
        if not isinstance(other, Span):
            return False
        return all([
            self.id == other.id,
            self.type == other.type,
            self.start_index == other.start_index,
            self.end_index == other.end_index,
            self.text == other.text,
        ])

    def to_brat_str(self, output_references=False):
        # output_references is unused but simplifies to_brat_str for Attribute
        return f"{self.id}\t{self.type} {self.start_index} {self.end_index}\t{self.text}"  # noqa


class Attribute(Annotation):
    def __init__(self, _id, _type, value, reference, _source_file=None):
        super().__init__(_id=_id, _source_file=_source_file)
        self.type = _type
        self.value = value
        self.reference = reference
        if not isinstance(self.reference, (Span, Event, type(None))):
            raise ValueError(f"Attribute reference must be instance of Span, Event, or None. Got {type(self.reference)}.")  # noqa

    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False
        return all([
            self.id == other.id,
            self.type == other.type,
            self.value == other.value,
            # Attributes and events can point to each
            # other, so we'll use IDs to avoid endless recursion.
            self.reference.id == other.reference.id,
        ])

    @property
    def start_index(self):
        if self.reference is None:
            idx = None
        elif isinstance(self.reference, Span):
            idx = self.reference.start_index
        elif isinstance(self.reference, Event):
            idx = self.reference.span.start_index
        else:
            raise ValueError(f"reference must be Span, Event, or None. Got {type(self.reference)}.")  # noqa
        return idx

    @property
    def end_index(self):
        if self.reference is None:
            idx = None
        elif isinstance(self.reference, Span):
            idx = self.reference.end_index
        elif isinstance(self.reference, Event):
            idx = self.reference.span.end_index
        else:
            raise ValueError(f"reference must be Span, Event, or None. Got {type(self.reference)}.")  # noqa
        return idx

    def to_brat_str(self, output_references=False):
        outlines = []
        if output_references is True:
            if self.reference is not None:
                ref_str = self.reference.to_brat_str(output_references=False)
                outlines.append(ref_str)
        ref_id = self.reference.id
        outlines.append(f"{self.id}\t{self.type} {ref_id} {self.value}")
        return '\n'.join(outlines)


class Event(Annotation):
    def __init__(self, _id, _type, span, attributes=None, _source_file=None):
        super().__init__(_id=_id, _source_file=_source_file)
        self.type = _type
        self.span = span
        self.attributes = attributes or {}
        for attr in self.attributes.values():
            attr.reference = self

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return all([
            self.id == other.id,
            self.type == other.type,
            self.span == other.span,
            self.attributes == other.attributes,
        ])

    @property
    def start_index(self):
        return self.span.start_index

    @property
    def end_index(self):
        return self.span.end_index

    def to_brat_str(self, output_references=False):
        event_str = f"{self.id}\t{self.type}:{self.span.id}"
        outlines = [event_str]
        if output_references is True:
            outlines.insert(0, self.span.to_brat_str())
            attr_strs = [a.to_brat_str(output_references=False)
                         for a in self.attributes.values()]
            outlines.extend(attr_strs)
        brat_str = '\n'.join(outlines)
        return brat_str


class BratAnnotations(object):

    @classmethod
    def from_file(cls, fpath):
        spans = []
        events = []
        attributes = []
        with open(fpath, 'r') as inF:
            for line in inF:
                line = line.strip()
                ann_type = line[0]
                if ann_type == 'T':
                    data = parse_brat_span(line)
                    data["_source_file"] = fpath
                    spans.append(data)
                elif ann_type == 'E':
                    data = parse_brat_event(line)
                    data["_source_file"] = fpath
                    events.append(data)
                elif ann_type == 'A':
                    data = parse_brat_attribute(line)
                    data["_source_file"] = fpath
                    attributes.append(data)
                else:
                    raise ValueError(f"Unsupported ann_type '{ann_type}'.")
        annotations = cls(spans=spans, events=events, attributes=attributes)
        return annotations

    @classmethod
    def from_events(cls, events_iter):
        annotations = cls(spans=[], events=[], attributes=[])
        annotations._events = list(events_iter)
        return annotations

    def __init__(self, spans, events, attributes):
        self._raw_spans = spans
        self._raw_events = events
        self._raw_attributes = attributes
        self._spans = []  # Will hold Span instances
        self._attributes = []  # Will hold Attribute instances
        self._events = []  # Will hold Event instances
        self._resolve()
        self._sorted_spans = None
        self._sorted_attributes = None
        self._sorted_events = None

    def __eq__(self, other):
        if not isinstance(other, BratAnnotations):
            return False
        if len(self.spans) != len(other.spans):
            return False
        for (this_span, other_span) in zip(self.spans, other.spans):
            if this_span != other_span:
                return False
        if len(self.attributes) != len(other.attributes):
            return False
        for (this_attr, other_attr) in zip(self.attributes, other.attributes):
            if this_attr != other_attr:
                return False
        if len(self.events) != len(other.events):
            return False
        for (this_event, other_event) in zip(self.events, other.events):
            if this_event != other_event:
                return False
        return True

    def get_events_by_type(self, event_type):
        return [e for e in self.events if e.type == event_type]

    def get_attributes_by_type(self, attr_type):
        return [a for a in self.attributes if a.type == attr_type]

    def get_spans_by_type(self, span_type):
        return [s for s in self.spans if s.type == span_type]

    @property
    def spans(self):
        if self._sorted_spans is None:
            self._sorted_spans = self._sort_spans_by_index()
        return self._sorted_spans

    @property
    def attributes(self):
        if self._sorted_attributes is None:
            self._sorted_attributes = self._sort_attributes_by_span_index()
        return self._sorted_attributes

    @property
    def events(self):
        if self._sorted_events is None:
            self._sorted_events = self._sort_events_by_span_index()
        return self._sorted_events

    def _sort_spans_by_index(self):
        return sorted(self._spans, key=lambda s: s.start_index)

    def _sort_attributes_by_span_index(self):
        """
        An Attribute may refer to a Span or an Event,
        so we have to check which is the case. If its an Event,
        we have to sort by the Event.span
        """
        span_indices = []
        for attr in self._attributes:
            if isinstance(attr.reference, Span):
                span_indices.append(attr.reference.start_index)
            elif isinstance(attr.reference, Event):
                span = attr.reference.span
                span_indices.append(span.start_index)
        sorted_indices = np.argsort(span_indices)
        return [self._attributes[i] for i in sorted_indices]

    def _sort_events_by_span_index(self):
        return sorted(self._events, key=lambda e: e.span.start_index)

    def _resolve(self):
        span_lookup = {}
        attribute_lookup = defaultdict(list)

        for raw_span in self._raw_spans:
            span = Span(**raw_span)
            span_lookup[raw_span["_id"]] = span
            self._spans.append(span)

        for raw_attr in self._raw_attributes:
            ref_id = raw_attr.pop("ref_id")
            ref = span_lookup.get(ref_id, None)
            attribute = Attribute(**raw_attr, reference=ref)
            attribute_lookup[ref_id].append(attribute)
            self._attributes.append(attribute)

        for raw_event in self._raw_events:
            ref_id = raw_event.pop("ref_span_id")
            span = span_lookup[ref_id]
            event = Event(**raw_event, span=span, attributes=None)
            attrs = attribute_lookup[raw_event["_id"]]
            for attr in attrs:
                attr.reference = event
            attrs_by_type = {attr.type: attr for attr in attrs}
            event.attributes = attrs_by_type
            self._events.append(event)

    def get_highest_level_annotations(self, type=None):
        if len(self._events) > 0:
            if type is not None:
                return self.get_events_by_type(type)
            else:
                return self.events
        elif len(self._attributes) > 0:
            if type is not None:
                return self.get_attributes_by_type(type)
            else:
                return self.attributes
        elif len(self._spans) > 0:
            if type is not None:
                return self.get_spans_by_type(type)
            else:
                return self.spans
        else:
            return []

    def __str__(self):
        outlines = []
        for ann in self.get_highest_level_annotations():
            outlines.append(ann.to_brat_str(output_references=True))
        return '\n'.join(outlines)

    def save_brat(self, outdir):
        for ann in self.get_highest_level_annotations():
            brat_str = ann.to_brat_str(output_references=True)
            bn = os.path.basename(ann._source_file)
            outfile = os.path.join(outdir, bn)
            with open(outfile, 'a') as outF:
                outF.write(brat_str + '\n')


def parse_brat_span(line):
    # Sometimes things like '&quot;' appear
    line = html.unescape(line)
    uid, label, other = line.split(maxsplit=2)
    # start1 end1;start2 end2
    if re.match(r'[0-9]+\s[0-9]+\s?;\s?[0-9]+\s[0-9]+', other):
        # Occasionally, non-contiguous spans occur in the n2c2 2022 data.
        # Merge these to be contiguous.
        text = ''
        spans = other.split(';', maxsplit=1)
        start_idx = None
        for span in spans:
            start_idx_tmp, end_idx_plus = span.split(maxsplit=1)
            if start_idx is None:
                start_idx = start_idx_tmp
            end_idx_split = end_idx_plus.split(maxsplit=1)
            if len(end_idx_split) > 1:
                end_idx, text = end_idx_split
            else:
                end_idx = end_idx_split[0]
    # start end
    else:
        start_idx, end_idx, text = other.split(maxsplit=2)

    return {"_id": uid,
            "_type": label,
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "text": text}


def parse_brat_event(line):
    fields = line.split()
    assert len(fields) == 2
    uid, label_and_ref = fields
    label, ref = label_and_ref.split(':')
    return {"_id": uid,
            "_type": label,
            "ref_span_id": ref}


def parse_brat_attribute(line):
    fields = line.split()
    if fields[1] == "Negation":
        if len(fields) == 3:
            fields.append("Negated")
    assert len(fields) == 4
    uid, label, ref, value = fields
    return {"_id": uid,
            "_type": label,
            "value": value,
            "ref_id": ref}
