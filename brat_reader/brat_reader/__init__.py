from collections import namedtuple, defaultdict


class Annotation(object):

    def __init__(self):
        pass

    def update(self, key, value):
        self.__dict__[key] = value

    def __repr__(self):
        class_name = str(self.__class__).split('.')[-1][:-2]
        rep = f"{class_name}({str(self.__dict__)})"
        return rep

    def copy(self):
        return self.__class__(**self.__dict__)


class Span(Annotation):
    def __init__(self, id, start_index, end_index, text):
        self.id = id
        self.start_index = start_index
        self.end_index = end_index
        self.text = text


class Attribute(Annotation):
    def __init__(self, id, type, value):
        self.id = id
        self.type = type
        self.value = value


class Event(Annotation):
    def __init__(self, id, type, span, attributes=None):
        self.id = id
        self.type = type
        self.span = span
        self.attributes = attributes or {}


class BratAnnotations(object):

    # Span = namedtuple("Span", ["id", "start_index", "end_index", "text"])
    # Attribute = namedtuple("Attribute", ["id", "type", "value"])
    # Event = namedtuple("Event", ["id", "type", "span", "attributes"])

    def __init__(self, spans, events, attributes):
        self._raw_spans = spans
        self._raw_events = events
        self._raw_attributes = attributes
        self._events = []  # Will hold Event instances
        self._resolve()
        self._sorted_events = None

    def get_events_by_type(self, event_type):
        return [e for e in self.events if e.type == event_type]

    def attributes_by_type(self, attr_type):
        raise NotImplementedError()

    @property
    def events(self):
        if self._sorted_events is None:
            self._sorted_events = self._sort_events_by_span_index()
        return self._sorted_events

    def _sort_events_by_span_index(self):
        return sorted(self._events, key=lambda e: e.span.start_index)

    def _resolve(self):
        if self._events != []:
            raise ValueError("Events have already by populated!")
        span_lookup = {}
        attribute_lookup = defaultdict(list)

        for raw_span in self._raw_spans:
            span_lookup[raw_span["id"]] = Span(**raw_span)

        for raw_attr in self._raw_attributes:
            ref_id = raw_attr["ref_event_id"]
            raw_attr.pop("ref_event_id")  # Won't use in the Attribute tuple
            attribute_lookup[ref_id].append(Attribute(**raw_attr))

        for raw_event in self._raw_events:
            ref_id = raw_event["ref_span_id"]
            raw_event.pop("ref_span_id")  # Won't use it in the Event tuple
            span = span_lookup[ref_id]
            attrs = attribute_lookup[raw_event["id"]]
            attrs_by_type = {attr.type: attr for attr in attrs}
            event = Event(
                    **raw_event, span=span, attributes=attrs_by_type)
            self._events.append(event)

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
                    spans.append(data)
                elif ann_type == 'E':
                    data = parse_brat_event(line)
                    events.append(data)
                elif ann_type == 'A':
                    data = parse_brat_attribute(line)
                    attributes.append(data)
                else:
                    raise ValueError(f"Unsupported ann_type '{ann_type}'.")
        annotations = cls(spans=spans, events=events, attributes=attributes)
        return annotations

    @classmethod
    def from_events(cls, events_iter):
        spans = []
        attributes = []
        for event in events_iter:
            spans.append(event.span)
            for attr in event.attributes.values():
                attributes.append(attr)
        annotations = cls(spans=spans,
                          events=events_iter,
                          attributes=attributes)
        return annotations


def parse_brat_span(line):
    uid, label, other = line.split(maxsplit=2)
    if ';' not in other:
        start_idx, end_idx, text = other.split(maxsplit=2)
    else:
        # Occasionally, non-contiguous spans occur in the training data.
        # Merge these to be contiguous.
        text = ''
        spans = other.split(';')
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

    return {"id": uid,
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "text": text}


def parse_brat_event(line):
    fields = line.split()
    assert len(fields) == 2
    uid, label_and_ref = fields
    label, ref = label_and_ref.split(':')
    return {"id": uid,
            "type": label,
            "ref_span_id": ref}


def parse_brat_attribute(line):
    fields = line.split()
    if fields[1] == "Negation":
        fields.append(True)
    assert len(fields) == 4
    uid, label, ref, value = fields
    return {"id": uid,
            "type": label,
            "value": value,
            "ref_event_id": ref}
