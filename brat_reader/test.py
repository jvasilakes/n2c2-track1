import os
import brat_reader


def test_spans():
    if os.path.isfile("test_files/test_outputs/spans.ann"):
        os.remove("test_files/test_outputs/spans.ann")
    anns = brat_reader.BratAnnotations.from_file(
        "test_files/inputs/spans.ann")
    assert len(anns.spans) > 0
    assert len(anns.attributes) == 0
    assert len(anns.events) == 0

    anns.save_brat("test_files/test_outputs/")
    reread_anns = brat_reader.BratAnnotations.from_file(
        "test_files/test_outputs/spans.ann")
    assert len(reread_anns.spans) > 0
    assert len(reread_anns.attributes) == 0
    assert len(reread_anns.events) == 0

    gold_anns = brat_reader.BratAnnotations.from_file(
        "test_files/gold_outputs/spans.ann")
    assert gold_anns == reread_anns

    gold_str = open("test_files/gold_outputs/spans.ann").read()
    ann_str = open("test_files/test_outputs/spans.ann").read()
    assert gold_str == ann_str


def test_attributes():
    if os.path.isfile("test_files/test_outputs/attributes.ann"):
        os.remove("test_files/test_outputs/attributes.ann")
    anns = brat_reader.BratAnnotations.from_file(
        "test_files/inputs/attributes.ann")
    assert len(anns.spans) > 0
    assert len(anns.attributes) > 0
    assert len(anns.events) == 0

    anns.save_brat("test_files/test_outputs/")
    reread_anns = brat_reader.BratAnnotations.from_file(
        "test_files/test_outputs/attributes.ann")
    assert len(reread_anns.spans) > 0
    assert len(reread_anns.attributes) > 0
    assert len(reread_anns.events) == 0

    gold_anns = brat_reader.BratAnnotations.from_file(
        "test_files/gold_outputs/attributes.ann")
    assert gold_anns == reread_anns

    gold_str = open("test_files/gold_outputs/attributes.ann").read()
    ann_str = open("test_files/test_outputs/attributes.ann").read()
    assert gold_str == ann_str


def test_events():
    if os.path.isfile("test_files/test_outputs/events.ann"):
        os.remove("test_files/test_outputs/events.ann")
    anns = brat_reader.BratAnnotations.from_file(
        "test_files/inputs/events.ann")
    assert len(anns._raw_spans) > 0
    assert len(anns._raw_attributes) > 0
    assert len(anns._raw_events) > 0
    anns.save_brat("test_files/test_outputs/")

    reread_anns = brat_reader.BratAnnotations.from_file(
        "test_files/test_outputs/events.ann")
    assert len(reread_anns._raw_spans) > 0
    assert len(reread_anns._raw_attributes) > 0
    assert len(reread_anns._raw_events) > 0

    gold_anns = brat_reader.BratAnnotations.from_file(
        "test_files/gold_outputs/events.ann")
    assert gold_anns == reread_anns

    gold_str = open("test_files/gold_outputs/events.ann").read()
    ann_str = open("test_files/test_outputs/events.ann").read()
    assert gold_str == ann_str


def test_merge_events():
    if os.path.isfile("test_files/test_outputs/events_merge.ann"):
        os.remove("test_files/test_outputs/events_merge.ann")
    anns1 = brat_reader.BratAnnotations.from_file(
        "test_files/inputs/events_merge1.ann")
    anns2 = brat_reader.BratAnnotations.from_file(
        "test_files/inputs/events_merge2.ann")
    merged = anns1.merge(anns2)
    merged.save_brat("test_files/test_outputs/", filename="events_merge.ann")

    reread_anns = brat_reader.BratAnnotations.from_file(
            "test_files/test_outputs/events_merge.ann")
    gold_anns = brat_reader.BratAnnotations.from_file(
            "test_files/gold_outputs/events_merge.ann")
    assert gold_anns == reread_anns


if __name__ == "__main__":
    test_spans()
    test_attributes()
    test_events()
    test_merge_events()
    print("PASSED!")
