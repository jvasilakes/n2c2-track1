import os
import json
import argparse
from xml.dom.minidom import getDOMImplementation
from matplotlib import cm
from matplotlib import colors
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="JSON lines file containing token masks")
    parser.add_argument("outfile", type=str,
                        help="Where to save the html file.")
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--collapse_wordpiece", action="store_true",
                        default=False)
    return parser.parse_args()


def getDOM():
    impl = getDOMImplementation()
    dt = impl.createDocumentType(
        "html",
        "-//W3C//DTD XHTML 1.0 Strict//EN",
        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd",
    )
    return impl.createDocument("http://www.w3.org/1999/xhtml", "html", dt)


def create_page(infile, outfile, max_examples, collapse_wordpiece=False):
    dom = getDOM()
    html = dom.documentElement
    h = dom.createElement("h1")
    h.appendChild(dom.createTextNode("Stochastic Mask Viewer"))
    html.appendChild(h)
    bn = os.path.basename(infile)
    h3 = dom.createElement("h3")
    h3.appendChild(dom.createTextNode(f"Viewing file: {bn}"))
    html.appendChild(h3)

    for example_data in get_json_data(
            infile, dom, max_examples, collapse_wordpiece):
        docid = example_data["docid"]
        gold = example_data["label"]
        pred = example_data["prediction"]

        # A little signal marking correct from incorrect predictions.
        sign = dom.createElement("span")
        sign.setAttribute("class", "highlight")
        if gold != pred:
            sign.setAttribute("style", "background-color:#ff0000")
            txt = "X"
        else:
            sign.setAttribute("style", "background-color:#00ff00")
            txt = "✓"
        sign.appendChild(dom.createTextNode(' ' + txt + ' '))
        html.appendChild(sign)

        b = dom.createElement("b")
        b.appendChild(dom.createTextNode(
            f"ID: {docid} | Gold: '{gold}' | Pred: '{pred}'"))
        html.appendChild(b)
        p = dom.createElement("p")
        highlighted_tokens = example_data["highlighted_text"]
        for ht in highlighted_tokens:
            p.appendChild(dom.createTextNode(' '))
            p.appendChild(ht)
            p.appendChild(dom.createTextNode(' '))
        html.appendChild(p)

    with open(outfile, 'w') as outF:
        outF.write(dom.toxml())


def get_json_data(jsonl_file, dom, max_examples=-1, collapse_wordpiece=False):
    data = [json.loads(line.strip()) for line in open(jsonl_file)]
    for (i, datum) in enumerate(data):
        if i == max_examples:
            break
        datum["highlighted_text"] = get_highlighted_text(
                datum["tokens_with_masks"], dom, collapse_wordpiece)
        yield datum


def get_highlighted_text(tokens_with_masks, dom, collapse=False):
    spans = []
    if collapse is True:
        tokens_with_masks = collapse_wordpiece(tokens_with_masks)
    for (tok, z) in tokens_with_masks:
        color = z2color(z)
        span = dom.createElement("span")
        span.setAttribute("class", "highlight")
        span.setAttribute("style", f"background-color:{color}")
        span.setAttribute("title", f"{z:.3f}")
        span.appendChild(dom.createTextNode(f"{tok}"))
        spans.append(span)
    return spans


def z2color(z):
    # Cap the colormap to make the highlighting more readable.
    norm = colors.Normalize(vmin=0, vmax=1.5)
    return colors.rgb2hex(cm.PuRd(norm(z)))


def collapse_wordpiece(tokens_with_masks):
    output = []
    current_tok = ''
    current_zs = []
    for (tok, z) in tokens_with_masks:
        if tok.startswith("##"):
            current_tok += tok.lstrip("##")
            current_zs.append(z)
        else:
            if len(current_zs) > 0:
                output.append((current_tok, np.mean(current_zs)))
                current_tok = ''
                current_zs = []
            current_tok = tok
            current_zs.append(z)
    output.append((current_tok, np.mean(current_zs)))
    return output


if __name__ == "__main__":
    args = parse_args()
    create_page(
            args.infile, args.outfile,
            args.max_examples, args.collapse_wordpiece)
