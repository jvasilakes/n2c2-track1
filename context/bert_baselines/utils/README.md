# Utilities

## Stochasitic Mask Viewer
`maskviewer.py`

`bert-rationale-classifier` predicts probabilistic masks over tokens in the input, which are used to modulate the embeddings sent to the classifier.

```
python run.py validate /path/to/logdir/config.yaml --output_token_masks
python utils/maskviewer.py /path/to/logdir/token_masks/{task}.jsonl
```
where `{task}` is one of the tasks specified under `tasks_to_load` in `config.yaml`, e.g. `Certainty.jsonl`.

Once `maskviewer.py` is running, navigate to `localhost:5000` in your browser to view the masks. Each token will be highlighted with
a color, ranging from light gray (0.0) to dark pink (1.0), according to the value of its corresponding mask. Hovering over a token with your cursor will bring up a tooltip with the actual value of the mask.
