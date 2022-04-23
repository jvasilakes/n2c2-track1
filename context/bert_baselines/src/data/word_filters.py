import spacy
from spacy.tokens import Doc
import scispacy  # noqa F401




class WordFilter(object):
    def __init__(self, spacy_model_str, bert_tokenizer):
        self.nlp = spacy.load(spacy_model_str)
        self.bert_tokenizer = bert_tokenizer
        self._cache = {}

    def __call__(self, token_spans, input_ids):
        seq_len = (input_ids != 0).sum()
        input_repr = str(token_spans) + "||" + str(input_ids[:seq_len].tolist())  # noqa
        try:
            keep_spans = self._cache[input_repr]
        except KeyError:
            tokens = [self.bert_tokenizer.decode(input_ids[span])
                      for span in token_spans]
            doc = Doc(self.nlp.vocab, tokens)
            processed_tokens = self.nlp(doc)
            keep_spans = []
            for (span, processed_tok) in zip(token_spans, processed_tokens):
                if self.keep(processed_tok) is True:
                    keep_spans.append(span)
            self._cache[input_repr] = keep_spans
        return keep_spans

    def keep(self, spacy_token):
        """
        True if we should keep this token, else False.
        Override this in child classes.
        """
        raise NotImplementedError()


class POSFilter(WordFilter):
    """
    Filter tokens according to POS tags.
    I.e., spacy_token.pos_
    Uses these tags:
      ADJ: adjective
      ADP: adposition
      ADV: adverb
      AUX: auxiliary
      CCONJ: coordinating conjunction
      DET: determiner
      INTJ: interjection
      NOUN: noun
      NUM: numeral
      PART: particle
      PRON: pronoun
      PROPN: proper noun
      PUNCT: punctuation
      SCONJ: subordinating conjunction
      SYM: symbol
      VERB: verb
      X: other
    """
    def __init__(self, spacy_model_str, bert_tokenizer, keep_tags=None):
        super().__init__(spacy_model_str, bert_tokenizer)
        if keep_tags is None:
            warnings.warn(f"No keep_tags specified to POSFilter. Keeping everything.")  # noqa
        self.keep_tags = keep_tags

    def keep(self, spacy_token):
        if self.keep_tags is None:
            return True
        if spacy_token.pos_ in self.keep_tags:
            return True
        return False

    def __str__(self):
        return f"POSFilter(keep_tags={self.keep_tags})"
