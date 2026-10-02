"""Tokenization shared by indexing and querying."""
import re

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that
the their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you
your yours yourself yourselves unk
""".split())


def _stem(t: str) -> str:
    # Light suffix stripping (an "S-stemmer"): conservative enough not to
    # merge unrelated words, but folds the common plural forms.
    if len(t) > 4 and t.endswith("ies") and not t.endswith(("eies", "aies")):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("es") and not t.endswith(("aes", "ees", "oes")):
        return t[:-1]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("us", "ss")):
        return t[:-1]
    return t


def tokenize(text: str, max_tokens: int | None = None) -> list[str]:
    out = []
    for m in _TOKEN.finditer(text.lower()):
        t = m.group()
        if t in STOPWORDS or len(t) > 40:
            continue
        out.append(_stem(t))
        if max_tokens and len(out) >= max_tokens:
            break
    return out


def norm_title(title: str) -> str:
    return " ".join(tokenize(title))
