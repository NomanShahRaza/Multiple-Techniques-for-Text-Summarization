"""
Multiple Techniques for Text Summarization

Extractive (pick the most important sentences) and abstractive (generate new
sentences) summarizers behind a single interface, with a CLI to compare them.

@author: Noman Raza Shah
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

DEFAULT_SENTENCES = 3
SAMPLE_FILE = Path(__file__).with_name("sample_text.txt")


# ============================================================================ #
# Helpers                                                                      #
# ============================================================================ #

@lru_cache(maxsize=None)
def _ensure_nltk() -> None:
    import nltk

    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    return [s.strip() for s in sent_tokenize(normalize_whitespace(text)) if s.strip()]


@lru_cache(maxsize=None)
def _stopwords() -> frozenset[str]:
    _ensure_nltk()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _in_original_order(sentences: list[str], indices) -> str:
    return " ".join(sentences[i] for i in sorted(indices))


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================ #
# Extractive techniques                                                        #
# ============================================================================ #

def frequency_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """Score sentences by the normalized frequency of their content words."""
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    stop = _stopwords()
    freq: dict[str, int] = {}
    for word in _words(text):
        if word not in stop:
            freq[word] = freq.get(word, 0) + 1
    if not freq:
        return " ".join(sentences[:num_sentences])
    top = max(freq.values())

    scores = []
    for sentence in sentences:
        content = [w for w in _words(sentence) if w not in stop]
        score = sum(freq.get(w, 0) / top for w in content) / (len(content) or 1)
        scores.append(score)

    best = sorted(range(len(sentences)), key=scores.__getitem__, reverse=True)[:num_sentences]
    return _in_original_order(sentences, best)


def textrank_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """TextRank: PageRank over a TF-IDF cosine-similarity graph of sentences.

    Replaces `gensim.summarization`, which was removed in gensim 4.0.
    """
    import networkx as nx
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    similarity = cosine_similarity(tfidf)
    np.fill_diagonal(similarity, 0.0)

    scores = nx.pagerank(nx.from_numpy_array(similarity))
    best = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    return _in_original_order(sentences, best)


def _sumy_summary(summarizer_cls, text: str, num_sentences: int) -> str:
    _ensure_nltk()
    from sumy.nlp.stemmers import Stemmer
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.utils import get_stop_words

    parser = PlaintextParser.from_string(normalize_whitespace(text), Tokenizer("english"))
    summarizer = summarizer_cls(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")
    return " ".join(str(s) for s in summarizer(parser.document, num_sentences))


def lexrank_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """LexRank: graph centrality using eigenvector scores over sentence similarity."""
    from sumy.summarizers.lex_rank import LexRankSummarizer

    return _sumy_summary(LexRankSummarizer, text, num_sentences)


def lsa_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """LSA: SVD on the term-sentence matrix to find latent topics."""
    from sumy.summarizers.lsa import LsaSummarizer

    return _sumy_summary(LsaSummarizer, text, num_sentences)


def luhn_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """Luhn: clusters of significant (frequent, non-stop) words."""
    from sumy.summarizers.luhn import LuhnSummarizer

    return _sumy_summary(LuhnSummarizer, text, num_sentences)


def kl_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """KL-Sum: greedily minimize KL divergence between summary and document words."""
    from sumy.summarizers.kl import KLSummarizer

    return _sumy_summary(KLSummarizer, text, num_sentences)


def sumbasic_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """SumBasic: word-probability scoring with redundancy down-weighting."""
    from sumy.summarizers.sum_basic import SumBasicSummarizer

    return _sumy_summary(SumBasicSummarizer, text, num_sentences)


@lru_cache(maxsize=None)
def _sentence_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embedding_mmr_summary(
    text: str,
    num_sentences: int = DEFAULT_SENTENCES,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    diversity: float = 0.3,
) -> str:
    """Semantic extractive summary with Maximal Marginal Relevance.

    Sentences are embedded with a Sentence-Transformer; each pick balances
    relevance to the whole document against redundancy with earlier picks.
    """
    import numpy as np

    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    embeddings = _sentence_encoder(model_name).encode(sentences, normalize_embeddings=True)
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    relevance = embeddings @ centroid
    similarity = embeddings @ embeddings.T

    selected = [int(np.argmax(relevance))]
    while len(selected) < num_sentences:
        candidates = [i for i in range(len(sentences)) if i not in selected]
        redundancy = similarity[np.ix_(candidates, selected)].max(axis=1)
        mmr = (1 - diversity) * relevance[candidates] - diversity * redundancy
        selected.append(candidates[int(np.argmax(mmr))])
    return _in_original_order(sentences, selected)


@lru_cache(maxsize=None)
def _hf_encoder(model_name: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(_device())
    model.eval()
    return tokenizer, model


def cluster_summary(text: str, model_name: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """Embed sentences with a transformer, cluster them with k-means, and keep
    the sentence closest to each cluster centre (the BERT-extractive approach).
    """
    import numpy as np
    import torch
    from sklearn.cluster import KMeans

    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    tokenizer, model = _hf_encoder(model_name)
    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            # Second-to-last layer: less specialised to the pre-training objective than the last one.
            hidden = model(**inputs).hidden_states[-2][0]
            embeddings.append(hidden.mean(dim=0).cpu().numpy())
    embeddings = np.vstack(embeddings)

    kmeans = KMeans(n_clusters=num_sentences, n_init=10, random_state=42).fit(embeddings)
    chosen = set()
    for centre in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - centre, axis=1)
        for idx in np.argsort(distances):
            if idx not in chosen:
                chosen.add(int(idx))
                break
    return _in_original_order(sentences, chosen)


def bert_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """BERT-large embeddings + k-means clustering."""
    return cluster_summary(text, "bert-large-uncased", num_sentences)


def gpt2_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """GPT-2 (medium) embeddings + k-means clustering."""
    return cluster_summary(text, "gpt2-medium", num_sentences)


def xlnet_summary(text: str, num_sentences: int = DEFAULT_SENTENCES) -> str:
    """XLNet embeddings + k-means clustering."""
    return cluster_summary(text, "xlnet-base-cased", num_sentences)


# ============================================================================ #
# Abstractive techniques (Hugging Face transformers)                           #
# ============================================================================ #

@lru_cache(maxsize=None)
def _hf_model(model_name: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device())
    model.eval()
    return tokenizer, model


def _chunk_by_tokens(sentences: list[str], tokenizer, limit: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        n = len(tokenizer.tokenize(sentence))
        if current and current_len + n > limit:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def _generate(chunk: str, model_name: str, prefix: str, max_input: int,
              max_length: int, min_length: int, num_beams: int) -> str:
    import torch

    tokenizer, model = _hf_model(model_name)
    inputs = tokenizer(prefix + chunk, return_tensors="pt", truncation=True, max_length=max_input)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    summary = tokenizer.decode(output[0], skip_special_tokens=True).replace("<n>", " ")
    return re.sub(r"\s+([.,!?;:])", r"\1", normalize_whitespace(summary))


def hf_summary(
    text: str,
    model_name: str,
    prefix: str = "",
    max_input_tokens: int = 1024,
    max_length: int = 150,
    min_length: int = 30,
    num_beams: int = 4,
    _depth: int = 0,
) -> str:
    """Summarize with any Hugging Face seq2seq model (BART, T5, PEGASUS, LED, ...).

    Texts longer than the model's input window are split on sentence
    boundaries, each chunk is summarized, and the partial summaries are
    summarized again (map-reduce).
    """
    tokenizer, _ = _hf_model(model_name)
    max_input = min(max_input_tokens, tokenizer.model_max_length)
    limit = max_input - len(tokenizer.tokenize(prefix)) - 8

    partials = []
    for chunk in _chunk_by_tokens(split_sentences(text), tokenizer, limit):
        n_tokens = len(tokenizer.tokenize(chunk))
        out_max = max(min(max_length, n_tokens), 16)
        out_min = min(min_length, out_max // 2)
        partials.append(_generate(chunk, model_name, prefix, max_input, out_max, out_min, num_beams))

    combined = " ".join(partials)
    if len(partials) > 1 and _depth < 2:
        return hf_summary(combined, model_name, prefix, max_input_tokens,
                          max_length, min_length, num_beams, _depth + 1)
    return combined


def t5_summary(text: str, **kw) -> str:
    """T5-base, the original text-to-text transformer."""
    return hf_summary(text, "t5-base", prefix="summarize: ", max_input_tokens=512, **kw)


def t5_large_summary(text: str, **kw) -> str:
    """T5-large: better than T5-base but about 3x larger."""
    return hf_summary(text, "t5-large", prefix="summarize: ", max_input_tokens=512, **kw)


def flan_t5_summary(text: str, **kw) -> str:
    """Instruction-tuned T5; generally stronger than the original T5."""
    return hf_summary(text, "google/flan-t5-base", prefix="Summarize: ", max_input_tokens=512, **kw)


def bart_summary(text: str, **kw) -> str:
    """BART-large fine-tuned on CNN/DailyMail news."""
    return hf_summary(text, "facebook/bart-large-cnn", **kw)


def distilbart_summary(text: str, **kw) -> str:
    """Distilled BART: smaller and faster than BART with similar quality."""
    return hf_summary(text, "sshleifer/distilbart-cnn-12-6", **kw)


def pegasus_summary(text: str, **kw) -> str:
    """PEGASUS: pre-trained with gap-sentence generation, built for summarization."""
    return hf_summary(text, "google/pegasus-cnn_dailymail", **kw)


def led_summary(text: str, **kw) -> str:
    """Longformer Encoder-Decoder: reads up to 16k tokens without chunking."""
    return hf_summary(text, "pszemraj/led-base-book-summary", max_input_tokens=16384, **kw)


# ============================================================================ #
# Registry                                                                     #
# ============================================================================ #

@dataclass(frozen=True)
class Technique:
    name: str
    kind: str  # "extractive" | "abstractive"
    func: Callable[..., str]
    install: str
    uses_sentence_count: bool = True


TECHNIQUES: dict[str, Technique] = {
    t.name: t
    for t in [
        Technique("frequency", "extractive", frequency_summary, "nltk"),
        Technique("textrank", "extractive", textrank_summary, "nltk scikit-learn networkx"),
        Technique("lexrank", "extractive", lexrank_summary, "sumy"),
        Technique("lsa", "extractive", lsa_summary, "sumy"),
        Technique("luhn", "extractive", luhn_summary, "sumy"),
        Technique("kl", "extractive", kl_summary, "sumy"),
        Technique("sumbasic", "extractive", sumbasic_summary, "sumy"),
        Technique("mmr", "extractive", embedding_mmr_summary, "sentence-transformers"),
        Technique("bert", "extractive", bert_summary, "transformers torch scikit-learn"),
        Technique("gpt2", "extractive", gpt2_summary, "transformers torch scikit-learn"),
        Technique("xlnet", "extractive", xlnet_summary, "transformers torch scikit-learn sentencepiece"),
        Technique("t5", "abstractive", t5_summary, "transformers torch sentencepiece", False),
        Technique("t5-large", "abstractive", t5_large_summary, "transformers torch sentencepiece", False),
        Technique("flan-t5", "abstractive", flan_t5_summary, "transformers torch sentencepiece", False),
        Technique("bart", "abstractive", bart_summary, "transformers torch", False),
        Technique("distilbart", "abstractive", distilbart_summary, "transformers torch", False),
        Technique("pegasus", "abstractive", pegasus_summary, "transformers torch sentencepiece protobuf", False),
        Technique("led", "abstractive", led_summary, "transformers torch", False),
    ]
}

CLASSIC = ["frequency", "textrank", "lexrank", "lsa", "luhn", "kl", "sumbasic"]


def summarize(text: str, method: str = "textrank", num_sentences: int = DEFAULT_SENTENCES, **kwargs) -> str:
    """Summarize `text` with the technique registered under `method`."""
    technique = TECHNIQUES[method]
    if technique.uses_sentence_count:
        kwargs.setdefault("num_sentences", num_sentences)
    return technique.func(text, **kwargs)


# ============================================================================ #
# Evaluation                                                                   #
# ============================================================================ #

def rouge_scores(reference: str, summary: str) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return {k: v.fmeasure for k, v in scorer.score(reference, summary).items()}


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #

def _resolve_methods(spec: str) -> list[str]:
    groups = {
        "all": list(TECHNIQUES),
        "classic": CLASSIC,
        "extractive": [n for n, t in TECHNIQUES.items() if t.kind == "extractive"],
        "abstractive": [n for n, t in TECHNIQUES.items() if t.kind == "abstractive"],
    }
    methods: list[str] = []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        for name in groups.get(item, [item]):
            if name not in TECHNIQUES:
                raise SystemExit(f"Unknown method '{name}'. Use --list to see available techniques.")
            if name not in methods:
                methods.append(name)
    return methods


def _read_input(args) -> str:
    if args.text:
        return args.text
    if args.file == "-":
        return sys.stdin.read()
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return SAMPLE_FILE.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare multiple text summarization techniques.")
    parser.add_argument("-m", "--methods", default="classic",
                        help="Comma-separated techniques or groups: classic, extractive, "
                             "abstractive, all (default: classic)")
    parser.add_argument("-f", "--file", help="Read input text from a file ('-' for stdin)")
    parser.add_argument("-t", "--text", help="Input text passed directly")
    parser.add_argument("-n", "--sentences", type=int, default=DEFAULT_SENTENCES,
                        help="Number of sentences for extractive techniques")
    parser.add_argument("--max-length", type=int, default=150, help="Max tokens for abstractive output")
    parser.add_argument("--min-length", type=int, default=30, help="Min tokens for abstractive output")
    parser.add_argument("-r", "--reference", help="Reference summary file for ROUGE evaluation")
    parser.add_argument("--list", action="store_true", help="List available techniques and exit")
    args = parser.parse_args(argv)

    if args.list:
        for t in TECHNIQUES.values():
            doc = (t.func.__doc__ or "").strip().splitlines()[0] if t.func.__doc__ else ""
            print(f"{t.name:<11} {t.kind:<12} {doc}")
        return

    text = _read_input(args)
    reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else None
    original_words = len(text.split())
    print(f"Input: {original_words} words, {len(split_sentences(text))} sentences\n")

    for name in _resolve_methods(args.methods):
        technique = TECHNIQUES[name]
        kwargs = {}
        if technique.kind == "abstractive":
            kwargs = {"max_length": args.max_length, "min_length": args.min_length}

        print("=" * 78)
        print(f"{name.upper()} ({technique.kind})")
        print("-" * 78)
        start = time.perf_counter()
        try:
            summary = summarize(text, name, num_sentences=args.sentences, **kwargs)
        except ImportError as exc:
            reason = exc.name or str(exc).strip().splitlines()[0]
            print(f"Skipped: import failed ({reason}). Install with: pip install {technique.install}\n")
            continue
        except Exception as exc:  # keep comparing the remaining techniques
            print(f"Failed: {type(exc).__name__}: {exc}\n")
            continue
        elapsed = time.perf_counter() - start

        words = len(summary.split())
        print(summary)
        stats = f"[{elapsed:.2f}s | {words} words | {100 * words / max(original_words, 1):.0f}% of input"
        if reference:
            try:
                scores = rouge_scores(reference, summary)
                stats += " | " + " ".join(f"{k.upper()}={v:.3f}" for k, v in scores.items())
            except ImportError:
                stats += " | ROUGE needs: pip install rouge-score"
        print(stats + "]\n")


if __name__ == "__main__":
    main()
