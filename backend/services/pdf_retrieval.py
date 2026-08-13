"""
Lightweight chunk retrieval for the uploaded-PDF Q&A feature: TF-IDF +
cosine similarity, not a vector/embedding search - deliberately, per the
project's own scope for a first version (no new vector database, no
Gemini embedding calls just to answer from a handful of PDF pages).

Two functions are the entire interface other modules use:
build_index() at upload time, retrieve() at question time. Swapping this
for real embeddings later (e.g. reusing GEMINI_EMBEDDING_MODEL the way
the project's since-removed FAISS pipeline did) only means changing what
build_index()/retrieve() do internally - callers (routes/documents.py,
routes/chat.py) never see a TF-IDF vector or a similarity score, just
"the chunks", so nothing downstream needs to change.
"""

import os

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = int(os.getenv("PDF_RETRIEVAL_TOP_K", "3"))

# scikit-learn's built-in "english" stop-word list drops a handful of
# words that are exactly the vocabulary financial documents rely on -
# "amount" and "due" are both in it. Left in, a chunk that says "...the
# budgeted amount is Rs. 400000..." scores a similarity of 0.0 against a
# question like "What is the expected payment amount?", because the one
# word they share gets stripped from both sides before either is
# vectorized. Carved out here rather than dropping stop-word filtering
# entirely - "the"/"is"/"are" etc. still need to be filtered, or two
# completely unrelated questions that happen to share only those words
# (e.g. "What are the available income categories?" against a document
# that never mentions income) would show a false-positive nonzero score.
_STOP_WORDS = list(ENGLISH_STOP_WORDS - {"amount", "due"})


def build_index(chunks):
    """Fit a TF-IDF index over one document's chunks. Returns
    (vectorizer, matrix) - store both alongside the chunks themselves
    (see document_store.create_document); retrieve() needs all three."""

    vectorizer = TfidfVectorizer(stop_words=_STOP_WORDS)
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def _scores(vectorizer, matrix, question):
    query_vector = vectorizer.transform([question])
    return cosine_similarity(query_vector, matrix)[0]


def retrieve(chunks, vectorizer, matrix, question, top_k=None):
    """Return the top_k chunks most relevant to question, highest first.

    Falls back to the first top_k chunks (in document order) if nothing
    scores above zero - a short or unusual question can legitimately
    share no vocabulary with any chunk under plain TF-IDF, and handing
    Gemini *something* to ground an answer in beats handing it nothing,
    while still being clearly labeled as PDF context Gemini itself must
    judge as relevant or not (see routes/chat.py's PDF/hybrid prompts).
    """

    top_k = top_k or TOP_K

    if not chunks:
        return []

    scores = _scores(vectorizer, matrix, question)

    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    top_indices = [i for i in ranked_indices if scores[i] > 0][:top_k]

    if not top_indices:
        top_indices = list(range(min(top_k, len(chunks))))

    return [chunks[i] for i in top_indices]


def top_score(chunks, vectorizer, matrix, question):
    """Return the single highest chunk-similarity score for question, or
    0.0 if there are no chunks. Used by routes/chat.py as a signal that a
    question is actually about the attached PDF even when it doesn't say
    "document"/"pdf" outright (see intent_router.classify_intent's
    pdf_relevant parameter) - a real content-overlap check rather than
    another fixed phrase list, so it generalizes to whatever the PDF
    actually contains instead of requiring the user to say "document"."""

    if not chunks:
        return 0.0

    return float(_scores(vectorizer, matrix, question).max())
