"""Section-aware passage splitting using the MiniLM WordPiece tokenizer."""
import re
from functools import lru_cache
from pathlib import Path

MAX_TOKENS = 220  # Includes special tokens; Chroma's MiniLM truncates at 256.
OVERLAP_TOKENS = 40
CHUNKING_VERSION = 'sections-sentences-v1'


@lru_cache(maxsize=1)
def get_tokenizer():
    from legal_assistant.embeddings import counting_tokenizer
    return counting_tokenizer()


def token_count(text):
    return len(get_tokenizer().encode(text).ids)


def sections(text):
    """Recognize explicit headings; otherwise retain the original section/paragraphs."""
    heading = ''
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        marked = bool(re.match(r'^#{1,6}\s+\S', stripped))
        numbered = bool(re.match(r'^\d+(?:\.\d+)*[.)]?\s+\S', stripped))
        upper = stripped.isupper() and len(stripped.split()) <= 10
        short = len(stripped.split()) <= 14 and not re.search(r'[.!?;:]$', stripped)
        is_heading = marked or (short and (numbered or upper))
        if is_heading:
            if lines and any(x.strip() for x in lines):
                yield heading, '\n'.join(lines).strip()
            heading = re.sub(r'^#{1,6}\s+', '', stripped)
            lines = []
        else:
            lines.append(line)
    if any(x.strip() for x in lines):
        yield heading, '\n'.join(lines).strip()
    elif heading:
        yield heading, ''


def sentence_units(text):
    # Conservative English heuristic, not a legal parser. Avoid common abbreviations.
    protected = re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|No|Art|Sec)\.',
                       lambda m: m[0].replace('.', '\u2024'), text)
    protected = protected.replace('e.g.', 'e\u2024g\u2024').replace('i.e.', 'i\u2024e\u2024')
    return [part.replace('\u2024', '.').strip() for part in
            re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])|\n\s*\n', protected) if part.strip()]


def split_long_unit(text, prefix, max_tokens):
    """Word-boundary fallback for an oversized sentence, preserving original text."""
    remaining = text.strip()
    while remaining:
        if token_count(prefix + remaining) <= max_tokens:
            yield remaining
            break
        offsets = get_tokenizer().encode(remaining, add_special_tokens=False).offsets
        # Reserve prefix and model special tokens; verify the exact joined text below.
        budget = max_tokens - token_count(prefix)
        end = offsets[max(0, min(budget, len(offsets)) - 1)][1]
        boundary = remaining.rfind(' ', 0, end + 1)
        if boundary > 0:
            end = boundary
        while end > 0 and token_count(prefix + remaining[:end]) > max_tokens:
            end -= 1
        if not end:
            raise ValueError('Chunk heading leaves no room for content.')
        yield remaining[:end].strip()
        remaining = remaining[end:].lstrip()


def chunk_text(text, source_type, section_title='', max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    """Yield text, section number/title and fallback flag; do not merge across sections."""
    if max_tokens < 16 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError('Use max_tokens >= 16 and 0 <= overlap_tokens < max_tokens.')
    # Short FAQ pairs remain verbatim, including their question wrapper.
    if source_type == 'legal_faq' and token_count(text) <= max_tokens:
        yield {'text': text, 'section_index': 0, 'section_title': section_title, 'split_fallback': False}
        return
    for index, (heading, body) in enumerate(sections(text)):
        title = heading or section_title
        prefix = heading + '\n' if heading else ''
        if source_type == 'legal_faq' and '\nAnswer:' in body:
            question, body = body.split('\nAnswer:', 1)
            # Repeat a reasonably sized question when a long answer must be split.
            if token_count(question) < max_tokens // 2:
                prefix = question + '\nAnswer: '
            else:
                body = question + '\nAnswer:' + body
        if token_count(prefix) >= max_tokens - 8:
            prefix = ''  # Heading is still preserved in metadata and body below.
            body = heading + '\n' + body
        units = []
        for sentence in sentence_units(body):
            fallback = token_count(prefix + sentence) > max_tokens
            for part in split_long_unit(sentence, prefix, max_tokens):
                units.append((part, fallback))
        if not units:
            units = [('', False)]
        current = []
        for unit in units:
            if current and token_count(prefix + ' '.join(x[0] for x in [*current, unit])) > max_tokens:
                yield {'text': prefix + ' '.join(x[0] for x in current), 'section_index': index,
                       'section_title': title, 'split_fallback': any(x[1] for x in current)}
                overlap = []
                for previous in reversed(current):
                    # Overlap only complete sentences, never fallback fragments.
                    if previous[1] or token_count(' '.join(x[0] for x in [previous, *overlap])) > overlap_tokens:
                        break
                    overlap.insert(0, previous)
                while overlap and token_count(prefix + ' '.join(x[0] for x in [*overlap, unit])) > max_tokens:
                    overlap.pop(0)
                current = overlap
            current.append(unit)
        if current:
            yield {'text': prefix + ' '.join(x[0] for x in current), 'section_index': index,
                   'section_title': title, 'split_fallback': any(x[1] for x in current)}
