""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.define import LANG_ID2SYMBOLS
from text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id, _id_to_symbol = {}, {}
for id, symbols in LANG_ID2SYMBOLS.items():
    _symbol_to_id[id] = {s: i for i, s in enumerate(symbols)}
    _id_to_symbol[id] = {i: s for i, s in enumerate(symbols)}
# print("AAAA_symbol_to_id", _symbol_to_id)
# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names, lang_id="en"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      lang_id: language identifier

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), lang_id)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), lang_id)
        sequence += _arpabet_to_sequence(m.group(2), lang_id)
        text = m.group(3)

    return sequence


def sequence_to_text(sequence, lang_id):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol[lang_id]:
            s = _id_to_symbol[lang_id][symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, lang_id):
    return [_symbol_to_id[lang_id][s] for s in symbols if _should_keep_symbol(s, lang_id)]


def _arpabet_to_sequence(text, lang_id):
    return _symbols_to_sequence(["@" + s for s in text.split()], lang_id)


def _should_keep_symbol(s, lang_id):
    return s in _symbol_to_id[lang_id] and s != "_" and s != "~"
