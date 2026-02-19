# Config
import traceback

# Global dictionary to track import errors
_IMPORT_ERRORS = {}

def _optional_import(name: str, hint: str = ""):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        _IMPORT_ERRORS[name] = (hint, traceback.format_exc())
        return None

# Dictionary mapping labels to emojis
EMOJI_MAP = {
    "marah": "😡", "sedih": "😢", "takut": "😨",
    "bahagia": "😄", "cinta": "🥰", "senang": "😄",
    "netral": "😐", "lainnya": "❓", "other": "❓",
}

def get_emoji(label: str) -> str:
    key = "" if label is None else str(label).strip().lower()
    return EMOJI_MAP.get(key, "❓")

# NLP Normalization Dictionary
DEFAULT_KAMUS_NORMALISASI = {
    "yg": "yang", "yng": "yang", "tdk": "tidak", "dgn": "dengan", "dr": "dari", "krn": "karena",
    "bgt": "banget", "udh": "udah", "udah": "sudah", "kalo": "kalau", "klo": "kalau", "sm": "sama",
    "gw": "saya", "gue": "saya", "lu": "kamu", "kmu": "kamu", "sy": "saya", "pls": "please",
    "utk": "untuk", "dlm": "dalam", "sdh": "sudah", "dg": "dengan", "jd": "jadi", "gt": "gitu",
    "jg": "juga", "tp": "tapi", "hrs": "harus", "mnrt": "menurut", "dl": "dulu", "lg": "lagi",
    "bnyk": "banyak", "bkn": "bukan", "br": "baru", "bhs": "bahasa", "kpd": "kepada", "sbg": "sebagai",
    "ga": "tidak", "gak": "tidak", "nggak": "tidak", "dah": "sudah", "pake": "pakai",
}
