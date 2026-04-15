"""
Smart Router — classifies incoming messages into routes WITHOUT using API calls.
Routes: light / medium / deep

This saves tokens by skipping unnecessary pipeline steps for simple messages.

Design philosophy:
  - light = small talk, greetings → Front Agent only (1 API call)
  - medium = regular sharing, stories, emotions → NLP + Belief Graph + Front (2 API calls)
  - deep = real distress, core belief crisis, loss of control → full pipeline + Strategy (3 API calls)

  medium handles most emotional conversations well enough.
  deep is reserved for moments where the Front Agent needs expert-informed
  strategic direction — distress so deep that a generic empathetic response
  won't cut it.
"""

import re
from typing import Literal

RouteType = Literal["light", "medium", "deep"]

# ─── Hebrew + English greeting / small talk patterns ───
_LIGHT_PATTERNS = [
    # Greetings
    r"^(היי|הי|שלום|אהלן|יו|בוקר טוב|ערב טוב|לילה טוב|מה קורה|מה נשמע|הלו)[\s?!.]*$",
    r"^(hey|hi|hello|yo|sup|what\'?s up)[\s?!.]*$",
    # Short affirmatives
    r"^(כן|לא|אוקיי|אוקי|ok|okay|בסדר|סבבה|תודה|תודה רבה|thanks|yep|nope|yeah|nah|cool|נכון|ברור|בטח|sure)[\s?!.]*$",
    # Very short (1-2 words, no emotional content)
    r"^[\w]{1,10}[\s?!.]*$",
    # Emoji only
    r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\s]+$",
]

# ─── Deep distress signals ───
# These detect PATTERNS of thinking, not specific words:
#   1. Self-rejection / self-hatred (any form)
#   2. Absolute helplessness / loss of control
#   3. Black-and-white generalizations (always/never/everyone/nobody)
#   4. Rejection / abandonment / humiliation
#   5. Existential crisis / pointlessness
_DEEP_SIGNALS = [
    # ── Pattern 1: Self-rejection (flexible — catches all forms) ──
    r"אני לא שווה",
    r"אני לא מסוגל",
    r"אני לא ראוי",
    r"אני פשוט לא",
    r"(אני\s+)?שונא.{0,10}(את עצמי|עצמי|שזה|שאני)",  # "שונא את עצמי" + "שונא שזה קורה"
    r"(שנאתי|שנאה).{0,15}(עצמי|את עצמי)",
    r"אני (חלש|פתטי|מגעיל|אפס|כלום|שום דבר)",

    # ── Pattern 2: Loss of control / helplessness ──
    r"אני לא יכול",  # any form — not just "יכול יותר"
    r"לא (יכול|יכולה|מצליח|מצליחה) (להפסיק|להתעלם|לשלוט|לעצור|להירגע)",
    r"זה (חוזר|חוזרת|מופיע|עולה).{0,10}(כל הזמן|תמיד|שוב|מחדש)",
    r"אין לי (סיכוי|כוח|אנרגיה|מוצא|דרך)",
    r"אני (נשבר|נשברת|מתפרק|מתפרקת|מתמוטט|מתמוטטת)",
    r"(מרגיש|הרגשתי|ההרגשה|מרגישה).{0,10}(ריק|חסר תקווה|אבוד|אבודה|מת|חנוק)",

    # ── Pattern 3: Black-and-white / absolute thinking ──
    r"(תמיד|אף פעם).{0,20}(אני|לי|אותי|ככה)",  # "תמיד אני", "תמיד ככה לי"
    r"כל הזמן",
    r"(כולם|אף אחד).{0,15}(חושב|חושבים|מבין|אוהב|רוצה)",
    r"שוב ושוב",
    r"כל פעם (שאני|שזה|מחדש)",
    r"הכל (נגדי|מתפרק|הולך לעזאזל|גמור)",
    r"אין טעם",
    r"זה (בטוח|תמיד) ייגמר רע",

    # ── Pattern 4: Rejection / abandonment / humiliation ──
    r"(עזבו|נטשו|בגדו|השפילו|זרקו|דחו).{0,10}אותי",
    r"(עזב|נטש|בגד|השפיל|זרק|דחה|דחתה).{0,10}אותי",
    r"אף אחד לא (אוהב|רוצה|מבין|שומע|רואה)",
    r"לא אכפת לאף אחד",
    r"אני לבד",

    # ── Pattern 5: Existential / suicidal ideation signals ──
    r"אין טעם (לחיות|להמשיך|בכלום|בחיים)",
    r"למה (אני|בכלל) (חי|קיים|ממשיך)",
    r"עדיף (בלעדיי|שלא הייתי|שאני לא)",

    # ── English deep signals ──
    r"nobody (cares|understands|loves|listens)",
    r"i('m| am) (worthless|useless|broken|nothing|alone|empty|numb)",
    r"everything is (falling apart|hopeless|pointless|over)",
    r"i can'?t (take|handle|do|stand|bear) (it|this|anymore)",
    r"what'?s the point",
    r"i (hate|despise) myself",
    r"i('m| am) (losing|lost) (control|it|myself|my mind)",

    # ── Pattern 6: Advice / guidance seeking ──
    r"מה אתה מציע",
    r"מה (כדאי|עדיף|אפשר) (לעשות|לומר|להגיד|להגיב)",
    r"(יש לך|יש לי|תן לי).{0,10}(עצה|טיפ|כלי|שיטה|טכניקה|פתרון|הצעה)",
    r"(תעזור|עזור|תוכל לעזור).{0,10}לי",
    r"NLP",
    r"(nlp|אן.{0,3}אל.{0,3}פי)",
    r"(איך|כיצד).{0,20}(להתמודד|לשנות|לשפר|לטפל|לפתור|להתנהג|לגשת)",

    # ── Pattern 7: Long emotional narrative (processed in classify_message) ──
]

# Compile patterns once at import time
_LIGHT_RE = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _LIGHT_PATTERNS]
_DEEP_RE = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _DEEP_SIGNALS]


# ─── Emotional word roots (Hebrew) ───
# Using roots instead of exact words so we catch all conjugations:
#   "מרגיש", "הרגשתי", "ההרגשה", "להרגיש" all match
_EMOTIONAL_ROOTS_HE = [
    re.compile(r"(מרגיש|הרגשתי|ההרגשה|להרגיש|מרגישה|הרגשה)", re.UNICODE),
    re.compile(r"(כואב|כאב|הכאב|לכאוב|כואבת)", re.UNICODE),
    re.compile(r"(פוחד|מפחד|פחד|מפחדת|פוחדת|הפחד)", re.UNICODE),
    re.compile(r"(עצוב|עצובה|עצב|העצב)", re.UNICODE),
    re.compile(r"(כועס|כעס|כועסת|הכעס|לכעוס)", re.UNICODE),
    re.compile(r"(בושה|מתבייש|מתביישת|התבייש)", re.UNICODE),
    re.compile(r"(אשמה|אשם|מרגיש אשם|מאשים)", re.UNICODE),
    re.compile(r"(חרדה|חרד|חרדת|מחריד)", re.UNICODE),
    re.compile(r"(פגוע|נפגע|נפגעת|פגיעה|פוגע)", re.UNICODE),
    re.compile(r"(בוכה|בכיתי|לבכות|בכי)", re.UNICODE),
    re.compile(r"(שבור|שבורה|נשבר|נשברתי|להישבר)", re.UNICODE),
    re.compile(r"(מדוכא|מדוכאת|דיכאון|דיכדוך)", re.UNICODE),
    re.compile(r"(מתוסכל|מתוסכלת|תסכול|מתסכל)", re.UNICODE),
    re.compile(r"(מיואש|מיואשת|ייאוש|נואש)", re.UNICODE),
    re.compile(r"(מבולבל|מבולבלת|בלבול|מבלבל)", re.UNICODE),
    re.compile(r"(מותש|מותשת|תשישות|מותשים)", re.UNICODE),
    re.compile(r"(בודד|בודדה|בדידות|לבד)", re.UNICODE),
    re.compile(r"(שונא|שנאתי|שנאה|לשנוא)", re.UNICODE),
    re.compile(r"(השפיל|השפלה|משפיל|הושפלתי)", re.UNICODE),
]

_EMOTIONAL_KEYWORDS_EN = {
    "feel", "feeling", "felt", "hurt", "hurting",
    "scared", "sad", "angry", "ashamed", "shame",
    "guilty", "guilt", "anxious", "anxiety",
    "broken", "crying", "cried", "depressed", "depression",
    "frustrated", "hopeless", "confused", "exhausted",
    "lonely", "alone", "numb", "empty", "worthless",
    "humiliated", "rejected", "abandoned",
}


def _count_emotional_hits(text: str, words: list[str]) -> int:
    """Count emotional signal hits using roots (Hebrew) and keywords (English)."""
    hits = 0
    # Hebrew: check roots against full text (catches all conjugations)
    for root_pattern in _EMOTIONAL_ROOTS_HE:
        if root_pattern.search(text):
            hits += 1
    # English: word-level match
    word_set = set(w.lower() for w in words)
    hits += len(word_set & _EMOTIONAL_KEYWORDS_EN)
    return hits


def classify_message(
    text: str,
    last_route: str = "light",
    recent_routes: list = None,
) -> RouteType:
    """
    Classify a user message into a route:
      - light:  small talk, greetings, very short → 1 API call
      - medium: regular sharing, stories, updates → 2 API calls
      - deep:   emotional distress, core beliefs  → 3 API calls

    last_route: the route of the previous turn.
    recent_routes: list of last N route strings — if 2+ were medium/deep,
                   the conversation is "emotionally active" and we don't drop to light.
    """
    stripped = text.strip()

    # ── Derived: is this conversation emotionally sustained? ──
    # If the last 3 turns had 2+ medium/deep → sustained emotional conversation.
    sustained = False
    if recent_routes:
        heavy = sum(1 for r in recent_routes[-3:] if r in ("medium", "deep"))
        sustained = heavy >= 2

    def _floor_route(candidate: RouteType) -> RouteType:
        """Apply momentum floor: sustained conversations don't drop to light."""
        if candidate == "light" and (last_route in ("deep", "medium") or sustained):
            return "medium"
        return candidate

    # ── Rule 1: Empty or very short ──
    if len(stripped) <= 3:
        if last_route == "deep":
            return "medium"
        return _floor_route("light")

    # ── Rule 2: Check light patterns ──
    for pattern in _LIGHT_RE:
        if pattern.match(stripped):
            return _floor_route("light")

    # ── Rule 3: Word count heuristic ──
    words = stripped.split()
    if len(words) <= 2 and not any(p.search(stripped) for p in _DEEP_RE):
        return _floor_route("light")

    # ── Rule 4: Check deep signals (pattern-based) ──
    for pattern in _DEEP_RE:
        if pattern.search(stripped):
            return "deep"

    # ── Rule 5: Emotional density heuristic ──
    emotional_hits = _count_emotional_hits(stripped, words)

    if emotional_hits >= 2 or (len(words) > 15 and emotional_hits >= 1) or len(words) > 40:
        return "deep"

    # ── Default: medium ──
    return "medium"
