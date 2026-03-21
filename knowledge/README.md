# Knowledge Base

## מבנה

```
knowledge/
├── knowledge_core.md      ← שכבה 1: כללי ברזל (תמיד בפרומפט, ~800 טוקנים)
├── sources/               ← מקורות גולמיים מ-NotebookLM (לשימוש עתידי ב-RAG)
├── distilled/             ← סיכומים מזוקקים פר מחברת
└── README.md              ← הקובץ הזה
```

## מחברות NotebookLM רלוונטיות

| מחברת | מקורות | נושא | סטטוס |
|-------|--------|------|-------|
| communicator | 93 | סגנון תקשורת "גברת רביע", חברות טיפולית, reframing | ✅ מזוקק |
| Steve Andreas | 7 | NLP: submodalities, דיאלוג פנימי, reframing, parts | ✅ מזוקק |
| Robert Dilts | 4 | Logical Levels, Reimprinting, Belief Change | ✅ מזוקק |
| Connirae Andreas | 1 | Core Transformation process | ✅ מזוקק |
| Core Transformation (book) | 2 | Core States, שרשרת כוונות | ✅ מזוקק |
| Michael Hall | 3 | Meta-States, Neuro-Semantics | ✅ מזוקק |
| Lucas Derks | 2 | Social Panoramas | ✅ מזוקק |
| Tad James | 2 | Time Line Therapy | ✅ מזוקק |
| Richard Bandler | 0 | (ריק — להוסיף מקורות) | ⏳ ממתין |
| Belief Networks | 18 | ארכיטקטורת גרפי אמונות | ⏳ לא רלוונטי לליבה |

## איך לעדכן

1. הוסף מחברת חדשה ב-NotebookLM
2. זקק את העקרונות המרכזיים (אפשר לבקש מ-NotebookLM סיכום)
3. שמור סיכום ב-`distilled/NOTEBOOK_NAME.md`
4. עדכן את `knowledge_core.md` עם עקרונות חדשים (שמור מתחת ל-800 טוקנים!)
