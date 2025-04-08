import spacy
from keybert import KeyBERT
import nltk
from nltk.tokenize import sent_tokenize

# Load spaCy and KeyBERT
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

# Input caption or text prompt
text = "There is a vase cup and orange on the table"

# Sentence-level split
sentences = sent_tokenize(text)

all_keywords = set()
entities = set()

for sent in sentences:
    doc = nlp(sent)

    # Named Entities
    entities.update([ent.text for ent in doc.ents])

    # Noun Phrases & Keywords
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    keywords = kw_model.extract_keywords(sent, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    all_keywords.update([kw[0] for kw in keywords])
    all_keywords.update(noun_phrases)

print("\n🔹 Named Entities:", entities)
print("🔸 Keywords + Noun Phrases:", all_keywords)
