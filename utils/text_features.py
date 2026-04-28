# utils/text_features.py

def extract_text_features(text):
    text = text.upper()

    keywords = ["PROOF", "COPY", "SPECIMEN", "VOID", "SAMPLE", "TEST"]

    features = []
    for word in keywords:
        features.append(1 if word in text else 0)

    return features  # length = 3