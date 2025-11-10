from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="cointegrated/rubert-tiny-sentiment-balanced")

with open("data.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

results = classifier(texts)

for text, res in zip(texts, results):
    print(f"Текст: {text}")
    print(f"Класс: {res['label']}, вероятность: {res['score']:.4f}\n")