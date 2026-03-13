import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

raw_corpus = {
    "China": "China, officially the People's Republic of China (PRC), is a vast country located in East Asia. It is the world's second-most populous country, with a population exceeding 1.4 billion. Beijing is the national capital, while Shanghai is the most populous city and largest financial center. China's landscape is vast and diverse, ranging from the Gobi and Taklamakan Deserts in the arid north to the subtropical forests in the wetter south. The Yangtze and Yellow Rivers, the third- and sixth-longest in the world, respectively, flow from the Tibetan Plateau to the densely populated eastern seaboard. China is a unitary one-party socialist republic and is recognized as a major global power, boasting the world's second-largest economy by nominal GDP. The Great Wall of China is one of its most famous historical landmarks.",
    
    "Russia": "Russia, officially the Russian Federation, is a transcontinental country spanning Eastern Europe and Northern Asia. It is the largest country in the world by land area, covering over 17 million square kilometers, and encompasses eleven time zones. Moscow is the capital and largest city, followed by Saint Petersburg, which serves as Russia's cultural center. The country's landscape includes vast plains, dense forests known as the taiga, and prominent mountain ranges such as the Urals and the Caucasus. Russia has a rich history characterized by the Russian Empire and its later role as the leading constituent of the Soviet Union. Today, it is a federal semi-presidential republic. The Russian economy is heavily reliant on its extensive natural resources, particularly oil and natural gas.",
    
    "Pakistan": "Pakistan, officially the Islamic Republic of Pakistan, is a country in South Asia. It is the world's fifth-most populous country, with a population of over 240 million people, and has the second-largest Muslim population. Islamabad is the nation's capital, while Karachi is its largest city and financial hub. Pakistan has a 1,046-kilometre coastline along the Arabian Sea and the Gulf of Oman in the south. The country's geography is highly diverse, featuring the towering peaks of the Karakoram and Himalayan mountain ranges in the north, including K2, the world's second-highest mountain. The Indus River flows through the length of the country, supporting agriculture and densely populated valleys. Pakistan was created in 1947 as an independent homeland for Indian Muslims following the partition of British India."
}

documents = list(raw_corpus.values())
columns_names = list(raw_corpus.keys())

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(tfidf_matrix.T.toarray(), index=feature_names, columns=columns_names)

df_sorted_by_china = df_tfidf.sort_values(by="China", ascending=False)
df_sorted_by_Russia = df_tfidf.sort_values(by="Russia", ascending=False)
df_sorted_by_Pakistan = df_tfidf.sort_values(by="Pakistan", ascending=False)
pd.set_option('display.max_rows', None)

# print("TF-IDF matrix after cleaning and filtering stop words")
# print(df_tfidf.head(15))
# print("\n====== China top5 word ======")
# print(df_sorted_by_china.head(5))
# print("\n====== Russia top5 word ======")
# print(df_sorted_by_Russia.head(5))
# print("\n====== Pakistan top5 word ======")
# print(df_sorted_by_Pakistan.head(5))

new_text = "This colossal nation spans two different continents, making it a bridge between distinct cultural spheres. Because of its sheer size, traveling from its western borders to its eastern shores means passing through multiple time zones. The natural environment is notoriously harsh, featuring vast, freezing plains and endless stretches of dense, needle-leaf forests. Historically, it has transitioned through powerful imperial eras and a major twentieth-century ideological union, leaving a complex legacy. Today, its global influence is heavily sustained by the extraction and exportation of immense underground energy reserves, particularly fossil fuels, which lie hidden beneath its freezing terrain."
new_vector = vectorizer.transform([new_text])

similarity_scores = cosine_similarity(new_vector, tfidf_matrix)[0]

results = dict(zip(columns_names, similarity_scores))

print("similarity score:")
for country, score in results.items():
    print(f" {country}: {score:.4f}")

best_match = max(results, key=results.get)
print(f"\n The country closest to this new text is: {best_match}")

new_text_nonzero_indices = new_vector.nonzero()[1]

russia_index = columns_names.index("Russia")
russia_vector = tfidf_matrix[russia_index]
russia_nonzero_indices = russia_vector.nonzero()[1]

shared_indices = np.intersect1d(new_text_nonzero_indices, russia_nonzero_indices)

evidence_data = []
for idx in shared_indices:
    word = feature_names[idx]
    weight_in_new = new_vector[0, idx]
    weight_in_russia = russia_vector[0, idx]
    evidence_data.append({
        "Matched resonance words": word,
        "Weight in new text": round(weight_in_new, 4),
        "Weight in the original Russian text": round(weight_in_russia, 4)
    })

df_evidence = pd.DataFrame(evidence_data).sort_values(by="Weight in the original Russian text", ascending=False)
print(df_evidence.to_string(index=False))


best_match = max(results, key=results.get)
print(f"\n The country closest to this new text is: {best_match}")
