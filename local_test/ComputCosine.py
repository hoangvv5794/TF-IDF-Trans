from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

if __name__ == '__main__':
    corpus = [
        'We\'re at a tipping point in human history, a species poised between gaining the stars and losing the planet we call home.',
        'Even in just the past few years, we\'ve greatly expanded our knowledge of how Earth fits within the context of our universe.',
        'NASA\'s Kepler mission has discovered thousands of potential planets around other stars, indicating that Earth is but one of billions of planets in our galaxy.']
    # Initialize an instance of tf-idf Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Generate the tf-idf vectors for the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # compute and print the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(cosine_sim.shape)
    print(cosine_sim)
    t = torch.from_numpy(cosine_sim)
    input = t.view(1, 1, 3, 3)
    # t = t[None, :]
    m = nn.Upsample(scale_factor=3, mode='nearest')
    result = m(input)
    print(result.numpy())
