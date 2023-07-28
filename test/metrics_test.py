"""
TextMetricsTest

Test the BaseTextMetrics class and the classes that inherit from it.
"""

import unittest

from src.metrics import base_text_metric, cosine_similarity, oov_words, transrate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TextMetricsTest(unittest.TestCase):
    def setUp(self):
        self.texts1 = ["This is a test", "This is another test"]
        self.texts2 = ["This is a test", "This is another test", "A completely different sentence"]
        self.text3 = ["The fagagdf sentence has no meaning", "This sentence has meaning", "aga gwthwgs gaerh"]
        self.labels = [1, 0, 1]
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.device = "cpu"
        self.text_metrics = [
            base_text_metric.BaseTextMetric(self.model, self.tokenizer, self.device),
            cosine_similarity.CosineSimilarity(self.model, self.tokenizer, self.device),
            oov_words.OOVWords(self.model,self.tokenizer,self.device),
            transrate.TransRate(self.model, self.tokenizer, self.device)
        ]


    def test_get_embeddings(self):
        for text_metric in self.text_metrics:
            embeddings = text_metric.get_embeddings(self.texts1)
            self.assertEqual(embeddings.shape, (2, 768))

    def test_cosine_similarity(self):
        cosine_similarity_metric = cosine_similarity.CosineSimilarity(self.model, self.tokenizer, self.device)
        cosine_similarity_metric.get_metric(self.texts1, self.texts2)
        self.assertEqual(cosine_similarity_metric.get_metric(self.texts1, self.texts2).shape, (2, 3))
        self.assertEqual(cosine_similarity_metric.get_metric(self.texts1, self.texts1).shape, (2, 2))
        self.assertAlmostEqual(cosine_similarity_metric.get_metric(self.texts1, self.texts1)[0][0], 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity_metric.get_metric(self.texts1, self.texts1)[1][1], 1.0, places=5)
        self.assertNotAlmostEqual(cosine_similarity_metric.get_metric(self.texts1, self.texts1)[0][1], 1.0, places=5)
        self.assertNotAlmostEqual(cosine_similarity_metric.get_metric(self.texts1, self.texts1)[1][0], 1.0, places=5)

    def test_oov_words(self):
        oov_words_metric = oov_words.OOVWords(self.model, self.tokenizer, self.device)
        self.assertEqual(oov_words_metric.get_metric(self.texts1), [0, 0])
        self.assertEqual(oov_words_metric.get_metric(self.texts2), [0, 0, 0])
        self.assertEqual(oov_words_metric.get_metric(self.text3), [1, 0, 3])


