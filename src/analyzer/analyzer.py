import pickle


class Analyzer:

    def __init__(self, path):
        self.model_path = path
        self.model = self._load_model()

    def _load_model(self):
        model = pickle.load(open(self.model_path, 'rb'))
        return model

    def analyze_sentence(self, text) -> (float, float):
        if self.model is None:
            self.model = self._load_model()
        predictions = self.model.predict_proba(text)

    def get_sentiment_analysis(self, texts: []) -> []:
        return [self.analyze_sentence(text) for text in texts]
