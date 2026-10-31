module TradeTestFixtures

using Classify, TradingStrategy

"""Classifier for Trade tests that never request predictions or trained artifacts."""
struct NoInferenceClassifier <: Classify.AbstractClassifier end

"""Return the zero history requirement of the no-inference test classifier."""
Classify.requiredminutes(::NoInferenceClassifier) = 0

"""Return an explicit strategy configuration independent of locally trained classifiers."""
strategy() = TradingStrategy.StrategyConfig(classifier=NoInferenceClassifier())

end
