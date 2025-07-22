# This script is for manually downloading additional NLTK data.
# Run it only once during initial setup.
import nltk
nltk.download('cmudict')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
print("NLTK data downloaded successfully!")