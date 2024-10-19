from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from waitress import serve  # Import waitress for production server

app = Flask(__name__)

# Load the model, label encoder, and count vectorizer
model = joblib.load('language_detection_model.pkl')
encoder = joblib.load('label_encoder.pkl')
CV = joblib.load('count_vectorizer.pkl')  # Load the CountVectorizer for text transformation

@app.route('/')
def home():
    return render_template('home.html')  # Render the home page template

@app.route('/detect')
def detect():
    return render_template('index.html')  # Render the language detection template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the form
        text = request.form['text']
        
        # Transform the text using the loaded CountVectorizer
        x = CV.transform([text]).toarray()
        
        # Predict the language using the loaded model
        prediction = model.predict(x)
        
        # Decode the predicted label using the loaded LabelEncoder
        language = encoder.inverse_transform(prediction)
        
        # Return the result as a JSON response
        return jsonify({'prediction': language[0]})
    
    except Exception as e:
        # Handle any errors that may occur
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Use waitress for serving the application in production
    serve(app, host='0.0.0.0', port=8080)
