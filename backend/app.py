"""
Flask Backend for Next Word Prediction App
Handles user registration with MongoDB and Clerk authentication
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import os
import pickle
from dotenv import load_dotenv
import base64
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow frontend requests from React app
CORS(app, origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"])

# Flask-Mail configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', os.getenv('MAIL_USERNAME'))

mail = Mail(app)

# MongoDB connection using environment variable
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "next_word_prediction")

# Initialize MongoDB client
try:
    # Connect with serverSelectionTimeoutMS to fail fast if connection issues
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Force connection test by calling server_info()
    client.server_info()
    
    db = client[DB_NAME]
    users_collection = db["users"]
    books_collection = db["books"]
    
    # Create index on clerkUserId for faster lookups (if doesn't exist)
    users_collection.create_index("clerkUserId", unique=True)
    books_collection.create_index("userId")
    books_collection.create_index([("userId", 1), ("createdAt", -1)])
    
    print("✅ Connected to MongoDB successfully!")
    print(f"📁 Database: {DB_NAME}")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    print("Please check your MONGO_URI in .env file")


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Next Word Prediction API is running!"
    })


@app.route("/api/contact", methods=["POST"])
def send_contact_email():
    """
    Send contact form email to admin
    Receives form data and sends email notification
    """
    try:
        # Get form data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
        
        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        subject = data.get("subject")
        message = data.get("message")
        
        # Check if all required fields are present
        if not all([name, email, phone, subject, message]):
            return jsonify({
                "status": "error",
                "message": "All fields are required"
            }), 400
        
        # Create email message
        msg = Message(
            subject=f"Contact Form: {subject}",
            recipients=["arunk330840@gmail.com"],
            reply_to=email
        )
        
        # Email body
        msg.body = f"""
New Contact Form Submission from Typen

Name: {name}
Email: {email}
Phone: {phone}
Subject: {subject}

Message:
{message}

---
This message was sent from the Typen contact form.
        """
        
        msg.html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                <h2 style="color: #4F46E5; border-bottom: 2px solid #4F46E5; padding-bottom: 10px;">
                    New Contact Form Submission from Typen
                </h2>
                
                <div style="margin: 20px 0;">
                    <p><strong>Name:</strong> {name}</p>
                    <p><strong>Email:</strong> <a href="mailto:{email}">{email}</a></p>
                    <p><strong>Phone:</strong> {phone}</p>
                    <p><strong>Subject:</strong> {subject}</p>
                </div>
                
                <div style="background-color: #f9fafb; padding: 15px; border-left: 4px solid #4F46E5; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #4F46E5;">Message:</h3>
                    <p style="white-space: pre-wrap;">{message}</p>
                </div>
                
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                <p style="color: #666; font-size: 12px;">
                    This message was sent from the Typen contact form.
                </p>
            </div>
        </body>
        </html>
        """
        
        # Send email
        mail.send(msg)
        
        return jsonify({
            "status": "success",
            "message": "Email sent successfully"
        }), 200
        
    except Exception as e:
        print(f"Error sending contact email: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to send email: {str(e)}"
        }), 500


@app.route("/api/users/register", methods=["POST"])
def register_user():
    """
    Register a new user after Clerk signup
    Stores user details in MongoDB
    Prevents duplicate users based on Clerk User ID
    """
    try:
        # Get user data from request body
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
        
        clerk_user_id = data.get("clerkUserId")
        email = data.get("email")
        username = data.get("username")
        full_name = data.get("fullName")
        
        # Check if required fields are present
        if not clerk_user_id or not email:
            return jsonify({
                "status": "error",
                "message": "Clerk User ID and email are required"
            }), 400
        
        # Check if user already exists in database
        existing_user = users_collection.find_one({"clerkUserId": clerk_user_id})
        
        if existing_user:
            # User already exists, return success (for login flow)
            return jsonify({
                "status": "success",
                "message": "User already exists",
                "user": {
                    "clerkUserId": existing_user["clerkUserId"],
                    "email": existing_user["email"],
                    "username": existing_user.get("username"),
                    "fullName": existing_user.get("fullName")
                }
            }), 200
        
        # Create new user document
        new_user = {
            "clerkUserId": clerk_user_id,
            "email": email,
            "username": username or email.split("@")[0],  # Default username from email
            "fullName": full_name or "",
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        
        # Insert user into MongoDB
        result = users_collection.insert_one(new_user)
        
        if result.inserted_id:
            return jsonify({
                "status": "success",
                "message": "User registered successfully",
                "user": {
                    "clerkUserId": clerk_user_id,
                    "email": email,
                    "username": new_user["username"],
                    "fullName": new_user["fullName"]
                }
            }), 201
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to register user"
            }), 500
            
    except Exception as e:
        print(f"Error registering user: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@app.route("/api/users/<clerk_user_id>", methods=["GET"])
def get_user(clerk_user_id):
    """
    Get user details by Clerk User ID
    Used to fetch user data on dashboard
    """
    try:
        user = users_collection.find_one({"clerkUserId": clerk_user_id})
        
        if user:
            return jsonify({
                "status": "success",
                "user": {
                    "clerkUserId": user["clerkUserId"],
                    "email": user["email"],
                    "username": user.get("username"),
                    "fullName": user.get("fullName"),
                    "createdAt": user["createdAt"].isoformat() if user.get("createdAt") else None
                }
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "User not found"
            }), 404
            
    except Exception as e:
        print(f"Error fetching user: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


# ==================== BOOK/DOCUMENT ENDPOINTS ====================

@app.route("/api/books", methods=["POST"])
def create_book():
    """
    Create a new book/document
    Requires: userId, title
    Optional: description, coverImage (base64), genre
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
        
        user_id = data.get("userId")
        title = data.get("title")
        
        if not user_id or not title:
            return jsonify({
                "status": "error",
                "message": "User ID and title are required"
            }), 400
        
        # Create new book document
        new_book = {
            "userId": user_id,
            "title": title,
            "description": data.get("description", ""),
            "coverImage": data.get("coverImage", ""),  # Base64 or URL
            "genre": data.get("genre", ""),
            "content": "",  # Will be updated in editor
            "wordCount": 0,
            "status": "draft",
            "isFavorite": False,
            "isArchived": False,
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc)
        }
        
        result = books_collection.insert_one(new_book)
        
        if result.inserted_id:
            return jsonify({
                "status": "success",
                "message": "Book created successfully",
                "book": {
                    "id": str(result.inserted_id),
                    "title": new_book["title"],
                    "description": new_book["description"],
                    "coverImage": new_book["coverImage"],
                    "genre": new_book["genre"],
                    "status": new_book["status"],
                    "createdAt": new_book["createdAt"].isoformat()
                }
            }), 201
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create book"
            }), 500
            
    except Exception as e:
        print(f"Error creating book: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@app.route("/api/books/user/<user_id>", methods=["GET"])
def get_user_books(user_id):
    """
    Get all books for a user
    Supports filtering by status, favorites, archived
    """
    try:
        # Build query
        query = {"userId": user_id}
        
        # Optional filters
        status = request.args.get("status")
        is_favorite = request.args.get("favorite")
        is_archived = request.args.get("archived")
        
        if status:
            query["status"] = status
        if is_favorite == "true":
            query["isFavorite"] = True
        if is_archived == "true":
            query["isArchived"] = True
        elif is_archived == "false":
            query["isArchived"] = False
        
        # Get books sorted by updatedAt descending
        books = list(books_collection.find(query).sort("updatedAt", -1))
        
        # Format response
        formatted_books = []
        for book in books:
            formatted_books.append({
                "id": str(book["_id"]),
                "title": book["title"],
                "description": book.get("description", ""),
                "coverImage": book.get("coverImage", ""),
                "genre": book.get("genre", ""),
                "content": book.get("content", "")[:100] + "..." if book.get("content") else "",
                "wordCount": book.get("wordCount", 0),
                "status": book.get("status", "draft"),
                "isFavorite": book.get("isFavorite", False),
                "isArchived": book.get("isArchived", False),
                "createdAt": book["createdAt"].isoformat() if book.get("createdAt") else None,
                "updatedAt": book["updatedAt"].isoformat() if book.get("updatedAt") else None
            })
        
        return jsonify({
            "status": "success",
            "books": formatted_books,
            "count": len(formatted_books)
        }), 200
        
    except Exception as e:
        print(f"Error fetching books: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@app.route("/api/books/<book_id>", methods=["GET"])
def get_book(book_id):
    """
    Get a single book by ID
    """
    try:
        book = books_collection.find_one({"_id": ObjectId(book_id)})
        
        if book:
            return jsonify({
                "status": "success",
                "book": {
                    "id": str(book["_id"]),
                    "userId": book["userId"],
                    "title": book["title"],
                    "description": book.get("description", ""),
                    "coverImage": book.get("coverImage", ""),
                    "genre": book.get("genre", ""),
                    "content": book.get("content", ""),
                    "wordCount": book.get("wordCount", 0),
                    "status": book.get("status", "draft"),
                    "isFavorite": book.get("isFavorite", False),
                    "isArchived": book.get("isArchived", False),
                    "createdAt": book["createdAt"].isoformat() if book.get("createdAt") else None,
                    "updatedAt": book["updatedAt"].isoformat() if book.get("updatedAt") else None
                }
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Book not found"
            }), 404
            
    except Exception as e:
        print(f"Error fetching book: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@app.route("/api/books/<book_id>", methods=["PUT"])
def update_book(book_id):
    """
    Update a book's content or metadata
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
        
        # Build update document
        update_data = {"updatedAt": datetime.now(timezone.utc)}
        
        # Fields that can be updated
        updatable_fields = ["title", "description", "coverImage", "genre", 
                           "content", "wordCount", "status", "isFavorite", "isArchived"]
        
        for field in updatable_fields:
            if field in data:
                update_data[field] = data[field]
        
        result = books_collection.update_one(
            {"_id": ObjectId(book_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return jsonify({
                "status": "success",
                "message": "Book updated successfully"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Book not found or no changes made"
            }), 404
            
    except Exception as e:
        print(f"Error updating book: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


@app.route("/api/books/<book_id>", methods=["DELETE"])
def delete_book(book_id):
    """
    Delete a book
    """
    try:
        result = books_collection.delete_one({"_id": ObjectId(book_id)})
        
        if result.deleted_count > 0:
            return jsonify({
                "status": "success",
                "message": "Book deleted successfully"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Book not found"
            }), 404
            
    except Exception as e:
        print(f"Error deleting book: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500


# ============================================
# WORD PREDICTION API
# ============================================

# Initialize Model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', 'next_word_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'next_word_model.h5')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.pkl')
prediction_model = None
prediction_tokenizer = None
INDEX_TO_WORD = {}
MAX_SEQUENCE_LEN = 30

# Load model at startup
try:
    if os.path.exists(MODEL_PATH):
        prediction_model = load_model(MODEL_PATH)
        print("✅ Next Word Prediction Model loaded successfully!")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading prediction model: {e}")

# Load tokenizer and model config so word indexes match the trained model.
try:
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, 'rb') as tokenizer_file:
            prediction_tokenizer = pickle.load(tokenizer_file)
        INDEX_TO_WORD = getattr(prediction_tokenizer, 'index_word', {})
        print(f"✅ Tokenizer loaded successfully! Vocabulary size: {len(INDEX_TO_WORD)}")
    else:
        print(f"⚠️ Tokenizer file not found at {TOKENIZER_PATH}")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")

try:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'rb') as config_file:
            model_config = pickle.load(config_file)
        MAX_SEQUENCE_LEN = int(model_config.get('max_sequence_len', MAX_SEQUENCE_LEN))
        print(f"✅ Model config loaded successfully! Max sequence length: {MAX_SEQUENCE_LEN}")
except Exception as e:
    print(f"❌ Error loading model config: {e}")

def model_input_length():
    """Return the sequence length expected by the next-word model."""
    return max(MAX_SEQUENCE_LEN - 1, 1)

def text_to_sequence(text, max_len=None):
    """Convert text to numeric sequence using the trained tokenizer."""
    if prediction_tokenizer is None:
        raise RuntimeError("Prediction tokenizer not available")

    sequence = prediction_tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([sequence], maxlen=max_len or model_input_length(), padding='pre')
    return padded

def word_from_index(index):
    """Map a model output index back to a tokenizer word."""
    word = INDEX_TO_WORD.get(int(index))
    if not word or word == '<OOV>':
        return None
    return word

def default_predictions_from_tokenizer(limit=8):
    """Return common tokenizer words when there is no text to predict from."""
    predictions = []
    for index in sorted(INDEX_TO_WORD):
        word = word_from_index(index)
        if not word:
            continue

        rank = len(predictions) + 1
        predictions.append({
            "id": rank,
            "word": word,
            "rank": str(rank) if rank <= 5 else f"C{rank - 5}",
            "type": "probable" if rank <= 5 else "creative",
            "confidence": None
        })

        if len(predictions) == limit:
            break

    return predictions


@app.route("/api/predict", methods=["POST"])
def predict_next_words():
    """
    Predict next words using the trained Neural Network model
    Returns top 5 probable words + 3 alternative creative words
    """
    try:
        if prediction_model is None:
            print("Prediction model not loaded")
            return jsonify({
                "status": "error",
                "message": "Prediction model not available"
            }), 500

        data = request.get_json()
        text = data.get("text", "").strip()
        genre = data.get("genre", "fiction").strip()

        if prediction_tokenizer is None:
            print("Prediction tokenizer not loaded")
            return jsonify({
                "status": "error",
                "message": "Prediction tokenizer not available"
            }), 500

        if not text:
            # Return common tokenizer words for empty text
            return jsonify({
                "status": "success",
                "predictions": default_predictions_from_tokenizer()
            }), 200

        # Convert text to model input sequence
        sequence = text_to_sequence(text)
        
        # Get predictions from model
        predictions = prediction_model.predict(sequence, verbose=0)
        
        # Get top word indices from the model output.
        top_indices = np.argsort(predictions[0])[::-1]
        
        # Build response predictions
        predictions_list = []
        
        # Top 5 as probable words
        for idx in top_indices:
            word = word_from_index(idx)
            if not word:
                continue

            confidence = float(predictions[0][idx])
            predictions_list.append({
                "id": len(predictions_list) + 1,
                "word": word,
                "rank": str(len(predictions_list) + 1),
                "type": "probable",
                "confidence": round(confidence, 4)
            })

            if len(predictions_list) == 5:
                break
        
        # Next 3 as creative alternatives
        for idx in top_indices:
            if len(predictions_list) == 8:
                break

            word = word_from_index(idx)
            if not word or any(prediction["word"] == word for prediction in predictions_list):
                continue

            confidence = float(predictions[0][idx])
            creative_rank = len(predictions_list) - 4
            predictions_list.append({
                "id": len(predictions_list) + 1,
                "word": word,
                "rank": f"C{creative_rank}",
                "type": "creative",
                "confidence": round(confidence, 4)
            })
        
        print(f"Predicted next words for input '{text}': {[p['word'] for p in predictions_list[:5]]}")
        
        return jsonify({
            "status": "success",
            "predictions": predictions_list,
            "model": "Neural Network LSTM"
        }), 200

    except Exception as e:
        print(f"Error predicting words: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Run the Flask app
if __name__ == "__main__":
    # Get port from environment or default to 5000
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"🚀 Starting Flask server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
