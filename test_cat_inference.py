#!/usr/bin/env python3
"""
Test the exported CoreML CLIP models on cat.jpeg to see if they can identify a cat.
"""

import numpy as np
import coremltools as ct
from PIL import Image
from transformers import CLIPTokenizerFast

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_cat_identification():
    print("🐱 Testing CoreML CLIP models on cat.jpeg")
    print("=" * 50)
    
    # Load the CoreML models
    try:
        text_model = ct.models.MLModel('out/TextEncoder.mlpackage')
        image_model = ct.models.MLModel('out/ImageEncoder.mlpackage')
        print("✓ CoreML models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load CoreML models: {e}")
        return
    
    # Load tokenizer
    tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
    
    # Test queries
    queries = [
        "a cat",
        "a dog", 
        "a bird",
        "a photo of a cat",
        "a cute kitten",
        "an animal",
        "a furry pet"
    ]
    
    print("\n📸 Processing cat.jpeg image...")
    try:
        # Load and process image
        cat_image = Image.open('images/cat.jpeg').convert('RGB')
        print(f"✓ Image loaded: {cat_image.size}")
        
        # Resize to 224x224 as expected by the model
        cat_image_resized = cat_image.resize((224, 224), Image.Resampling.LANCZOS)
        print(f"✓ Image resized to: {cat_image_resized.size}")
        
        # Get image embedding
        image_result = image_model.predict({'colorImage': cat_image_resized})
        image_embedding = image_result['embOutput'][0]  # First output
        print(f"✓ Image embedding shape: {image_embedding.shape}")
        
    except Exception as e:
        print(f"✗ Failed to process image: {e}")
        return
    
    print(f"\n🔤 Testing text queries...")
    print("-" * 30)
    
    similarities = []
    
    for query in queries:
        try:
            # Tokenize text (using 77 tokens as that's what our model expects)
            tokens = tokenizer(
                query, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=77, 
                truncation=True
            )["input_ids"].numpy().astype(np.int32)
            
            # Get text embedding
            text_result = text_model.predict({'prompt': tokens})
            text_embedding = text_result['embOutput'][0]  # First output
            
            # Calculate similarity
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((query, similarity))
            
            print(f"'{query:15}' → {similarity:.4f}")
            
        except Exception as e:
            print(f"✗ Failed to process '{query}': {e}")
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top matches:")
    print("-" * 30)
    for i, (query, sim) in enumerate(similarities[:3]):
        print(f"{i+1}. '{query}' (similarity: {sim:.4f})")
    
    # Check if cat-related queries are at the top
    top_query = similarities[0][0]
    if 'cat' in top_query.lower() or 'kitten' in top_query.lower():
        print(f"\n🎉 SUCCESS! The model correctly identified the image contains a cat!")
        print(f"   Best match: '{top_query}' with similarity {similarities[0][1]:.4f}")
    else:
        print(f"\n⚠️  The model's top match was '{top_query}', not cat-related")
        cat_queries = [s for s in similarities if 'cat' in s[0].lower() or 'kitten' in s[0].lower()]
        if cat_queries:
            best_cat = cat_queries[0]
            print(f"   Best cat match: '{best_cat[0]}' with similarity {best_cat[1]:.4f}")

if __name__ == "__main__":
    test_cat_identification()