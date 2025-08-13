#!/usr/bin/env python3
"""
Compare CLIP models on multiple images: cat, dog, and backpack.
Tests both OpenAI CLIP-ViT-Base-32 and CLIP-ViT-Large-14 models.
"""

import os
import numpy as np
import coremltools as ct
from PIL import Image
from transformers import CLIPTokenizerFast
from pathlib import Path

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_models(model_dir):
    """Load text and image models from a directory."""
    text_model_path = Path(model_dir) / "TextEncoder.mlpackage"
    image_model_path = Path(model_dir) / "ImageEncoder.mlpackage"
    
    if not text_model_path.exists() or not image_model_path.exists():
        raise FileNotFoundError(f"Models not found in {model_dir}")
    
    text_model = ct.models.MLModel(str(text_model_path))
    image_model = ct.models.MLModel(str(image_model_path))
    
    return text_model, image_model

def test_image_with_queries(image_path, image_model, text_model, tokenizer, queries, model_name):
    """Test a single image against multiple text queries."""
    print(f"\n📸 Testing {image_path} with {model_name}")
    print("-" * 60)
    
    # Load and process image
    try:
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Get image embedding
        image_result = image_model.predict({'colorImage': image_resized})
        image_embedding = image_result['embOutput'][0]
        
    except Exception as e:
        print(f"❌ Failed to process image: {e}")
        return []
    
    # Test all queries
    similarities = []
    for query in queries:
        try:
            # Tokenize text
            tokens = tokenizer(
                query, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=77, 
                truncation=True
            )["input_ids"].numpy().astype(np.int32)
            
            # Get text embedding
            text_result = text_model.predict({'prompt': tokens})
            text_embedding = text_result['embOutput'][0]
            
            # Calculate similarity
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((query, similarity))
            
        except Exception as e:
            print(f"❌ Failed to process '{query}': {e}")
            similarities.append((query, 0.0))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    for i, (query, sim) in enumerate(similarities):
        emoji = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{emoji} {query:<20} → {sim:.4f}")
    
    return similarities

def main():
    print("🔬 CLIP Model Comparison Test")
    print("=" * 70)
    
    # Test images and their expected categories
    test_images = [
        ("images/cat.jpeg", "cat"),
        ("images/dog.jpeg", "dog"),
        ("images/backpack.jpg", "backpack")
    ]
    
    # Text queries for testing
    queries = [
        # Animals
        "a cat",
        "a dog", 
        "a bird",
        "a cute kitten",
        "a puppy",
        "an animal",
        
        # Objects
        "a backpack",
        "a bag",
        "a suitcase",
        "luggage",
        "travel gear",
        
        # Generic descriptions
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a backpack",
        "outdoor equipment",
        "a furry pet"
    ]
    
    # Models to test
    models_to_test = [
        {
            "name": "CLIP-ViT-Base-32 (OpenAI)",
            "dir": "out",
            "tokenizer_id": "openai/clip-vit-base-patch32"
        },
        {
            "name": "CLIP-ViT-Large-14 (OpenAI)", 
            "dir": "out_large",
            "tokenizer_id": "openai/clip-vit-large-patch14"
        }
    ]
    
    # Results storage
    all_results = {}
    
    for model_info in models_to_test:
        model_name = model_info["name"]
        model_dir = model_info["dir"]
        tokenizer_id = model_info["tokenizer_id"]
        
        print(f"\n🤖 Loading {model_name}")
        
        try:
            # Load models
            text_model, image_model = load_models(model_dir)
            tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_id)
            
            print(f"✅ Models loaded successfully")
            
            # Test each image
            model_results = {}
            for image_path, expected_category in test_images:
                if not Path(image_path).exists():
                    print(f"⚠️  Image not found: {image_path}")
                    continue
                
                similarities = test_image_with_queries(
                    image_path, image_model, text_model, tokenizer, queries, model_name
                )
                model_results[image_path] = similarities
            
            all_results[model_name] = model_results
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    # Summary comparison
    print(f"\n\n📊 SUMMARY COMPARISON")
    print("=" * 70)
    
    for image_path, expected_category in test_images:
        if not Path(image_path).exists():
            continue
            
        print(f"\n🖼️  {Path(image_path).name} (Expected: {expected_category})")
        print("-" * 40)
        
        for model_name in all_results:
            if image_path in all_results[model_name]:
                similarities = all_results[model_name][image_path]
                top_match = similarities[0]
                
                # Check if the top match is correct
                is_correct = expected_category.lower() in top_match[0].lower()
                status = "✅ CORRECT" if is_correct else "❌ WRONG"
                
                print(f"{model_name}:")
                print(f"  Best: '{top_match[0]}' ({top_match[1]:.4f}) {status}")
                
                # Find best match for expected category
                expected_matches = [s for s in similarities if expected_category.lower() in s[0].lower()]
                if expected_matches:
                    best_expected = expected_matches[0]
                    rank = similarities.index(best_expected) + 1
                    print(f"  '{expected_category}' rank: #{rank} ({best_expected[1]:.4f})")
    
    print(f"\n✨ Test complete! Check the detailed results above.")

if __name__ == "__main__":
    main()