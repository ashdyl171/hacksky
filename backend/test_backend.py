#!/usr/bin/env python3
"""
Test script for the FastAPI backend
"""

import requests
import json
import os

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model status: {data.get('model_status')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running on http://localhost:8000?")
        return False

def test_model_info():
    """Test the model info endpoint"""
    try:
        response = requests.get("http://localhost:8000/model-info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved")
            print(f"   Classifier loaded: {data.get('classifier_loaded')}")
            print(f"   ResNet loaded: {data.get('resnet_loaded')}")
            print(f"   BERT loaded: {data.get('bert_loaded')}")
            print(f"   ASR loaded: {data.get('asr_loaded')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend")
        return False

def test_detection_endpoint():
    """Test the detection endpoint with a sample video"""
    # Check if there are any video files in the uploads directory
    uploads_dir = "../public/uploads"
    if os.path.exists(uploads_dir):
        video_files = [f for f in os.listdir(uploads_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        if video_files:
            test_file = os.path.join(uploads_dir, video_files[0])
            print(f"📹 Testing with video file: {video_files[0]}")
            
            try:
                with open(test_file, 'rb') as f:
                    files = {'file': f}
                    response = requests.post("http://localhost:8000/detect", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    print("✅ Detection test passed")
                    print(f"   Prediction: {data.get('prediction')}")
                    print(f"   Confidence: {data.get('confidence'):.2%}")
                    print(f"   Image importance: {data.get('image_importance'):.2%}")
                    print(f"   Audio importance: {data.get('audio_importance'):.2%}")
                    print(f"   Text importance: {data.get('text_importance'):.2%}")
                    return True
                else:
                    print(f"❌ Detection test failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Detection test error: {e}")
                return False
        else:
            print("⚠️  No video files found in uploads directory for testing")
            return True
    else:
        print("⚠️  Uploads directory not found")
        return True

def main():
    """Run all tests"""
    print("🧪 Testing FastAPI Backend")
    print("=" * 40)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info),
        ("Detection Endpoint", test_detection_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend setup.")

if __name__ == "__main__":
    main() 