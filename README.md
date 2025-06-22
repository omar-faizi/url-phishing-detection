# URL Phishing Detection Using Machine Learning

A machine learning project to detect malicious URLs (phishing, malware, defacement) using Python and Random Forest classifier.

---

## 📋 Project Overview

This project uses URL features like length, number of dots, hyphens, digits, and other simple text-based characteristics to classify URLs as **benign** or **malicious**.

The goal is to help detect phishing and malicious URLs, which is a critical task in cybersecurity for protecting users from scams and malware.

---

## 🚀 Features Extracted

- URL length  
- Number of dots (`.`)  
- Number of hyphens (`-`)  
- Number of `@` symbols  
- Number of slashes (`/`)  
- Number of digits  
- Number of letters

---

## 🛠️ How It Works

1. Load the dataset of URLs labeled benign or malicious.  
2. Convert labels to binary (`benign` or `malicious`).  
3. Extract simple URL features listed above.  
4. Train a Random Forest classifier on the features.  
5. Evaluate the model on a test set and print accuracy, confusion matrix, and classification report.

---

## 📈 Results

The model achieves strong accuracy in distinguishing malicious URLs from benign ones based on these features.

---

## ⚙️ Usage

1. Clone this repository:  
   ```bash
   git clone https://github.com/omar-faizi/url-phishing-detection.git
