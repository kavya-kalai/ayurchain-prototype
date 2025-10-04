# AyurChain AI Authenticator (Prototype) V2.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL_HERE)

This project is a functional web-based prototype for the AyurChain system, designed to bring trust and transparency to the Ayurvedic herbal supply chain. It uses AI-powered computer vision to instantly identify medicinal herbs from an image, helping to combat fraud and ensure product quality.



## ✨ Live Demo

You can access the live, deployed prototype here:
**[https://YOUR_APP_URL_HERE](https://YOUR_APP_URL_HERE)**

## 🚀 Features

* **AI-Powered Herb Identification:** The core feature uses a TensorFlow Lite model to identify 5 different Ayurvedic herbs from an uploaded image:
    * Tulsi
    * Mint
    * Ashwagandha
    * Shatavari
    * Brahmi
* **Conditional Quality Check:** For identified Tulsi leaves, the app uses a second AI model to perform a quality check, classifying the leaf as either "Healthy" or "Diseased" and providing actionable advice.
* **Rich Informative Results:** The app displays a detailed "profile card" for each identified herb, including its common uses, where it's found, and a link to its Wikipedia page.
* **Multi-Language Support:** The entire user interface and all results can be instantly switched between **English**, **Hindi (हिन्दी)**, and **Kannada (ಕನ್ನಡ)**.

## ⚙️ How It Works

The application follows a two-step AI workflow:

1.  **Identification:** A user uploads an image of a leaf. This image is first passed through the `herb_identifier` model to predict the species.
2.  **Quality Control:** If the predicted species is "Tulsi," a "Check Leaf Quality" button appears. If clicked, the same image is passed through the `quality_checker` model to determine its health status.
3.  **Display:** The app then dynamically displays the results and detailed information in the user's selected language.

## 🛠️ Technology Stack

* **AI / Machine Learning:**
    * Google Teachable Machine for model training.
    * TensorFlow Lite for the model format.
* **Application Framework:**
    * Streamlit for the web app interface.
* **Programming Language:**
    * Python
* **Deployment:**
    * Code hosted on GitHub.
    * App deployed and hosted on Streamlit Community Cloud.

## 📂 Project Files

* `app.py`: The main Streamlit application script.
* `requirements.txt`: Python libraries needed to run the app.
* `herb_identifier.tflite`: The 5-herb identification model.
* `labels_herb.txt`: The labels for the herb model.
* `quality_checker.tflite`: The Tulsi quality check model.
* `labels_quality.txt`: The labels for the quality model.

## 📜 Project Status

This project is a proof-of-concept prototype developed as part of a rapid 6-day sprint. It successfully demonstrates the core functionalities of the AyurChain camera authentication feature.
