
# This entire block of code should be in ONE cell, starting with %%writefile app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --- Page Configuration ---
st.set_page_config(page_title="AyurChain ", layout="centered")

# --- Translation Dictionaries ---
# UI Text Translations
UI_TEXT = {
    "English": {
        "title": "🌿 AyurChain Authenticator ",
        "description": "Upload an image of a medicinal leaf to verify its authenticity using our AI model.",
        "uploader_label": "Choose a leaf image...",
        "sidebar_about": "About AyurChain",
        "sidebar_info": "This prototype demonstrates AyurChain's camera authentication feature to prevent fraud.",
        "sidebar_model_info_title": "This AI model can identify:",
        "sidebar_warning": "This is a demonstration only. Do not use for medical decisions.",
        "lang_select": "Select Language",
        "auth_complete": "Authentication Complete!",
        "predicted_herb": "Predicted Herb",
        "confidence": "Confidence",
        "about_herb": "About",
        "uses_advantages": "Common Uses & Advantages",
        "found_in": "Where It's Found",
        "learn_more": "Learn more on Wikipedia",
        "quality_check_button": "Check Leaf Quality",
        "quality_status": "Quality Status",
        "spinner_identifying": "Identifying herb...",
        "spinner_quality": "Checking quality..."
    },
    "हिन्दी": {
        "title": "🌿 आयुर्चेन ऑथेंटिकेटर ",
        "description": "हमारे AI मॉडल का उपयोग करके किसी औषधीय पत्ते की प्रामाणिकता को सत्यापित करने के लिए उसकी एक छवि अपलोड करें।",
        "uploader_label": "पत्ते की एक छवि चुनें...",
        "sidebar_about": "आयुर्चेन के बारे में",
        "sidebar_info": "यह प्रोटोटाइप धोखाधड़ी को रोकने के लिए आयुर्चेन की कैमरा प्रमाणीकरण सुविधा को प्रदर्शित करता है।",
        "sidebar_model_info_title": "यह AI मॉडल पहचान सकता है:",
        "sidebar_warning": "यह केवल एक प्रदर्शन है। चिकित्सा निर्णयों के लिए इसका उपयोग न करें।",
        "lang_select": "भाषा चुनें",
        "auth_complete": "प्रमाणीकरण पूर्ण!",
        "predicted_herb": "अनुमानित जड़ी-बूटी",
        "confidence": "आत्मविश्वास",
        "about_herb": "के बारे में",
        "uses_advantages": "सामान्य उपयोग और लाभ",
        "found_in": "यह कहाँ पाया जाता है",
        "learn_more": "विकिपीडिया पर और जानें",
        "quality_check_button": "पत्ती की गुणवत्ता जांचें",
        "quality_status": "गुणवत्ता की स्थिति",
        "spinner_identifying": "जड़ी-बूटी की पहचान हो रही है...",
        "spinner_quality": "गुणवत्ता की जांच हो रही है..."
    },
    "ಕನ್ನಡ": {
        "title": "🌿 ಆಯುರ್ಚೈನ್ ಅಥೆಂಟಿಕೇಟರ್ ",
        "description": "ನಮ್ಮ AI ಮಾದರಿಯನ್ನು ಬಳಸಿಕೊಂಡು ಔಷಧೀಯ ಎಲೆಯ ಸತ್ಯಾಸತ್ಯತೆಯನ್ನು ಪರಿಶೀಲಿಸಲು ಅದರ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "uploader_label": "ಎಲೆಯ ಚಿತ್ರವನ್ನು ಆರಿಸಿ...",
        "sidebar_about": "ಆಯುರ್ಚೈನ್ ಬಗ್ಗೆ",
        "sidebar_info": "ಈ ಮೂಲಮಾದರಿಯು ವಂಚನೆಯನ್ನು ತಡೆಗಟ್ಟಲು ಆಯುರ್ಚೈನ್‌ನ ಕ್ಯಾಮೆರಾ ದೃಢೀಕರಣ ವೈಶಿಷ್ಟ್ಯವನ್ನು ಪ್ರದರ್ಶಿಸುತ್ತದೆ.",
        "sidebar_model_info_title": "ಈ AI ಮಾದರಿಯು ಗುರುತಿಸಬಲ್ಲದು:",
        "sidebar_warning": "ಇದು ಕೇವಲ ಪ್ರದರ್ಶನವಾಗಿದೆ. ವೈದ್ಯಕೀಯ ನಿರ್ಧಾರಗಳಿಗೆ ಬಳಸಬೇಡಿ.",
        "lang_select": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "auth_complete": "ದೃಢೀಕರಣ ಪೂರ್ಣಗೊಂಡಿದೆ!",
        "predicted_herb": "ಊಹಿಸಲಾದ ಗಿಡಮೂಲಿಕೆ",
        "confidence": "ವಿಶ್ವಾಸ",
        "about_herb": "ಬಗ್ಗೆ",
        "uses_advantages": "ಸಾಮಾನ್ಯ ಉಪಯೋಗಗಳು ಮತ್ತು ಅನುಕೂಲಗಳು",
        "found_in": "ಇದು ಎಲ್ಲಿ ಕಂಡುಬರುತ್ತದೆ",
        "learn_more": "ವಿಕಿಪೀಡಿಯಾದಲ್ಲಿ ಇನ್ನಷ್ಟು ತಿಳಿಯಿರಿ",
        "quality_check_button": "ಎಲೆಯ ಗುಣಮಟ್ಟವನ್ನು ಪರಿಶೀಲಿಸಿ",
        "quality_status": "ಗುಣಮಟ್ಟದ ಸ್ಥಿತಿ",
        "spinner_identifying": "ಗಿಡಮೂಲಿಕೆಗಳನ್ನು ಗುರುತಿಸಲಾಗುತ್ತಿದೆ...",
        "spinner_quality": "ಗುಣಮಟ್ಟವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ..."
    }
}

# Herb Data Translations
HERB_DATA = {
    "Tulsi": {
        "English": {"display_name": "Tulsi (Holy Basil)", "info": "Known as the 'Queen of Herbs', Tulsi is a sacred plant in Hinduism, revered for its medicinal properties.", "uses": "Used to treat respiratory issues, reduce stress, boost immunity, and improve skin health.", "found": "Native to the Indian subcontinent and widespread throughout Southeast Asia.", "wiki": "https://en.wikipedia.org/wiki/Ocimum_tenuiflorum"},
        "हिन्दी": {"display_name": "तुलसी", "info": "'जड़ी-बूटियों की रानी' के रूप में जानी जाने वाली तुलसी हिंदू धर्म में एक पवित्र पौधा है, जो अपने औषधीय गुणों के लिए पूजनीय है।", "uses": "श्वसन संबंधी समस्याओं के इलाज, तनाव कम करने, प्रतिरक्षा बढ़ाने और त्वचा के स्वास्थ्य में सुधार के लिए उपयोग किया जाता है।", "found": "भारतीय उपमहाद्वीप का मूल निवासी और पूरे दक्षिण पूर्व एशिया में व्यापक है।", "wiki": "https://hi.wikipedia.org/wiki/%E0%A4%A4%E0%A5%81%E0%A4%B2%E0%A4%B8%E0%A5%80"},
        "ಕನ್ನಡ": {"display_name": "ತುಳಸಿ", "info": "'ಗಿಡಮೂಲಿಕೆಗಳ ರಾಣಿ' ಎಂದು ಕರೆಯಲ್ಪಡುವ ತುಳಸಿ, ಹಿಂದೂ ಧರ್ಮದಲ್ಲಿ ಪವಿತ್ರ ಸಸ್ಯವಾಗಿದ್ದು, ಅದರ ಔಷಧೀಯ ಗುಣಗಳಿಗಾಗಿ ಪೂಜಿಸಲ್ಪಡುತ್ತದೆ.", "uses": "ಉಸಿರಾಟದ ತೊಂದರೆಗಳಿಗೆ ಚಿಕಿತ್ಸೆ ನೀಡಲು, ಒತ್ತಡವನ್ನು ಕಡಿಮೆ ಮಾಡಲು, ರೋಗನಿರೋಧಕ ಶಕ್ತಿಯನ್ನು ಹೆಚ್ಚಿಸಲು ಮತ್ತು ಚರ್ಮದ ಆರೋಗ್ಯವನ್ನು ಸುಧಾರಿಸಲು ಬಳಸಲಾಗುತ್ತದೆ.", "found": "ಭಾರತೀಯ ಉಪಖಂಡದ ಸ್ಥಳೀಯ ಮತ್ತು ಆಗ್ನೇಯ ಏಷ್ಯಾದಾದ್ಯಂತ ವ್ಯಾಪಕವಾಗಿದೆ.", "wiki": "https://kn.wikipedia.org/wiki/%E0%B2%A4%E0%B3%81%E0%B2%B3%E0%B2%B8%E0%B2%BF"}
    },
    "Mint": {
        "English": {"display_name": "Mint (Pudina)", "info": "Mint is an aromatic herb known for its refreshing flavor and cooling sensation.", "uses": "Commonly used to aid digestion, freshen breath, and relieve symptoms of the common cold.", "found": "Widespread across Europe, Asia, Africa, Australia, and North America.", "wiki": "https://en.wikipedia.org/wiki/Mentha"},
        "हिन्दी": {"display_name": "पुदीना", "info": "पुदीना एक सुगंधित जड़ी-बूटी है जो अपने ताज़गी भरे स्वाद और ठंडक के एहसास के लिए जानी जाती है।", "uses": "आमतौर पर पाचन में सहायता, सांसों को ताज़ा करने और सामान्य सर्दी के लक्षणों से राहत के लिए उपयोग किया जाता है।", "found": "यूरोप, एशिया, अफ्रीका, ऑस्ट्रेलिया और उत्तरी अमेरिका में व्यापक है।", "wiki": "https://hi.wikipedia.org/wiki/%E0%A4%AA%E0%A5%81%E0%A4%A6%E0%A5%80%E0%A4%A8%E0%A4%BE"},
        "ಕನ್ನಡ": {"display_name": "ಪುದೀನ", "info": "ಪುದೀನ ಒಂದು ಸುವಾಸನಾಯುಕ್ತ ಗಿಡಮೂಲಿಕೆಯಾಗಿದ್ದು, ಅದರ ರಿಫ್ರೆಶ್ ಪರಿಮಳ ಮತ್ತು ತಂಪಾಗಿಸುವ ಸಂವೇದನೆಗೆ ಹೆಸರುವಾಸಿಯಾಗಿದೆ.", "uses": "ಸಾಮಾನ್ಯವಾಗಿ ಜೀರ್ಣಕ್ರಿಯೆಗೆ ಸಹಾಯ ಮಾಡಲು, ಉಸಿರಾಟವನ್ನು ತಾಜಾಗೊಳಿಸಲು ಮತ್ತು ಸಾಮಾನ್ಯ ಶೀತದ ಲಕ್ಷಣಗಳನ್ನು ನಿವಾರಿಸಲು ಬಳಸಲಾಗುತ್ತದೆ.", "found": "ಯುರೋಪ್, ಏಷ್ಯಾ, ಆಫ್ರಿಕಾ, ಆಸ್ಟ್ರೇಲಿಯಾ ಮತ್ತು ಉತ್ತರ ಅಮೆರಿಕಾದಾದ್ಯಂತ ವ್ಯಾಪಕವಾಗಿದೆ.", "wiki": "https://kn.wikipedia.org/wiki/%E0%B2%AA%E0%B3%81%E0%B2%A6%E0%B3%80%E0%B2%A8"}
    },
    "Ashwagandha": {
        "English": {"display_name": "Ashwagandha", "info": "An ancient medicinal herb classified as an adaptogen, meaning it can help your body manage stress.", "uses": "Known to boost brain function, lower cortisol levels, and help fight symptoms of anxiety and depression.", "found": "Native to India, North Africa, and the Middle East.", "wiki": "https://en.wikipedia.org/wiki/Withania_somnifera"},
        "हिन्दी": {"display_name": "अश्वगंधा", "info": "एक प्राचीन औषधीय जड़ी-बूटी जिसे एडाप्टोजेन के रूप में वर्गीकृत किया गया है, जिसका अर्थ है कि यह आपके शरीर को तनाव प्रबंधन में मदद कर सकती है।", "uses": "मस्तिष्क की कार्यक्षमता बढ़ाने, कोर्टिसोल के स्तर को कम करने और चिंता और अवसाद के लक्षणों से लड़ने में मदद करने के लिए जाना जाता है।", "found": "भारत, उत्तरी अफ्रीका और मध्य पूर्व का मूल निवासी।", "wiki": "https://hi.wikipedia.org/wiki/%E0%A4%85%E0%A4%B6%E0%A5%8D%E0%A4%B5%E0%A4%97%E0%A4%A8%E0%A5%8D%E0%A4%A7%E0%A4%BE"},
        "ಕನ್ನಡ": {"display_name": "ಅಶ್ವಗಂಧ", "info": "ಒಂದು ಪ್ರಾಚೀನ ಔಷಧೀಯ ಸಸ್ಯ, ಇದನ್ನು ಅಡಾಪ್ಟೋಜೆನ್ ಎಂದು ವರ್ಗೀಕರಿಸಲಾಗಿದೆ, ಅಂದರೆ ಇದು ನಿಮ್ಮ ದೇಹಕ್ಕೆ ಒತ್ತಡವನ್ನು ನಿರ್ವಹಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.", "uses": "ಮೆದುಳಿನ ಕಾರ್ಯವನ್ನು ಹೆಚ್ಚಿಸಲು, ಕಾರ್ಟಿಸೋಲ್ ಮಟ್ಟವನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಮತ್ತು ಆತಂಕ ಮತ್ತು ಖಿನ್ನತೆಯ ಲಕ್ಷಣಗಳ ವಿರುದ್ಧ ಹೋರಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.", "found": "ಭಾರತ, ಉತ್ತರ ಆಫ್ರಿಕಾ ಮತ್ತು ಮಧ್ಯಪ್ರಾಚ್ಯದ ಸ್ಥಳೀಯ.", "wiki": "https://kn.wikipedia.org/wiki/%E0%B2%85%E0%B2%B6%E0%B3%8D%E0%B2%B5%E0%B2%97%E0%B2%82%E0%B2%A7"}
    },
    "Shatavari": {
         "English": {"display_name": "Shatavari", "info": "Known as a reproductive tonic, Shatavari is a species of asparagus common in India and the Himalayas.", "uses": "Primarily used to support the female reproductive system, boost the immune system, and act as an antioxidant.", "found": "Found in tropical and subtropical parts of India, Asia, Australia, and Africa.", "wiki": "https://en.wikipedia.org/wiki/Asparagus_racemosus"},
         "हिन्दी": {"display_name": "शतावरी", "info": "एक प्रजनन टॉनिक के रूप में जानी जाने वाली शतावरी, भारत और हिमालय में आम शतावरी की एक प्रजाति है।", "uses": "मुख्य रूप से महिला प्रजनन प्रणाली का समर्थन करने, प्रतिरक्षा प्रणाली को बढ़ावा देने और एक एंटीऑक्सिडेंट के रूप में कार्य करने के लिए उपयोग किया जाता है।", "found": "भारत, एशिया, ऑस्ट्रेलिया और अफ्रीका के उष्णकटिबंधीय और उपोष्णकटिबंधीय भागों में पाया जाता है।", "wiki": "https://hi.wikipedia.org/wiki/%E0%A4%B6%E0%A4%A4%E0%A4%BE%E0%A4%B5%E0%A4%B0%E0%A5%80"},
         "ಕನ್ನಡ": {"display_name": "ಶತಾವರಿ", "info": "ಸಂತಾನೋತ್ಪತ್ತಿ τονic ಎಂದು ಕರೆಯಲ್ಪಡುವ ಶತಾವರಿ, ಭಾರತ ಮತ್ತು ಹಿಮಾಲಯದಲ್ಲಿ ಸಾಮಾನ್ಯವಾದ ಶತಾವರಿ ಜಾತಿಯಾಗಿದೆ.", "uses": "ಮುಖ್ಯವಾಗಿ ಸ್ತ್ರೀ ಸಂತಾನೋತ್ಪತ್ತಿ ವ್ಯವಸ್ಥೆಯನ್ನು ಬೆಂಬಲಿಸಲು, ರೋಗನಿರೋಧಕ ಶಕ್ತಿಯನ್ನು ಹೆಚ್ಚಿಸಲು ಮತ್ತು ಉತ್ಕರ್ಷಣ ನಿರೋಧಕವಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸಲು ಬಳಸಲಾಗುತ್ತದೆ.", "found": "ಭಾರತ, ಏಷ್ಯಾ, ಆಸ್ಟ್ರೇಲಿಯಾ ಮತ್ತು ಆಫ್ರಿಕಾದ ಉಷ್ಣವಲಯದ ಮತ್ತು ಉಪೋಷ್ಣವಲಯದ ಭಾಗಗಳಲ್ಲಿ ಕಂಡುಬರುತ್ತದೆ.", "wiki": "https://kn.wikipedia.org/wiki/%E0%B2%B6%E0%B2%A4%E0%B2%BE%E0%B2%B5%E0%B2%B0%E0%B2%BF"}
    },
    "Brahmi": {
         "English": {"display_name": "Brahmi", "info": "A staple in traditional Ayurvedic medicine, Brahmi is a non-aromatic herb known for its benefits to the brain.", "uses": "Used to improve memory, reduce anxiety, and treat epilepsy. It has strong antioxidant properties.", "found": "Native to the wetlands of southern and Eastern India, Australia, Europe, Africa, Asia, and North and South America.", "wiki": "https://en.wikipedia.org/wiki/Bacopa_monnieri"},
         "हिन्दी": {"display_name": "ब्राह्मी", "info": "पारंपरिक आयुर्वेदिक चिकित्सा में एक प्रमुख, ब्राह्मी एक गैर-सुगंधित जड़ी-बूटी है जो मस्तिष्क के लिए अपने लाभों के लिए जानी जाती है।", "uses": "स्मृति में सुधार, चिंता कम करने और मिर्गी का इलाज करने के लिए उपयोग किया जाता है। इसमें मजबूत एंटीऑक्सीडेंट गुण होते हैं।", "found": "दक्षिणी और पूर्वी भारत, ऑस्ट्रेलिया, यूरोप, अफ्रीका, एशिया और उत्तरी और दक्षिणी अमेरिका के आर्द्रभूमियों का मूल निवासी।", "wiki": "https://hi.wikipedia.org/wiki/%E0%A4%AC%E0%A5%8D%E0%A4%B0%E0%A4%BE%E0%A4%B9%E0%A5%8D%E0%A4%AE%E0%A5%80"},
         "ಕನ್ನಡ": {"display_name": "ಬ್ರಾಹ್ಮಿ", "info": "ಸಾಂಪ್ರದಾಯಿಕ ಆಯುರ್ವೇದ ಔಷಧದಲ್ಲಿ ಪ್ರಮುಖವಾದ ಬ್ರಾಹ್ಮಿ, ಮೆದುಳಿಗೆ ಅದರ ಪ್ರಯೋಜನಗಳಿಗೆ ಹೆಸರುವಾಸಿಯಾದ ಸುವಾಸನೆಯಿಲ್ಲದ ಗಿಡಮೂಲಿಕೆಯಾಗಿದೆ.", "uses": "ನೆನಪಿನ ಶಕ್ತಿಯನ್ನು ಸುಧಾರಿಸಲು, ಆತಂಕವನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಮತ್ತು ಅಪಸ್ಮಾರಕ್ಕೆ ಚಿಕಿತ್ಸೆ ನೀಡಲು ಬಳಸಲಾಗುತ್ತದೆ. ಇದು ಪ್ರಬಲವಾದ ಉತ್ಕರ್ಷಣ ನಿರೋಧಕ ಗುಣಗಳನ್ನು ಹೊಂದಿದೆ.", "found": "ದಕ್ಷಿಣ ಮತ್ತು ಪೂರ್ವ ಭಾರತ, ಆಸ್ಟ್ರೇಲಿಯಾ, ಯುರೋಪ್, ಆಫ್ರಿಕಾ, ಏಷ್ಯಾ, ಮತ್ತು ಉತ್ತರ ಮತ್ತು ದಕ್ಷಿಣ ಅಮೆರಿಕಾದ ತೇವ ಪ್ರದೇಶಗಳ ಸ್ಥಳೀಯ.", "wiki": "https://en.wikipedia.org/wiki/Bacopa_monnieri"}
    }
}

# --- Model Loading Functions ---
@st.cache_resource
def load_herb_model():
    model = tf.lite.Interpreter(model_path='herb_identifier.tflite')
    model.allocate_tensors()
    return model

@st.cache_resource
def load_quality_model():
    model = tf.lite.Interpreter(model_path='quality_checker.tflite')
    model.allocate_tensors()
    return model

# --- Prediction Function ---
def predict(image_data, model):
    size = (224, 224)
    image = image_data.resize(size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    model.set_tensor(input_details[0]['index'], image_array)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    
    return output_data

# --- Sidebar ---
selected_language_full = st.sidebar.selectbox("Select Language / भाषा चुनें / ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ", ["English", "हिन्दी", "ಕನ್ನಡ"])
lang_key = selected_language_full.split(" ")[0]

st.sidebar.title(UI_TEXT[lang_key]["sidebar_about"])
st.sidebar.info(UI_TEXT[lang_key]["sidebar_info"])
st.sidebar.success(f"{UI_TEXT[lang_key]['sidebar_model_info_title']} {', '.join(HERB_DATA.keys())}.")
st.sidebar.warning(UI_TEXT[lang_key]["sidebar_warning"]) # <-- THIS LINE IS NOW CORRECTED

# --- Main App Interface ---
st.title(UI_TEXT[lang_key]["title"])
st.write(UI_TEXT[lang_key]["description"])

uploaded_file = st.file_uploader(UI_TEXT[lang_key]["uploader_label"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image for Authentication.', width=300)

    with st.spinner(UI_TEXT[lang_key]["spinner_identifying"]):
        herb_model = load_herb_model()
        herb_prediction = predict(image, herb_model)
        
        with open('labels_herb.txt', 'r') as f:
            herb_labels = [line.strip() for line in f.readlines()]
        
        scores = [float(f) / 255.0 for f in herb_prediction[0]]
        top_herb_index = np.argmax(scores)
        herb_label_english = herb_labels[top_herb_index]
        herb_confidence = scores[top_herb_index]

    if herb_label_english in HERB_DATA:
        herb_info = HERB_DATA[herb_label_english][lang_key]
        
        st.success(f"**{UI_TEXT[lang_key]['auth_complete']}**")
        st.metric(label=UI_TEXT[lang_key]['predicted_herb'], value=herb_info['display_name'], delta=f"{UI_TEXT[lang_key]['confidence']}: {herb_confidence:.2%}")
        
        st.subheader(f"{UI_TEXT[lang_key]['about_herb']} {herb_info['display_name']}")
        st.write(herb_info['info'])
        
        st.subheader(UI_TEXT[lang_key]['uses_advantages'])
        st.write(herb_info['uses'])
        
        st.subheader(UI_TEXT[lang_key]['found_in'])
        st.write(herb_info['found'])
        
        st.markdown(f"[{UI_TEXT[lang_key]['learn_more']}]({herb_info['wiki']})", unsafe_allow_html=True)
        
        if herb_label_english == 'Tulsi':
            if st.button(UI_TEXT[lang_key]['quality_check_button']):
                with st.spinner(UI_TEXT[lang_key]['spinner_quality']):
                    quality_model = load_quality_model()
                    quality_prediction = predict(image, quality_model)
                    
                    with open('labels_quality.txt', 'r') as f:
                        quality_labels = [line.strip() for line in f.readlines()]

                    quality_scores = [float(f) / 255.0 for f in quality_prediction[0]]
                    top_quality_index = np.argmax(quality_scores)
                    quality_label = quality_labels[top_quality_index]
                
                st.metric(label=UI_TEXT[lang_key]['quality_status'], value=quality_label)
    else:
        st.error("Could not identify a known herb from the database.")
else:
    st.info('Please upload an image to get started.')
