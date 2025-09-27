# pyright: reportMissingModuleSource=false
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from style_utils import set_page_background
import numpy as np
from io import BytesIO

def validate_input(selected_model, inputs):
    """
    Validates user inputs based on the selected disease model.

    Args:
        selected_model (str): The name of the currently selected model.
        inputs (dict): A dictionary of the user's input values.

    Returns:
        tuple: A tuple containing a boolean (True if valid, False otherwise) and a message.
    """
    validation_rules = {
        'Diabetes Prediction': {
            'Pregnancies': {'min': 0, 'max': 4},
            'Glucose': {'min': 1, 'max': 200},
            'BloodPressure': {'min': 1, 'max': 122},
            'SkinThickness': {'min': 0, 'max': 99},
            'Insulin': {'min': 0, 'max': 846},
            'BMI': {'min': 0.1, 'max': 67.1},
            'DiabetesPedigreeFunction': {'min': 0.078, 'max': 2.42},
            'Age': {'min': 21, 'max': 81}
        },
        'Heart Disease Prediction': {
            'age': {'min': 29, 'max': 77},
            'trestbps': {'min': 94, 'max': 200},
            'chol': {'min': 126, 'max': 564},
            'thalach': {'min': 71, 'max': 202},
            'oldpeak': {'min': 0.0, 'max': 6.2}
        },
        'Parkinsons Prediction': {
            'MDVP:Fo(Hz)': {'min': 88.333, 'max': 260.105},
            'MDVP:Fhi(Hz)': {'min': 102.145, 'max': 592.03},
            'MDVP:Flo(Hz)': {'min': 65.476, 'max': 239.17},
            'MDVP:Jitter(%)': {'min': 0.00168, 'max': 0.0331},
            'MDVP:Jitter(Abs)': {'min': 0.000007, 'max': 0.00026},
            'MDVP:RAP': {'min': 0.00068, 'max': 0.022},
            'MDVP:PPQ': {'min': 0.00092, 'max': 0.0333},
            'Jitter:DDP': {'min': 0.00204, 'max': 0.0660},
            'MDVP:Shimmer': {'min': 0.00161, 'max': 0.0685},
            'MDVP:Shimmer(dB)': {'min': 0.014, 'max': 0.648},
            'Shimmer:APQ3': {'min': 0.00065, 'max': 0.0331},
            'Shimmer:APQ5': {'min': 0.00094, 'max': 0.0371},
            'MDVP:APQ': {'min': 0.00115, 'max': 0.0573},
            'Shimmer:DDA': {'min': 0.00196, 'max': 0.10278},
            'NHR': {'min': 0.00065, 'max': 0.3424},
            'HNR': {'min': 8.441, 'max': 33.047},
            'RPDE': {'min': 0.25657, 'max': 0.68516},
            'DFA': {'min': 0.57428, 'max': 0.82522},
            'spread1': {'min': -7.96498, 'max': -2.43403},
            'spread2': {'min': 0.00627, 'max': 0.45049},
            'D2': {'min': 1.42328, 'max': 3.67116},
            'PPE': {'min': 0.04439, 'max': 0.52737}
        },
        'Liver Disease Prediction': {
            'Age': {'min': 4, 'max': 90},
            'Total Bilirubin': {'min': 0.4, 'max': 75.0},
            'Direct Bilirubin': {'min': 0.1, 'max': 19.7},
            'Alkaline Phosphotase': {'min': 63, 'max': 2110},
            'Alamine Aminotransferase (ALT)': {'min': 10, 'max': 2000},
            'Aspartate Aminotransferase (AST)': {'min': 10, 'max': 1500},
            'Total Proteins': {'min': 2.7, 'max': 9.6},
            'Albumin': {'min': 0.9, 'max': 5.5},
            'Albumin and Globulin Ratio': {'min': 0.3, 'max': 2.8}
        },
        'Breast Cancer Prediction': {
            'radius_mean': {'min': 6.981, 'max': 28.11},
            'texture_mean': {'min': 9.71, 'max': 39.28},
            'perimeter_mean': {'min': 43.79, 'max': 188.5},
            'area_mean': {'min': 143.5, 'max': 2501.0},
            'smoothness_mean': {'min': 0.05263, 'max': 0.1634},
            'compactness_mean': {'min': 0.01938, 'max': 0.3454},
            'concavity_mean': {'min': 0.025, 'max': 0.4268},
            'concave points_mean': {'min': 0.0, 'max': 0.2012},
            'symmetry_mean': {'min': 0.106, 'max': 0.304},
            'fractal_dimension_mean': {'min': 0.04996, 'max': 0.09744},
            'radius_se': {'min': 0.1115, 'max': 2.873},
            'texture_se': {'min': 0.3602, 'max': 4.885},
            'perimeter_se': {'min': 0.757, 'max': 21.98},
            'area_se': {'min': 6.802, 'max': 542.2},
            'smoothness_se': {'min': 0.001713, 'max': 0.03113},
            'compactness_se': {'min': 0.002252, 'max': 0.1354},
            'concavity_se': {'min': 0.20, 'max': 0.396},
            'concave points_se': {'min': 0.05, 'max': 0.05279},
            'symmetry_se': {'min': 0.007882, 'max': 0.07895},
            'fractal_dimension_se': {'min': 0.000895, 'max': 0.02984},
            'radius_worst': {'min': 7.93, 'max': 36.04},
            'texture_worst': {'min': 12.02, 'max': 49.54},
            'perimeter_worst': {'min': 50.41, 'max': 251.2},
            'area_worst': {'min': 185.2, 'max': 4254.0},
            'smoothness_worst': {'min': 0.07117, 'max': 0.2226},
            'compactness_worst': {'min': 0.02729, 'max': 1.058},
            'concavity_worst': {'min': 0.0, 'max': 1.252},
            'concave points_worst': {'min': 0.0, 'max': 0.291},
            'symmetry_worst': {'min': 0.1565, 'max': 0.6638},
            'fractal_dimension_worst': {'min': 0.05504, 'max': 0.2075}
        }
        
    }
    rules = validation_rules.get(selected_model)
    if not rules:
        return True, "No validation rules found for this model."

    for feature, value in inputs.items():
        if feature in rules:
            min_val = rules[feature].get('min')
            max_val = rules[feature].get('max')
            if not (min_val <= value <= max_val):
                return False, f"Invalid value for '{feature}'. Please enter a value between {min_val} and {max_val}."
    
    return True, "All inputs are valid."


def get_positive_probability(model, input_data_reshaped):
    """
    Return the probability of the positive class (label 1) if available.
    Falls back to None if the model does not support predict_proba.
    """
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data_reshaped)
            if hasattr(model, 'classes_') and 1 in getattr(model, 'classes_', []):
                positive_index = list(model.classes_).index(1)
                return float(probabilities[0][positive_index])
            # Assume the second column corresponds to positive class
            return float(probabilities[0][1])
    except Exception:
        pass
    return None


def generate_multidisease_assistant_reply(user_text: str) -> str:
    """
    Enhanced rule-based responder for the Multidisease Assistant.
    Provides comprehensive guidance with 20+ predefined question patterns.
    """
    text = (user_text or '').strip().lower()
    # Initialize conversational context
    if 'assistant_context' not in st.session_state:
        st.session_state.assistant_context = {'last_topic': None}
    ctx = st.session_state.assistant_context

    # 1. Greetings and pleasantries
    if any(k in text for k in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        return "Hello! I'm your AI health assistant. I can help you with disease predictions, symptoms, lifestyle advice, and medical guidance. What would you like to know?"
    
    # 2. Thank you responses
    if any(k in text for k in ["thank", "thanks", "thank you"]):
        return "You're very welcome! I'm here to help with any health-related questions. Feel free to ask about symptoms, prevention, or use our prediction tools."
    
    # 3. Goodbye responses
    if any(k in text for k in ["bye", "goodbye", "see you", "exit"]):
        return "Take care and stay healthy! Remember to consult healthcare professionals for any serious concerns. Have a great day!"

    # 4. What can you do / capabilities
    if any(k in text for k in ["what can you do", "capabilities", "features", "what do you know", "abilities"]):
        return (
            "I can help you with:\n"
            "🔹 Disease predictions (Diabetes, Heart Disease, Parkinson's, Liver Disease, Breast Cancer)\n"
            "🔹 Symptom analysis and guidance\n"
            "🔹 Lifestyle and dietary recommendations\n"
            "🔹 Medical test interpretations\n"
            "🔹 Prevention strategies\n"
            "🔹 When to see a doctor\n"
            "Ask me anything about these topics!"
        )

    # 5. How accurate are predictions
    if any(k in text for k in ["accurate", "accuracy", "reliable", "trust", "confidence"]):
        return (
            "Our AI models are trained on medical datasets with good accuracy, but remember:\n"
            "⚠️ These are screening tools, not diagnostic tools\n"
            "⚠️ Always consult qualified healthcare professionals\n"
            "⚠️ Use predictions as guidance, not final diagnosis\n"
            "⚠️ Emergency symptoms need immediate medical attention"
        )

    # 6. Privacy and data safety
    if any(k in text for k in ["privacy", "data", "safe", "secure", "confidential"]):
        return (
            "Your privacy is important:\n"
            "🔒 Data is processed locally and securely\n"
            "🔒 No personal health data is stored permanently\n"
            "🔒 Predictions are for your reference only\n"
            "🔒 Always discuss results with your doctor"
        )

    # 7. Emergency situations
    if any(k in text for k in ["emergency", "urgent", "chest pain severe", "heart attack", "stroke", "911"]):
        return (
            "🚨 EMERGENCY SITUATIONS 🚨\n"
            "Call emergency services immediately if you have:\n"
            "• Severe chest pain or pressure\n"
            "• Difficulty breathing\n"
            "• Sudden weakness or numbness\n"
            "• Loss of consciousness\n"
            "• Severe bleeding\n"
            "Don't wait - seek immediate medical help!"
        )

    # 8. Age-related health questions
    if any(k in text for k in ["age", "elderly", "senior", "old age", "aging"]):
        return (
            "Age-related health considerations:\n"
            "👴 Regular health screenings become more important\n"
            "👴 Preventive care can catch issues early\n"
            "👴 Stay active with age-appropriate exercises\n"
            "👴 Maintain social connections for mental health\n"
            "👴 Follow medication schedules carefully\n"
            "Ask me about specific age-related conditions!"
        )

    # 9. Prevention strategies
    if any(k in text for k in ["prevent", "prevention", "avoid", "reduce risk"]):
        return (
            "General disease prevention strategies:\n"
            "🛡️ Maintain healthy weight\n"
            "🛡️ Exercise regularly (150 min/week)\n"
            "🛡️ Eat balanced, nutritious diet\n"
            "🛡️ Don't smoke, limit alcohol\n"
            "🛡️ Get adequate sleep (7-9 hours)\n"
            "🛡️ Manage stress effectively\n"
            "🛡️ Regular health check-ups\n"
            "Ask about prevention for specific diseases!"
        )

    # 10. When to see a doctor
    if any(k in text for k in ["doctor", "physician", "when to see", "medical help"]):
        return (
            "See a doctor when you have:\n"
            "🩺 Persistent symptoms lasting >2 weeks\n"
            "🩺 Sudden changes in health\n"
            "🩺 Family history of serious conditions\n"
            "🩺 Abnormal test results\n"
            "🩺 Medication side effects\n"
            "🩺 Concerns about symptoms\n"
            "Regular check-ups are also important for prevention!"
        )

    # 11. Mental health questions
    if any(k in text for k in ["mental health", "depression", "anxiety", "stress", "mood"]):
        return (
            "Mental health is crucial for overall wellbeing:\n"
            "🧠 Practice stress management techniques\n"
            "🧠 Maintain social connections\n"
            "🧠 Get regular exercise and sleep\n"
            "🧠 Consider counseling if needed\n"
            "🧠 Don't hesitate to seek professional help\n"
            "Mental health affects physical health too!"
        )

    # 12. Medication questions
    if any(k in text for k in ["medication", "medicine", "pills", "drugs", "prescription"]):
        return (
            "Medication safety tips:\n"
            "💊 Take as prescribed by your doctor\n"
            "💊 Don't skip doses or stop suddenly\n"
            "💊 Be aware of side effects\n"
            "💊 Check for drug interactions\n"
            "💊 Store medications properly\n"
            "💊 Keep updated medication list\n"
            "Always consult your pharmacist or doctor about medications!"
        )

    # 13. Family history importance
    if any(k in text for k in ["family history", "genetics", "hereditary", "inherited"]):
        return (
            "Family history is important because:\n"
            "🧬 Many diseases have genetic components\n"
            "🧬 Helps identify your risk factors\n"
            "🧬 Guides screening recommendations\n"
            "🧬 Influences prevention strategies\n"
            "🧬 Important for early detection\n"
            "Share family history with your healthcare provider!"
        )

    # 14. Exercise and fitness
    if any(k in text for k in ["exercise", "fitness", "workout", "physical activity"]):
        return (
            "Exercise benefits for disease prevention:\n"
            "💪 Reduces diabetes risk by 30-40%\n"
            "💪 Lowers heart disease risk\n"
            "💪 Helps maintain healthy weight\n"
            "💪 Improves mental health\n"
            "💪 Strengthens immune system\n"
            "💪 Aim for 150 minutes moderate activity/week\n"
            "Start slowly and build up gradually!"
        )

    # 15. Nutrition and diet
    if any(k in text for k in ["nutrition", "diet", "food", "eating", "healthy eating"]):
        return (
            "Healthy eating guidelines:\n"
            "🥗 Eat variety of fruits and vegetables\n"
            "🥗 Choose whole grains over refined\n"
            "🥗 Include lean proteins\n"
            "🥗 Limit processed foods and added sugars\n"
            "🥗 Control portion sizes\n"
            "🥗 Stay hydrated with water\n"
            "🥗 Limit sodium and saturated fats"
        )

    # 16. Sleep importance
    if any(k in text for k in ["sleep", "insomnia", "tired", "fatigue", "rest"]):
        return (
            "Sleep is vital for health:\n"
            "😴 Adults need 7-9 hours nightly\n"
            "😴 Poor sleep increases disease risk\n"
            "😴 Affects immune system function\n"
            "😴 Impacts mental health\n"
            "😴 Keep regular sleep schedule\n"
            "😴 Create comfortable sleep environment\n"
            "Chronic sleep problems need medical evaluation!"
        )

    # 17. Smoking and tobacco
    if any(k in text for k in ["smoking", "tobacco", "cigarettes", "quit smoking"]):
        return (
            "Smoking cessation benefits:\n"
            "🚭 Reduces cancer risk significantly\n"
            "🚭 Improves heart and lung health\n"
            "🚭 Benefits start within 20 minutes of quitting\n"
            "🚭 Reduces stroke and diabetes risk\n"
            "🚭 Improves circulation and immunity\n"
            "🚭 Many resources available to help quit\n"
            "It's never too late to quit smoking!"
        )

    # 18. Alcohol consumption
    if any(k in text for k in ["alcohol", "drinking", "wine", "beer", "liquor"]):
        return (
            "Alcohol and health:\n"
            "🍷 Moderate consumption: up to 1 drink/day (women), 2/day (men)\n"
            "🍷 Excessive drinking increases disease risk\n"
            "🍷 Can interact with medications\n"
            "🍷 Affects liver, heart, and brain health\n"
            "🍷 Consider alcohol-free days\n"
            "🍷 Seek help if you can't control drinking"
        )

    # 19. Weight management
    if any(k in text for k in ["weight", "obesity", "overweight", "bmi", "lose weight"]):
        return (
            "Healthy weight management:\n"
            "⚖️ BMI 18.5-24.9 is generally healthy\n"
            "⚖️ Focus on gradual, sustainable changes\n"
            "⚖️ Combine diet and exercise\n"
            "⚖️ Excess weight increases disease risk\n"
            "⚖️ Even 5-10% loss has health benefits\n"
            "⚖️ Consult healthcare provider for guidance"
        )

    # 20. Vaccination importance
    if any(k in text for k in ["vaccine", "vaccination", "immunization", "shots"]):
        return (
            "Vaccination importance:\n"
            "💉 Prevents serious infectious diseases\n"
            "💉 Protects community through herd immunity\n"
            "💉 Especially important for high-risk groups\n"
            "💉 Keep vaccination records updated\n"
            "💉 Follow recommended schedules\n"
            "💉 Discuss with healthcare provider\n"
            "Vaccines are safe and effective!"
        )

    # Lifestyle guidance (generic)
    if any(k in text for k in [
        "lifestyle", "diet", "exercise", "workout", "sleep", "stress", "smoke", "alcohol", "food", "walking", "water"
    ]):
        return (
            "General lifestyle guidance:\n"
            "- Diet: prefer whole grains, vegetables, fruits, lean proteins; limit added sugar & trans-fats.\n"
            "- Exercise: at least 150 minutes/week moderate activity (e.g., brisk walking) + 2 days strength training.\n"
            "- Sleep: aim for 7–9 hours nightly; consistent schedule.\n"
            "- Stress: practice breathing, mindfulness, or yoga 10–15 min/day.\n"
            "- Avoid smoking; keep alcohol minimal. Stay hydrated (1.5–2.5 L/day depending on climate/size).\n"
            "Ask me for disease‑specific tips like: 'diabetes lifestyle' or 'heart lifestyle'."
        )

    # Disease-specific lifestyle tips
    if any(k in text for k in ["diabetes lifestyle", "diabetic lifestyle", "diabetes diet", "lower sugar"]):
        ctx['last_topic'] = 'diabetes'
        return (
            "Diabetes lifestyle tips:\n"
            "- Diet: low glycemic index carbs, high fiber; distribute carbs evenly across meals.\n"
            "- Exercise: 30–45 min/day brisk walk or cycling; add resistance training.\n"
            "- Weight: aim for 5–7% loss if overweight.\n"
            "- Monitor: check fasting glucose/HbA1c as advised."
        )
    if any(k in text for k in ["heart lifestyle", "cardiac lifestyle", "heart diet", "lower cholesterol"]):
        ctx['last_topic'] = 'heart'
        return (
            "Heart disease lifestyle tips:\n"
            "- Diet: DASH/Mediterranean style; reduce salt (<5g/day), saturated fat, and processed foods.\n"
            "- Exercise: 150–300 min/week cardio + 2 sessions strength.\n"
            "- Stop smoking, limit alcohol; manage blood pressure, lipids, and blood sugar."
        )
    if any(k in text for k in ["liver lifestyle", "liver diet", "fatty liver", "hepatitis diet"]):
        ctx['last_topic'] = 'liver'
        return (
            "Liver health tips:\n"
            "- Diet: balanced, avoid excess fructose/sugary drinks; adequate protein; limit alcohol.\n"
            "- Exercise & weight: gradual weight loss if overweight (5–10%).\n"
            "- Med safety: avoid unnecessary hepatotoxic meds; vaccinate for Hep A/B per guidance."
        )
    if any(k in text for k in ["parkinson lifestyle", "parkinsons lifestyle", "parkinson diet", "parkinson exercise"]):
        ctx['last_topic'] = 'parkinsons'
        return (
            "Parkinson's lifestyle tips:\n"
            "- Regular physiotherapy, balance & flexibility exercises; speech therapy if hypophonia.\n"
            "- Mediterranean-style diet; adequate hydration & fiber (manage constipation).\n"
            "- Structure daily routine; optimize sleep hygiene."
        )
    if any(k in text for k in ["breast cancer lifestyle", "cancer lifestyle", "oncology lifestyle"]):
        ctx['last_topic'] = 'cancer'
        return (
            "Breast cancer risk reduction:\n"
            "- Maintain healthy BMI, regular exercise, limit alcohol, avoid smoking.\n"
            "- Follow age‑appropriate screening schedules (mammography)."
        )

    # Recommended tests/intents
    if any(k in text for k in ["recommended test", "which tests", "what tests", "diagnostic test", "screening"]):
        topic = ctx.get('last_topic')
        if topic == 'diabetes':
            return (
                "Recommended tests for diabetes: Fasting Plasma Glucose, HbA1c, Oral Glucose Tolerance Test (as indicated), lipid profile, kidney function, urine microalbumin."
            )
        if topic == 'heart':
            return (
                "Cardiac tests: ECG, echocardiogram, lipid profile, fasting glucose/HbA1c, treadmill test or imaging stress test (as advised), cardiac enzymes if acute symptoms."
            )
        if topic == 'parkinsons':
            return (
                "Parkinson's is clinical; supportive tests may include DAT-SPECT (where available). Screen for reversible mimics (thyroid, B12). Neurology assessment recommended."
            )
        if topic == 'liver':
            return (
                "Liver tests: LFTs (ALT, AST, ALP, bilirubin, albumin), ultrasound, viral hepatitis panel, INR; FibroScan/CT based on clinician judgment."
            )
        if topic == 'cancer':
            return (
                "Breast cancer screening: mammography per age/risk, ultrasound/MRI as indicated; biopsy for definitive diagnosis under oncology guidance."
            )
        return "Tell me which condition (diabetes/heart/parkinsons/liver/breast cancer) and I'll suggest appropriate tests."

    # Symptom-based hints
    if any(k in text for k in ["chest pain", "shortness of breath", "angina", "palpitations"]):
        ctx['last_topic'] = 'heart'
        return (
            "Chest-related symptoms could be cardiac. You can try the Heart Disease Prediction page and also consult a cardiologist if symptoms persist or are severe."
        )
    if any(k in text for k in ["tremor", "stiffness", "slow movement", "bradykinesia"]):
        ctx['last_topic'] = 'parkinsons'
        return (
            "Those symptoms may relate to Parkinson's. Use the Parkinson's Prediction page for guidance and consider seeing a neurologist."
        )
    if any(k in text for k in ["jaundice", "yellow eyes", "liver pain", "abdominal pain", "bilirubin high"]):
        ctx['last_topic'] = 'liver'
        return (
            "These signs could involve liver function. Try the Liver Disease Prediction and consult a hepatologist for evaluation."
        )
    if any(k in text for k in ["high sugar", "hyperglycemia", "frequent urination", "excessive thirst"]):
        ctx['last_topic'] = 'diabetes'
        return (
            "These may be related to diabetes. Use the Diabetes Prediction page and follow up with a clinician."
        )
    if any(k in text for k in ["breast lump", "breast pain", "mammogram", "oncology"]):
        ctx['last_topic'] = 'cancer'
        return (
            "Consider the Breast Cancer Prediction page for feature guidance. Please consult an oncologist for proper screening."
        )
    if not text:
        return "Hi! I can help with diabetes, heart disease, Parkinson's, liver disease, and breast cancer predictions. Ask me about inputs, data ranges, or interpreting results."

    if any(k in text for k in ["diabetes", "sugar", "glucose"]):
        ctx['last_topic'] = 'diabetes'
        return (
            "For Diabetes Prediction, provide: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age. "
            "Interpretation: higher probability suggests increased diabetes risk. Always confirm with a clinician."
        )
    if any(k in text for k in ["heart", "cardio", "cardiac"]):
        ctx['last_topic'] = 'heart'
        return (
            "For Heart Disease Prediction, provide: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, "
            "Exercise-induced angina, Oldpeak, Slope, CA, Thal. Interpretation: positive prediction → consult a cardiologist."
        )
    if any(k in text for k in ["parkinson", "tremor", "parkinsons"]):
        ctx['last_topic'] = 'parkinsons'
        return (
            "For Parkinson's Prediction, provide acoustic features like MDVP Fo/Fhi/Flo, jitter/shimmer metrics, NHR, HNR, RPDE, DFA, spread1/2, D2, PPE. "
            "Interpretation: model output is supportive, not diagnostic—see a neurologist."
        )
    if any(k in text for k in ["liver", "hepat", "bilirubin"]):
        ctx['last_topic'] = 'liver'
        return (
            "For Liver Disease Prediction, provide: Age, Gender, Total/Direct Bilirubin, Alkaline Phosphatase, ALT, AST, Total Proteins, Albumin, A/G Ratio. "
            "Abnormal enzymes and high probabilities warrant hepatology consult."
        )
    if any(k in text for k in ["cancer", "breast", "malignan", "benign"]):
        ctx['last_topic'] = 'cancer'
        return (
            "For Breast Cancer Prediction, provide features like radius/texture/perimeter/area means and SEs, compactness/concavity, symmetry, fractal dimension. "
            "Screening results should be confirmed by imaging and pathology—consult an oncologist."
        )

    if any(k in text for k in ["hospital", "hospitals", "near", "nearby", "emergency", "clinic"]):
        location_hint = st.session_state.get('user_location', '').strip()
        if location_hint:
            return (
                f"Here are nearby hospitals for '{location_hint}'. Open this link: "
                f"https://www.google.com/maps/search/hospitals+near+{location_hint.replace(' ', '+')}"
            )
        return (
            "To find nearby hospitals, enter your location above and click Find Hospitals. "
            "You can also ask like: 'nearest hospitals in Mumbai'."
        )

    if any(k in text for k in ["help", "how", "guide", "input", "range", "tips", "advice"]):
        topic = ctx.get('last_topic')
        if topic == 'diabetes':
            return (
                "Diabetes inputs: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age. Keep values within suggested ranges on the page."
            )
        if topic == 'heart':
            return (
                "Heart inputs: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, FBS, Resting ECG, Max HR, Exercise Angina, Oldpeak, Slope, CA, Thal."
            )
        if topic == 'parkinsons':
            return (
                "Parkinson's inputs: MDVP Fo/Fhi/Flo, jitter/shimmer family, NHR, HNR, RPDE, DFA, spread1/2, D2, PPE."
            )
        if topic == 'liver':
            return (
                "Liver inputs: Age, Gender, Total/Direct Bilirubin, Alkaline Phosphatase, ALT, AST, Total Proteins, Albumin, A/G Ratio."
            )
        if topic == 'cancer':
            return (
                "Breast Cancer inputs: radius/texture/perimeter/area means and SEs, smoothness/compactness/concavity, concave points, symmetry, fractal dimension."
            )
        return (
            "You can open a disease page from the sidebar, fill inputs within suggested ranges, and click the Test Result button. "
            "I'll explain inputs and how to read probabilities." 
        )

    # Default fallback with gentle guidance
    return (
        "I can help with inputs, ranges, and interpretation for diabetes, heart, Parkinson's, liver disease, and breast cancer. "
        "Ask about a specific disease, your symptoms, nearby hospitals, or say 'help'."
    )



def generate_result_pdf(title: str, user_email: str, inputs: dict, diagnosis_text: str) -> bytes:
    """
    Generate a simple PDF with login info, inputs, and prediction result.
    Uses reportlab if available; otherwise returns a basic text PDF-like fallback.
    """
    buffer = BytesIO()
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore[reportMissingModuleSource]
        from reportlab.pdfgen import canvas  # type: ignore[reportMissingModuleSource]
        from reportlab.lib.units import cm  # type: ignore[reportMissingModuleSource]
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 2*cm
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, y, title)
        y -= 1*cm
        c.setFont("Helvetica", 11)
        c.drawString(2*cm, y, f"User: {user_email}")
        y -= 0.7*cm
        c.drawString(2*cm, y, "Inputs:")
        y -= 0.6*cm
        for k, v in inputs.items():
            line = f"- {k}: {v}"
            c.drawString(2.5*cm, y, line[:95])
            y -= 0.5*cm
            if y < 3*cm:
                c.showPage(); y = height - 2*cm
        y -= 0.3*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Result:")
        y -= 0.6*cm
        c.setFont("Helvetica", 11)
        for segment in [diagnosis_text[i:i+95] for i in range(0, len(diagnosis_text), 95)]:
            c.drawString(2.5*cm, y, segment)
            y -= 0.5*cm
            if y < 3*cm:
                c.showPage(); y = height - 2*cm
        c.showPage()
        c.save()
        pdf = buffer.getvalue()
        buffer.close()
        return pdf
    except Exception:
        # Fallback: simple text as bytes with PDF mime; not a true PDF but ensures user can download
        content = f"{title}\nUser: {user_email}\n\nInputs:\n" + "\n".join([f"- {k}: {v}" for k, v in inputs.items()]) + f"\n\nResult:\n{diagnosis_text}\n"
        buffer.write(content.encode('utf-8'))
        return buffer.getvalue()


def render_performance_card(title: str, accuracy: float, precision: float, recall: float, f1: float, roc_auc: float | None, cm: 'np.ndarray|None' = None, roc_fpr: 'list[float]|None' = None, roc_tpr: 'list[float]|None' = None):
    """
    Renders a styled performance card with metrics and lightweight visualizations
    using Streamlit built-ins (no extra dependencies).
    """
    import numpy as np
    import pandas as pd

    st.markdown(
        f"""
        <div style='padding:18px;border-radius:16px;background:linear-gradient(135deg,#2b3340,#1f2530);box-shadow:0 10px 30px rgba(0,0,0,0.25);margin:18px 0;'>
          <h3 style='color:#ffffff;margin-top:0;margin-bottom:10px;'>{title}</h3>
          <div style='color:#dfe6ee;line-height:1.6;'>
            <b>Accuracy</b>: {accuracy*100:.2f}% &nbsp;&nbsp;
            <b>Recall</b>: {recall*100:.2f}% &nbsp;&nbsp;
            <b>Precision</b>: {precision*100:.2f}% &nbsp;&nbsp;
            <b>F1 Score</b>: {f1*100:.2f}% &nbsp;&nbsp;
            <b>ROC-AUC</b>: {('-' if roc_auc is None else f"{roc_auc*100:.2f}%")}
          </div>
          <div style='height:8px'></div>
          <h4 style='color:#ffffff;margin-top:8px;margin-bottom:8px;'>Visualizations</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cm, col_roc = st.columns(2)

    with col_cm:
        # If actual confusion matrix is not provided, synthesize a small illustrative one
        if cm is None:
            total = 20
            tp = max(0, min(total, int(round(f1 * total / 2))))
            tn = max(0, min(total, int(round(accuracy * total)) - tp))
            fp = max(0, 10 - tp)
            fn = max(0, 10 - tn)
            cm = np.array([[tn, fp], [fn, tp]])

        # Preferred: Altair heatmap (works without matplotlib)
        try:
            import altair as alt
            import pandas as _pd_cm

            cm_df = _pd_cm.DataFrame({
                'Actual': ['Actual 0', 'Actual 0', 'Actual 1', 'Actual 1'],
                'Predicted': ['Predicted 0', 'Predicted 1', 'Predicted 0', 'Predicted 1'],
                'Count': [int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])]
            })

            base = alt.Chart(cm_df).encode(
                x=alt.X('Predicted:N', title='Predicted Label', sort=['Predicted 0', 'Predicted 1']),
                y=alt.Y('Actual:N', title='Actual Label', sort=['Actual 0', 'Actual 1'])
            )

            heatmap = base.mark_rect().encode(
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'))
            )

            text = base.mark_text(fontWeight='bold').encode(
                text='Count:Q',
                color=alt.condition(alt.datum.Count > (max(cm_df['Count'])/2), alt.value('white'), alt.value('black'))
            )

            chart = (heatmap + text).properties(title='Confusion Matrix', height=260)
            st.altair_chart(chart, use_container_width=True)

            # Numeric table removed per request; showing heatmap only
        except Exception:
            # Fallback: only the 2x2 numeric table
            import pandas as _pd_cm
            cm_simple_df = _pd_cm.DataFrame(
                [
                    [int(cm[0,0]), int(cm[0,1])],
                    [int(cm[1,0]), int(cm[1,1])],
                ],
                index=['Actual 0', 'Actual 1'],
                columns=['Predicted 0', 'Predicted 1']
            )
            st.caption('Confusion Matrix (2x2)')
            st.table(cm_simple_df)

    with col_roc:
        if roc_fpr is not None and roc_tpr is not None and len(roc_fpr) > 0 and len(roc_tpr) > 0:
            # Plot true ROC curve from FPR/TPR using Altair for consistent sizing
            import altair as alt
            import pandas as pd
            roc_df = pd.DataFrame({'FPR': roc_fpr, 'TPR': roc_tpr})
            roc_chart = alt.Chart(roc_df).mark_line().encode(
                x=alt.X('FPR:Q', title='False Positive Rate'),
                y=alt.Y('TPR:Q', title='True Positive Rate')
            ).properties(title='ROC curve', height=260)
            # Diagonal reference
            ref_df = pd.DataFrame({'FPR': [0,1], 'TPR': [0,1]})
            ref_chart = alt.Chart(ref_df).mark_rule(color='#777', strokeDash=[4,4]).encode(x='FPR:Q', y='TPR:Q')
            st.altair_chart(roc_chart + ref_chart, use_container_width=True)
        elif roc_auc is not None:
            # Fallback approximate curve from AUC
            auc = max(0.5, min(roc_auc, 0.99))
            fpr = np.array([0.0, 1 - auc, 1.0])
            tpr = np.array([0.0, auc, 1.0])
            roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
            st.caption('Approximate ROC curve')
            st.line_chart(roc_df.set_index('FPR'))
        else:
            st.info('ROC curve unavailable for this model (no probability/score output).')


def evaluate_model_on_csv(model_path: str, csv_path: str, feature_columns: list, target_column: str):
    """
    Load a pickled model and evaluate it on the given CSV.
    Returns: (accuracy, precision, recall, f1, roc_auc, confusion_matrix)
    """
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Could not load model: {model_path}. Error: {e}")
        return None


def evaluate_algorithms_on_csv(csv_path: str, feature_columns: list, target_column: str):
    """
    Train/evaluate a suite of sklearn classifiers on the given CSV to mirror
    the algorithms used in notebooks. Returns a list of metric dicts.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not load dataset: {csv_path}. Error: {e}")
        return []

    X = df[feature_columns]
    y = df[target_column]
    
    # Handle categorical variables in features
    for col in X.columns:
        if X[col].dtype == 'O':  # Object type (categorical)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Handle categorical target variable
    if y.dtype == 'O':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("Logistic Regression", make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000)) if not np.issubdtype(X.dtypes.values[0], np.number) else LogisticRegression(max_iter=1000)),
        ("Support Vector Machine", make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf'))),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
    ]

    results = []
    for name, model in models:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            # Compute ROC-AUC and ROC points if possible
            roc = None
            roc_fpr = None
            roc_tpr = None
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)[:, 1]
                    roc = roc_auc_score(y_test, proba)
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    roc_fpr = fpr.tolist()
                    roc_tpr = tpr.tolist()
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_test)
                    roc = roc_auc_score(y_test, scores)
                    fpr, tpr, _ = roc_curve(y_test, scores)
                    roc_fpr = fpr.tolist()
                    roc_tpr = tpr.tolist()
                else:
                    roc = roc_auc_score(y_test, y_pred)
            except Exception:
                pass
            cm = confusion_matrix(y_test, y_pred)
            results.append({
                'Algorithm': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC_AUC': roc,
                'ROC_FPR': roc_fpr,
                'ROC_TPR': roc_tpr,
                'CM': cm,
            })
        except Exception as e:
            st.warning(f"{name} failed: {e}")
    return results

def show_home_page():
    set_page_background()
    st.title("🏥 Multi-Disease Prediction System")
    
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Load models
    try:
        diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
        heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
        parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
        Breastcancer = pickle.load(open(f'{working_dir}/saved_models/Breastcancer.sav', 'rb'))
        liver = pickle.load(open(f'{working_dir}/saved_models/liver.sav', 'rb'))
    except Exception as e:
        st.error(f"Error loading models. Please make sure all .sav files are in the 'saved_models' directory. Error: {e}")
        return

    with st.sidebar:
        selected_label = option_menu(
            'Select Disease Prediction',
            [
                'Home',
                'Diabetes',
                'Heart Disease',
                'Parkinsons',
                'Liver Disease',
                'Breast Cancer',
                'Performance Analysis',
                'AI Assistant'
            ],
            icons=['house', 'activity', 'heart', 'person', 'capsules', 'bandaid', 'graph-up', 'chat-dots-fill'],
            default_index=0
        )

    # Map menu labels to internal keys
    menu_map = {
        'Home': 'Home',
        'Diabetes': 'Diabetes Prediction',
        'Heart Disease': 'Heart Disease Prediction',
        'Parkinsons': 'Parkinsons Prediction',
        'Liver Disease': 'Liver Disease Prediction',
        'Breast Cancer': 'Breast Cancer Prediction',
        'Performance Analysis': 'Performance Analysis',
        'AI Assistant': 'Multidisease Assistant',
    }
    selected = menu_map.get(selected_label, 'Home')
    
    if selected == 'Home':
        # Welcome page for Multi-Disease Prediction System
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
                Welcome to our comprehensive medical prediction platform. Select a disease from the sidebar to get started with predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for disease cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; margin: 1rem 0;'>
                <h3>🩸 Diabetes Prediction</h3>
                <p>Predict diabetes risk based on clinical parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; text-align: center; margin: 1rem 0;'>
                <h3>❤️ Heart Disease</h3>
                <p>Assess cardiovascular disease risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; text-align: center; margin: 1rem 0;'>
                <h3>🧠 Parkinson's</h3>
                <p>Early detection of Parkinson's disease</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; text-align: center; margin: 1rem 0;'>
                <h3>🫀 Liver Disease</h3>
                <p>Liver function assessment and prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; text-align: center; margin: 1rem 0;'>
                <h3>🎗️ Breast Cancer</h3>
                <p>Breast cancer risk evaluation</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='padding: 1.5rem; border-radius: 10px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; text-align: center; margin: 1rem 0;'>
                <h3>📊 Performance Analysis</h3>
                <p>Compare algorithm performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;'>
            <h4 style='color: #2E86AB;'>🚀 Getting Started</h4>
            <p style='color: #666;'>Use the sidebar menu to navigate to any disease prediction module. Each module provides:</p>
            <ul style='text-align: left; display: inline-block; color: #666;'>
                <li>Input forms with validation</li>
                <li>Real-time predictions</li>
                <li>Probability scores</li>
                <li>Detailed results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif selected == 'Diabetes Prediction':
        st.title('Diabetes Prediction')
        st.write('Enter the following parameters:')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
        with col2:
            glucose = st.number_input('Glucose Level', min_value=0, step=1)
        with col3:
            blood_pressure = st.number_input('Blood Pressure value', min_value=0, step=1)
        with col1:
            skin_thickness = st.number_input('Skin Thickness value', min_value=0, step=1)
        with col2:
            insulin = st.number_input('Insulin Level', min_value=0, step=1)
        with col3:
            bmi = st.number_input('BMI value', min_value=0.0, format="%.2f")
        with col1:
            dpf = st.number_input('Diabetes Pedigree Function value', min_value=0.0, format="%.3f")
        with col2:
            age = st.number_input('Age of the Person', min_value=0, step=1)
        
        diabetes_diagnosis = ''
        
        inputs_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        if st.button('Get Diabetes Test Result'):
            is_valid, message = validate_input(selected, inputs_dict)
            
            if is_valid:
                input_data = [
                    pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, dpf, age
                ]
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                
                prediction = diabetes_model.predict(input_data_reshaped)
                proba = get_positive_probability(diabetes_model, input_data_reshaped)
                displayed_proba = proba if proba is not None else (1.0 if prediction[0] == 1 else 0.0)
                
                if prediction[0] == 1:
                    diabetes_diagnosis = f'The person is diabetic (probability: {displayed_proba*100:.1f}%). Please consult a doctor.'
                else:
                    diabetes_diagnosis = f'The person is not diabetic (probability of diabetes: {displayed_proba*100:.1f}%).'
            
                st.success(diabetes_diagnosis)
                # PDF download
                if 'user_email' in st.session_state:
                    pdf_bytes = generate_result_pdf('Diabetes Prediction', st.session_state.user_email, inputs_dict, diabetes_diagnosis)
                    st.download_button(
                        label='Download PDF Report',
                        data=pdf_bytes,
                        file_name='diabetes_report.pdf',
                        mime='application/pdf'
                        )
            else:
                st.error(message)

    elif selected == 'Heart Disease Prediction':
        st.title('Heart Disease Prediction')
        st.write('Enter the following parameters:')

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=0, step=1)
        with col2:
            sex = st.selectbox('Sex', ['Male', 'Female'])
        with col3:
            cp = st.selectbox('Chest Pain Types', ['Type 0', 'Type 1', 'Type 2', 'Type 3'])
        with col1:
            trestbps = st.number_input('Resting Blood Pressure', min_value=0, step=1)
        with col2:
            chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, step=1)
        with col3:
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
        with col1:
            restecg = st.selectbox('Resting ECG Results', ['Type 0', 'Type 1', 'Type 2'])
        with col2:
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, step=1)
        with col3:
            exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        with col1:
            oldpeak = st.number_input('Oldpeak', min_value=0.0, format="%.2f")
        with col2:
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Slope 0', 'Slope 1', 'Slope 2'])
        with col3:
            ca = st.number_input('Major Vessels Colored by Flourosopy', min_value=0, max_value=4, step=1)
        with col1:
            thal = st.selectbox('Thal', ['Type 0', 'Type 1', 'Type 2', 'Type 3'])

        heart_diagnosis = ''
        
        sex_num = 1 if sex == 'Male' else 0
        cp_num = int(cp.split(' ')[1])
        fbs_num = 1 if fbs == 'True' else 0
        restecg_num = int(restecg.split(' ')[1])
        exang_num = 1 if exang == 'Yes' else 0
        slope_num = int(slope.split(' ')[1])
        thal_num = int(thal.split(' ')[1])

        inputs_dict = {
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak
        }

        if st.button('Get Heart Disease Test Result'):
            is_valid, message = validate_input(selected, inputs_dict)
            if is_valid:
                input_data = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                
                prediction = heart_disease_model.predict(input_data_reshaped)
                proba = get_positive_probability(heart_disease_model, input_data_reshaped)
                displayed_proba = proba if proba is not None else (1.0 if prediction[0] == 1 else 0.0)
                
                if prediction[0] == 1:
                    heart_diagnosis = f'The person is predicted to have Heart Disease (probability: {displayed_proba*100:.1f}%). Please consult a cardiologist.'
                else:
                    heart_diagnosis = f'The person is predicted not to have Heart Disease (probability of disease: {displayed_proba*100:.1f}%).'

                st.success(heart_diagnosis)
                if 'user_email' in st.session_state:
                    pdf_bytes = generate_result_pdf('Heart Disease Prediction', st.session_state.user_email, {
                        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
                    }, heart_diagnosis)
                    st.download_button('Download PDF Report', pdf_bytes, 'heart_report.pdf', 'application/pdf')
            else:
                st.error(message)

    elif selected == 'Parkinsons Prediction':
        st.title('Parkinsons Prediction')
        st.write('Enter the following parameters:')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, format="%.4f")
        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, format="%.4f")
        with col3:
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, format="%.4f")
        with col1:
            jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, format="%.5f")
        with col2:
            jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, format="%.6f")
        with col3:
            rap = st.number_input('MDVP:RAP', min_value=0.0, format="%.5f")
        with col1:
            ppq = st.number_input('MDVP:PPQ', min_value=0.0, format="%.5f")
        with col2:
            ddp = st.number_input('Jitter:DDP', min_value=0.0, format="%.5f")
        with col3:
            shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, format="%.5f")
        with col1:
            shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, format="%.5f")
        with col2:
            shimmer_apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, format="%.5f")
        with col3:
            shimmer_apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, format="%.5f")
        with col1:
            apq = st.number_input('MDVP:APQ', min_value=0.0, format="%.5f")
        with col2:
            shimmer_dda = st.number_input('Shimmer:DDA', min_value=0.0, format="%.5f")
        with col3:
            nhr = st.number_input('NHR', min_value=0.0, format="%.5f")
        with col1:
            hnr = st.number_input('HNR', min_value=0.0, format="%.4f")
        with col2:
            rpde = st.number_input('RPDE', min_value=0.0, format="%.5f")
        with col3:
            dfa = st.number_input('DFA', min_value=0.0, format="%.5f")
        with col1:
            spread1 = st.number_input('spread1', min_value=-10.0, format="%.6f")
        with col2:
            spread2 = st.number_input('spread2', min_value=0.0, format="%.6f")
        with col3:
            d2 = st.number_input('D2', min_value=0.0, format="%.6f")
        with col1:
            ppe = st.number_input('PPE', min_value=0.0, format="%.6f")
        
        parkinsons_diagnosis = ''
        
        inputs_dict = {
            'MDVP:Fo(Hz)': fo, 'MDVP:Fhi(Hz)': fhi, 'MDVP:Flo(Hz)': flo,
            'MDVP:Jitter(%)': jitter_percent, 'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap, 'MDVP:PPQ': ppq, 'Jitter:DDP': ddp,
            'MDVP:Shimmer': shimmer, 'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': shimmer_apq3, 'Shimmer:APQ5': shimmer_apq5,
            'MDVP:APQ': apq, 'Shimmer:DDA': shimmer_dda, 'NHR': nhr,
            'HNR': hnr, 'RPDE': rpde, 'DFA': dfa, 'spread1': spread1,
            'spread2': spread2, 'D2': d2, 'PPE': ppe
        }
        
        if st.button('Get Parkinsons Test Result'):
            is_valid, message = validate_input(selected, inputs_dict)
            if is_valid:
                input_data = [
                    fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                    shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, apq,
                    shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
                ]
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                
                prediction = parkinsons_model.predict(input_data_reshaped)
                proba = get_positive_probability(parkinsons_model, input_data_reshaped)
                displayed_proba = proba if proba is not None else (1.0 if prediction[0] == 1 else 0.0)
                
                if prediction[0] == 1:
                    parkinsons_diagnosis = f"The person is predicted to have Parkinson's Disease (probability: {displayed_proba*100:.1f}%). Please consult a neurologist."
                else:
                    parkinsons_diagnosis = f"The person is predicted not to have Parkinson's Disease (probability of disease: {displayed_proba*100:.1f}%)."
                
                st.success(parkinsons_diagnosis)
                if 'user_email' in st.session_state:
                    pdf_bytes = generate_result_pdf("Parkinson's Prediction", st.session_state.user_email, inputs_dict, parkinsons_diagnosis)
                    st.download_button('Download PDF Report', pdf_bytes, 'parkinsons_report.pdf', 'application/pdf')
            else:
                st.error(message)

    elif selected == 'Liver Disease Prediction':
        st.title('Liver Disease Prediction')
        st.write('Enter the following parameters:')

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', min_value=0, step=1)
        with col2:
            gender = st.selectbox('Gender', ['Male', 'Female'])
        with col3:
            tb = st.number_input('Total Bilirubin', min_value=0.0, format="%.2f")
        with col1:
            db = st.number_input('Direct Bilirubin', min_value=0.0, format="%.2f")
        with col2:
            alkphos = st.number_input('Alkaline Phosphotase', min_value=0)
        with col3:
            sgpt = st.number_input('Alamine Aminotransferase (ALT)', min_value=0)
        with col1:
            sgot = st.number_input('Aspartate Aminotransferase (AST)', min_value=0)
        with col2:
            tp = st.number_input('Total Proteins', min_value=0.0, format="%.2f")
        with col3:
            alb = st.number_input('Albumin', min_value=0.0, format="%.2f")
        with col1:
            ag_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, format="%.2f")

        liver_diagnosis = ''
        
        gender_num = 1 if gender == 'Male' else 0

        inputs_dict = {
            'Age': age,
            'Total Bilirubin': tb,
            'Direct Bilirubin': db,
            'Alkaline Phosphotase': alkphos,
            'Alamine Aminotransferase (ALT)': sgpt,
            'Aspartate Aminotransferase (AST)': sgot,
            'Total Proteins': tp,
            'Albumin': alb,
            'Albumin and Globulin Ratio': ag_ratio
        }

        if st.button('Get Liver Disease Test Result'):
            is_valid, message = validate_input(selected, inputs_dict)
            if is_valid:
                input_data = [age, gender_num, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                
                prediction = liver.predict(input_data_reshaped)
                proba = get_positive_probability(liver, input_data_reshaped)
                displayed_proba = proba if proba is not None else (1.0 if prediction[0] == 1 else 0.0)
                
                if prediction[0] == 1:
                    liver_diagnosis = f'The person is predicted to have Liver Disease (probability: {displayed_proba*100:.1f}%). Please consult a hepatologist.'
                else:
                    liver_diagnosis = f'The person is predicted not to have Liver Disease (probability of disease: {displayed_proba*100:.1f}%).'
                
                st.success(liver_diagnosis)
                if 'user_email' in st.session_state:
                    pdf_bytes = generate_result_pdf('Liver Disease Prediction', st.session_state.user_email, {
                        'Age': age, 'Gender': gender, 'Total Bilirubin': tb, 'Direct Bilirubin': db,
                        'Alkaline Phosphotase': alkphos, 'ALT': sgpt, 'AST': sgot, 'Total Proteins': tp,
                        'Albumin': alb, 'A/G Ratio': ag_ratio
                    }, liver_diagnosis)
                    st.download_button('Download PDF Report', pdf_bytes, 'liver_report.pdf', 'application/pdf')
            else:
                st.error(message)
    
    elif selected == 'Breast Cancer Prediction':
        st.title('Breast Cancer Prediction')
        st.write('Enter the following parameters:')

        col1, col2, col3 = st.columns(3)
        with col1:
            radius_mean = st.number_input('Radius Mean', min_value=0.0, format="%.4f")
        with col2:
            texture_mean = st.number_input('Texture Mean', min_value=0.0, format="%.4f")
        with col3:
            perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, format="%.4f")
        with col1:
            area_mean = st.number_input('Area Mean', min_value=0.0, format="%.4f")
        with col2:
            smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, format="%.4f")
        with col3:
            compactness_mean = st.number_input('Compactness Mean', min_value=0.0, format="%.4f")
        with col1:
            concavity_mean = st.number_input('Concavity Mean', min_value=0.0, format="%.4f")
        with col2:
            concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, format="%.4f")
        with col3:
            symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, format="%.4f")
        with col1:
            fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, format="%.4f")
        with col2:
            radius_se = st.number_input('Radius SE', min_value=0.0, format="%.4f")
        with col3:
            texture_se = st.number_input('Texture SE', min_value=0.0, format="%.4f")
        with col1:
            perimeter_se = st.number_input('Perimeter SE', min_value=0.0, format="%.4f")
        with col2:
            area_se = st.number_input('Area SE', min_value=0.0, format="%.4f")
        with col3:
            smoothness_se = st.number_input('Smoothness SE', min_value=0.0, format="%.4f")
        with col1:
            compactness_se = st.number_input('Compactness SE', min_value=0.0, format="%.4f")
        with col2:
            concavity_se = st.number_input('Concavity SE', min_value=0.0, format="%.4f")
        with col3:
            concave_points_se = st.number_input('Concave Points SE', min_value=0.0, format="%.4f")
        with col1:
            symmetry_se = st.number_input('Symmetry SE', min_value=0.0, format="%.4f")
        with col2:
            fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0, format="%.4f")
        with col3:
            radius_worst = st.number_input('Radius Worst', min_value=0.0, format="%.4f")
        with col1:
            texture_worst = st.number_input('Texture Worst', min_value=0.0, format="%.4f")
        with col2:
            perimeter_worst = st.number_input('Perimeter Worst', min_value=0.0, format="%.4f")

        breast_cancer_diagnosis = ''
        inputs_dict = {
            'radius_mean': radius_mean, 'texture_mean': texture_mean, 'perimeter_mean': perimeter_mean,
            'area_mean': area_mean, 'smoothness_mean': smoothness_mean, 'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean, 'concave points_mean': concave_points_mean, 'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean, 'radius_se': radius_se, 'texture_se': texture_se,
            'perimeter_se': perimeter_se, 'area_se': area_se, 'smoothness_se': smoothness_se,
            'compactness_se': compactness_se, 'concavity_se': concavity_se, 'concave_points_se': concave_points_se,
            'symmetry_se': symmetry_se, 'fractal_dimension_se': fractal_dimension_se, 'radius_worst': radius_worst,
            'texture_worst': texture_worst, 'perimeter_worst': perimeter_worst
        }

        if st.button('Get Breast Cancer Test Result'):
            is_valid, message = validate_input(selected, inputs_dict)
            if is_valid:
                input_data = [
                    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                    fractal_dimension_mean, radius_se, texture_se, perimeter_se,
                    area_se, smoothness_se, compactness_se, concavity_se,
                    concave_points_se, symmetry_se, fractal_dimension_se,
                    radius_worst, texture_worst, perimeter_worst
                ]
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                
                prediction = Breastcancer.predict(input_data_reshaped)
                proba = get_positive_probability(Breastcancer, input_data_reshaped)
                displayed_proba = proba if proba is not None else (1.0 if int(prediction[0]) == 1 else 0.0)

                
                if prediction[0] == 1:
                    breast_cancer_diagnosis = f'The person is predicted to have Malignant Breast Cancer (probability: {displayed_proba*100:.1f}%). Please consult an oncologist.'
                else:
                    breast_cancer_diagnosis = f'The person is predicted to have Benign Breast Cancer (probability of malignancy: {displayed_proba*100:.1f}%).'
                
                st.success(breast_cancer_diagnosis)
                if 'user_email' in st.session_state:
                    pdf_bytes = generate_result_pdf('Breast Cancer Prediction', st.session_state.user_email, inputs_dict, breast_cancer_diagnosis)
                    st.download_button('Download PDF Report', pdf_bytes, 'breast_cancer_report.pdf', 'application/pdf')
            else:
                st.error(message)
    
    elif selected == 'Performance Analysis':
        st.title('📊 Performance Analysis')
        st.write('Compare algorithm performance across different diseases. Select a disease to view detailed performance metrics.')
        
        # Disease selection
        selected_disease = st.selectbox(
            "Select a Disease",
            ['All Diseases', 'Diabetes', 'Heart', 'Parkinsons', 'Liver', 'Cancer']
        )
        
        # Helper to aggregate results
        def render_results_list(results, header: str | None = None):
            import pandas as pd
            if header:
                st.subheader(header)
            if not results:
                st.info('No results available.')
                return
            # Show cards
            for r in results:
                render_performance_card(
                    r['Algorithm'], r['Accuracy'], r['Precision'], r['Recall'], r['F1'], r['ROC_AUC'], r.get('CM'), r.get('ROC_FPR'), r.get('ROC_TPR')
                )
            # Summary table
            df = pd.DataFrame(results).drop(columns=['CM'], errors='ignore')
            st.markdown('**Summary (this section)**')
            st.dataframe(df, use_container_width=True)

        # Show performance analysis depending on selection
        if selected_disease == 'All Diseases':
            import pandas as pd
            working_dir = os.path.dirname(os.path.abspath(__file__))

            datasets = [
                (
                    'Diabetes',
                    f"{working_dir}/dataset/diabetes.csv",
                    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                    'Outcome'
                ),
                (
                    'Heart',
                    f"{working_dir}/dataset/heart.csv",
                    ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
                    'target'
                ),
                (
                    "Parkinsons",
                    f"{working_dir}/dataset/parkinsons.csv",
                    ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE'],
                    'status'
                ),
                (
                    'Liver',
                    f"{working_dir}/dataset/indian_liver_patient.csv",
                    ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio'],
                    'Dataset'
                ),
                (
                    'Cancer',
                    f"{working_dir}/dataset/cancer.csv",
                    ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst'],
                    'diagnosis'
                ),
            ]

            per_disease_results = {}
            for disease_name, path, feats, target in datasets:
                # Breast cancer: ensure features exist
                if disease_name == 'Cancer':
                    import pandas as _pd
                    _df = _pd.read_csv(path)
                    feats = [c for c in feats if c in _df.columns]
                res = evaluate_algorithms_on_csv(path, feats, target)
                per_disease_results[disease_name] = res

            # Render each disease with a collapsible section
            for disease_name, res in per_disease_results.items():
                with st.expander(f"{disease_name} - Detailed Metrics", expanded=False):
                    render_results_list(res)

            # Compute overall averages per algorithm across diseases
            algo_rows = []
            for disease_name, res in per_disease_results.items():
                for r in res:
                    algo_rows.append({
                        'Disease': disease_name,
                        'Algorithm': r['Algorithm'],
                        'Accuracy': r['Accuracy'],
                        'Precision': r['Precision'],
                        'Recall': r['Recall'],
                        'F1': r['F1'],
                        'ROC_AUC': r['ROC_AUC'],
                    })

            if algo_rows:
                df_all = pd.DataFrame(algo_rows)
                avg_df = df_all.groupby('Algorithm')[['Accuracy','Precision','Recall','F1','ROC_AUC']].mean().reset_index()
                st.subheader('Overall Average Metrics (Across All Diseases)')
                # Convert to percentage strings for display
                disp_df = avg_df.copy()
                for col in ['Accuracy','Precision','Recall','F1','ROC_AUC']:
                    disp_df[col] = (disp_df[col] * 100).round(2).astype(str) + '%'
                st.dataframe(disp_df, use_container_width=True)

                # Optional: highlight best algorithm by F1
                best_algo = avg_df.sort_values('F1', ascending=False).iloc[0]
                st.success(f"Best overall algorithm by F1: {best_algo['Algorithm']} (F1={best_algo['F1']*100:.2f}%)")

                # Small note
                st.caption('Averages computed over the per-disease test folds used in each evaluation.')

            st.stop()
        if selected_disease == 'Diabetes':
            st.subheader("Performance Analysis - Diabetes Prediction")
            working_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = f"{working_dir}/dataset/diabetes.csv"
            features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            target = 'Outcome'
            alg_results = evaluate_algorithms_on_csv(csv_path, features, target)
            for r in alg_results:
                render_performance_card(r['Algorithm'], r['Accuracy'], r['Precision'], r['Recall'], r['F1'], r['ROC_AUC'], r['CM'], r.get('ROC_FPR'), r.get('ROC_TPR'))

        elif selected_disease == 'Heart':
            st.subheader("Performance Analysis - Heart Disease Prediction")
            working_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = f"{working_dir}/dataset/heart.csv"
            features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
            target = 'target'
            alg_results = evaluate_algorithms_on_csv(csv_path, features, target)
            for r in alg_results:
                render_performance_card(r['Algorithm'], r['Accuracy'], r['Precision'], r['Recall'], r['F1'], r['ROC_AUC'], r['CM'], r.get('ROC_FPR'), r.get('ROC_TPR'))

        elif selected_disease == 'Parkinsons':
            st.subheader("Performance Analysis - Parkinson's Prediction")
            working_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = f"{working_dir}/dataset/parkinsons.csv"
            features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']
            target = 'status'
            alg_results = evaluate_algorithms_on_csv(csv_path, features, target)
            for r in alg_results:
                render_performance_card(r['Algorithm'], r['Accuracy'], r['Precision'], r['Recall'], r['F1'], r['ROC_AUC'], r['CM'], r.get('ROC_FPR'), r.get('ROC_TPR'))

        elif selected_disease == 'Liver':
            st.subheader("Performance Analysis - Liver Disease Prediction")
            working_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = f"{working_dir}/dataset/indian_liver_patient.csv"
            features = ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']
            target = 'Dataset'
            alg_results = evaluate_algorithms_on_csv(csv_path, features, target)
            for r in alg_results:
                render_performance_card(r['Algorithm'], r['Accuracy'], r['Precision'], r['Recall'], r['F1'], r['ROC_AUC'], r['CM'], r.get('ROC_FPR'), r.get('ROC_TPR'))

        elif selected_disease == 'Cancer':
            st.subheader("Performance Analysis - Breast Cancer Prediction")
            working_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = f"{working_dir}/dataset/cancer.csv"
            features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst']
            target = 'diagnosis'
            import pandas as _pd
            _df = _pd.read_csv(csv_path)
            features = [c for c in features if c in _df.columns]
            alg_results = evaluate_algorithms_on_csv(csv_path, features, target)
            for r in alg_results:
                render_performance_card(r['Algorithm'], r['Accuracy'], r['Precision'], r['Recall'], r['F1'], r['ROC_AUC'], r['CM'], r.get('ROC_FPR'), r.get('ROC_TPR'))
    
    elif selected == 'Multidisease Assistant':
        st.title('🤖 Multidisease Assistant')
        st.write('Ask me about disease predictions, symptoms, or health guidance')
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'user_location' not in st.session_state:
            st.session_state.user_location = ''

        avatar_map = {"assistant": "🔵", "user": "🟢"}

        # Location/hospital controls removed from chatbot per request

        # Show only assistant replies for a cleaner bot-only view
        for role, content in st.session_state.chat_history:
            if role == "assistant":
                with st.chat_message("assistant", avatar=avatar_map.get("assistant")):
                    st.markdown(content)

        # Quick suggested prompts removed per request

        user_text = st.chat_input('Type your question here...')
        if user_text is not None:
            st.session_state.chat_history.append(("user", user_text))
            reply = generate_multidisease_assistant_reply(user_text)
            st.session_state.chat_history.append(("assistant", reply))
            with st.chat_message("assistant", avatar=avatar_map.get("assistant")):
                st.markdown(reply)



if __name__ == '__main__':
    show_home_page()
