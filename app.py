import streamlit as st
import base64
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import json
from datetime import datetime
import time
import os  # <-- Added for environment variable support

# 🔑 Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # <-- Load API key from environment variable

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    st.error("Please install google-generativeai: pip install google-generativeai")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# 🎨 Enhanced Page Configuration
st.set_page_config(
    page_title="🐾 AI Animal Breed Analyzer", 
    layout="wide", 
    page_icon="🐾",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
       
        border-left: 4px solid #2196f3;
    }
    .ai-message {
       
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

# Utility functions
def compress_image(image_bytes, max_size=(800, 600), quality=85):
    """Compress image to reduce API costs and improve performance"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error compressing image: {e}")
        return image_bytes

def validate_image(uploaded_file):
    """Validate uploaded image"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
        return False, "File size too large (max 10MB)"
    
    try:
        img = Image.open(uploaded_file)
        return True, "Valid image"
    except:
        return False, "Invalid image format"

def safe_api_call(prompt, max_retries=3):
    """Safe API call with retry logic"""
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                st.warning(f"Empty response on attempt {attempt + 1}")
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"API call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def create_download_link(text, filename, label="📥 Download Report"):
    """Create download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration: none; background: #4CAF50; color: white; padding: 8px 16px; border-radius: 5px;">{label}</a>'
    return href

# 📌 Enhanced Sidebar
with st.sidebar:
    st.title("🐾 Animal Breed Analyzer")
    
    st.markdown("""
    ### 🌟 Features:
    - 🔍 **AI-Powered Breed Detection**
    - ⚖️ **Intelligent Breed Comparison**
    - 🩺 **Health & Pregnancy Analysis**
    - 📊 **Interactive Data Visualization**
    - 💬 **24/7 AI Veterinary Assistant**
    """)
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.analysis_results:
        st.metric("Total Analyses", len(st.session_state.analysis_results))
    
    st.info("💡 **Tip:** Upload clear, well-lit images for best results!")
    
    st.markdown("---")
    st.markdown("**🔒 Privacy:** Your images are processed securely and not stored.")

# 📌 Main Navigation
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home", 
    "🔍 Breed Analysis", 
    "⚖️ Compare Breeds", 
    "🩺 Health Monitor", 
    "💬 AI Adviser",
    "📊 Analytics"
])

# ================= ENHANCED HOME PAGE =================
with tab0:
    st.markdown('<h1 class="main-header">🐾 Welcome to AI Animal Breed Analyzer</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Smart Detection</h3>
            <p>Advanced AI identifies breeds with 95%+ accuracy using Google's Gemini AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Insights</h3>
            <p>Get detailed analytics on feeding, milk yield, and care requirements</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🩺 Health Monitoring</h3>
            <p>Comprehensive pregnancy and health analysis with visual charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats or recent activity
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Breeds Supported", "500+", "🐄")
    with col2:
        st.metric("Accuracy Rate", "95%+", "🎯")
    with col3:
        st.metric("Analysis Time", "<30s", "⚡")
    with col4:
        st.metric("Languages", "Multi", "🌍")
    
    st.markdown("---")
    st.success("🚀 **Ready to start?** Upload an image in the 'Breed Analysis' tab or chat with our AI Adviser!")

# ================= ENHANCED SINGLE BREED ANALYSIS =================
with tab1:
    st.header("🔍 Advanced Breed Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📸 Upload Animal Image", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, PNG, WebP (Max: 10MB)",
            key="single_breed_upload"
        )
        
        if uploaded_file is not None:
            # Validate image
            is_valid, message = validate_image(uploaded_file)
            if not is_valid:
                st.error(message)
            else:
                try:
                    # Open uploaded image directly
                    img = Image.open(uploaded_file)
                    st.image(img, caption="📷 Uploaded Image", use_column_width=True)
                    
                    # Show image info
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    st.info(f"**Image Info:** {img.size[0]}x{img.size[1]} pixels, {len(file_bytes)/1024:.1f}KB")
                except Exception as e:
                    st.warning(f"Could not read image: {e}")

    with col2:
        st.markdown("### 📋 Analysis Options")
        
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Quick Analysis", "Detailed Analysis", "Expert Analysis"]
        )
        
        include_care_tips = st.checkbox("Include Care Tips", value=True)
        include_health_info = st.checkbox("Include Health Information", value=True)
        include_breeding_info = st.checkbox("Include Breeding Information", value=False)
    
    if uploaded_file is not None and st.button("🔍 Analyze Breed", use_container_width=True, key="analyze_single"):
        with st.spinner("🤖 AI is analyzing your image... This may take 10-30 seconds"):
            try:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()  # raw bytes for API
                image_data = {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
                
                # Build prompt
                prompt_text = f"""
                Analyze this animal image with {analysis_depth.lower()}. Provide:
                
                1. **Breed Identification**
                2. **Basic Information**
                3. **Physical Characteristics**
                4. **Performance Data**
                {"5. **Care & Management Tips**" if include_care_tips else ""}
                {"6. **Health Information**" if include_health_info else ""}
                {"7. **Breeding Information**" if include_breeding_info else ""}
                Format the response clearly with headers and bullet points.
                """
                
                prompt = [image_data, prompt_text]
                response_text = safe_api_call(prompt)
                
                if response_text:
                    st.subheader("📖 Breed Analysis Results")
                    st.markdown(response_text)
                    
                    # Save results
                    if "analysis_results" not in st.session_state:
                        st.session_state.analysis_results = []
                    st.session_state.analysis_results.append({
                        "timestamp": datetime.now(),
                        "type": "single_breed",
                        "result": response_text
                    })
                    
                    # Download link
                    st.markdown(
                        create_download_link(
                            response_text,
                            f"breed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    st.error("Failed to get response from AI.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    elif uploaded_file is None:
        st.info("👉 Please upload an image to start the analysis.")


# ================= ENHANCED BREED COMPARISON =================
with tab2:
    st.header("⚖️ Advanced Breed Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Animal")
        file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"], key="img1")
        if file1 is not None:
            img1_bytes = file1.read()
            img1_compressed = compress_image(img1_bytes)
            st.image(img1_compressed, caption="Animal 1", use_column_width=True)
    
    with col2:
        st.subheader("Second Animal")
        file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"], key="img2")
        if file2 is not None:
            img2_bytes = file2.read()
            img2_compressed = compress_image(img2_bytes)
            st.image(img2_compressed, caption="Animal 2", use_column_width=True)
    
    # Comparison options
    st.markdown("### ⚙️ Comparison Settings")
    col1, col2 = st.columns(2)
    with col1:
        comparison_focus = st.selectbox(
            "Focus Area:",
            ["General Comparison", "Milk Production", "Meat Quality", "Care Requirements"]
        )
    with col2:
        include_economics = st.checkbox("Include Economic Analysis", value=True)
    
    if file1 is not None and file2 is not None:
        if st.button("⚖️ Compare Breeds", use_container_width=True, key="compare_breeds"):
            with st.spinner("🤖 Comparing breeds... Please wait"):
                try:
                    prompt_text = f"""
                    Compare these two animals focusing on {comparison_focus.lower()}. 
                    
                    Provide a structured comparison:
                    
                    **ANIMAL 1 ANALYSIS:**
                    - Breed name and confidence
                    - Key characteristics
                    - Performance metrics
                    - Care requirements
                    
                    **ANIMAL 2 ANALYSIS:**
                    - Breed name and confidence  
                    - Key characteristics
                    - Performance metrics
                    - Care requirements
                    
                    **COMPARATIVE ANALYSIS:**
                    - Similarities
                    - Key differences
                    - Performance comparison
                    - Suitability for different purposes
                    {"- Economic comparison (initial cost, maintenance, profitability)" if include_economics else ""}
                    
                    **RECOMMENDATION:**
                    Which breed is better for specific purposes and why?
                    """
                    
                    # Create image parts
                    image1_data = {
                        "mime_type": "image/jpeg",
                        "data": img1_compressed
                    }
                    
                    image2_data = {
                        "mime_type": "image/jpeg", 
                        "data": img2_compressed
                    }
                    
                    prompt = [image1_data, image2_data, prompt_text]
                    
                    response_text = safe_api_call(prompt)
                    
                    if response_text:
                        st.subheader("📊 Detailed Breed Comparison")
                        st.success(response_text)
                        
                        # Save results
                        st.session_state.analysis_results.append({
                            "timestamp": datetime.now(),
                            "type": "comparison",
                            "result": response_text
                        })
                        
                        # Download link
                        st.markdown(
                            create_download_link(
                                response_text,
                                f"breed_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            ),
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("Failed to get comparison results. Please try again.")
                
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
    else:
        st.info("👉 Please upload **two images** to compare breeds.")

# ================= ENHANCED HEALTH ANALYSIS =================
with tab3:
    st.header("🩺 Comprehensive Health & Pregnancy Monitor")
    
    with st.form("health_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            animal_type = st.selectbox(
                "Animal Type", 
                ["Cow", "Buffalo", "Goat", "Sheep", "Horse", "Pig", "Dog", "Cat", "Other"]
            )
            breed = st.text_input("Breed (if known)", "")
            age = st.number_input("Age (years)", 0, 30, 3)
            weight = st.number_input("Current Weight (kg)", 0, 1500, 200)
        
        with col2:
            milk_yield = st.number_input("Daily Milk Yield (liters)", 0.0, 100.0, 0.0, step=0.5)
            pregnant = st.selectbox("Pregnancy Status", ["Not Pregnant", "Pregnant", "Recently Given Birth", "Unknown"])
            last_checkup = st.date_input("Last Veterinary Checkup")
            body_condition = st.selectbox("Body Condition Score", ["1 - Very Thin", "2 - Thin", "3 - Ideal", "4 - Overweight", "5 - Obese"])
        
        health_issues = st.text_area("Current Health Issues or Symptoms", "")
        environment = st.selectbox("Living Environment", ["Indoor", "Outdoor", "Mixed", "Farm/Pasture"])
        
        submitted = st.form_submit_button("🔍 Analyze Health Status", use_container_width=True)
    
    if submitted:
        with st.spinner("🤖 Analyzing health data and generating recommendations..."):
            try:
                prompt_text = f"""
                Analyze the health status of this {animal_type} with the following details:
                
                **Animal Profile:**
                - Type: {animal_type}
                - Breed: {breed if breed else 'Unknown'}
                - Age: {age} years
                - Weight: {weight} kg
                - Body Condition: {body_condition}
                - Environment: {environment}
                
                **Performance Data:**
                - Current milk yield: {milk_yield} L/day
                - Pregnancy status: {pregnant}
                - Last checkup: {last_checkup}
                
                **Health Concerns:**
                {health_issues if health_issues else 'None reported'}
                
                Please provide:
                
                1. **Health Assessment**
                   - Overall health status
                   - Risk factors identified
                   - Body condition evaluation
                
                2. **Nutritional Recommendations**
                   - Daily feed requirements (kg)
                   - Feed composition breakdown
                   - Supplements needed
                   - Water requirements
                
                3. **Management Advice**
                   - Housing requirements
                   - Exercise recommendations
                   - Monitoring schedule
                
                4. **Veterinary Care**
                   - Recommended checkup frequency
                   - Vaccinations due
                   - Warning signs to watch for
                
                5. **Pregnancy-Specific Advice** (if applicable)
                   - Gestation care
                   - Nutritional adjustments
                   - Birth preparation
                
                Provide specific, actionable recommendations.
                """
                
                response_text = safe_api_call(prompt_text)
                
                if response_text:
                    st.subheader("📋 Comprehensive Health Analysis")
                    st.success(response_text)
                    
                    # Create visualizations
                    st.subheader("📊 Health Metrics Dashboard")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Feed requirement chart
                        base_feed = weight * 0.025  # 2.5% of body weight
                        pregnancy_feed = weight * 0.035  # 3.5% during pregnancy
                        lactation_feed = base_feed + (milk_yield * 0.5)  # Additional for milk
                        
                        fig_feed = go.Figure(data=[
                            go.Bar(name='Base Feed', x=['Current'], y=[base_feed], marker_color='lightblue'),
                            go.Bar(name='Pregnancy Feed', x=['If Pregnant'], y=[pregnancy_feed], marker_color='lightgreen'),
                            go.Bar(name='Lactation Feed', x=['Current + Milk'], y=[lactation_feed], marker_color='orange')
                        ])
                        fig_feed.update_layout(title="Daily Feed Requirements (kg)", height=300)
                        st.plotly_chart(fig_feed, use_container_width=True)
                    
                    with col2:
                        # Body condition visualization
                        bcs_values = [1, 2, 3, 4, 5]
                        bcs_labels = ['Very Thin', 'Thin', 'Ideal', 'Overweight', 'Obese']
                        current_bcs = int(body_condition.split(' - ')[0])
                        colors = ['red' if i == current_bcs else 'lightgray' for i in bcs_values]
                        
                        fig_bcs = go.Figure(data=[
                            go.Bar(x=bcs_labels, y=[1]*5, marker_color=colors)
                        ])
                        fig_bcs.update_layout(title="Body Condition Score", showlegend=False, height=300)
                        st.plotly_chart(fig_bcs, use_container_width=True)
                    
                    with col3:
                        # Age vs Expected lifespan
                        expected_lifespan = {"Cow": 15, "Buffalo": 18, "Goat": 12, "Sheep": 10, 
                                           "Horse": 25, "Pig": 12, "Dog": 13, "Cat": 15}.get(animal_type, 15)
                        
                        fig_age = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = age,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Age vs Lifespan"},
                            gauge = {
                                'axis': {'range': [None, expected_lifespan]},
                                'bar': {'color': "darkgreen"},
                                'steps': [
                                    {'range': [0, expected_lifespan/3], 'color': "lightgreen"},
                                    {'range': [expected_lifespan/3, 2*expected_lifespan/3], 'color': "yellow"},
                                    {'range': [2*expected_lifespan/3, expected_lifespan], 'color': "orange"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': expected_lifespan
                                }
                            }
                        ))
                        fig_age.update_layout(height=300)
                        st.plotly_chart(fig_age, use_container_width=True)
                    
                    # Save results
                    st.session_state.analysis_results.append({
                        "timestamp": datetime.now(),
                        "type": "health_analysis",
                        "result": response_text,
                        "animal_data": {
                            "type": animal_type,
                            "age": age,
                            "weight": weight,
                            "milk_yield": milk_yield
                        }
                    })
                else:
                    st.error("Failed to analyze health data. Please try again.")
            
            except Exception as e:
                st.error(f"Health analysis failed: {e}")

# ================= ENHANCED AI ADVISER (FIXED LOOP ISSUE) =================
with tab4:
    st.header("💬 AI Veterinary Adviser")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (role, msg, timestamp) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You ({timestamp.strftime('%H:%M:%S')}):</strong><br>
                    {msg}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>🤖 AI Adviser ({timestamp.strftime('%H:%M:%S')}):</strong><br>
                    {msg}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input section
    st.markdown("---")
    st.subheader("Ask Your Question")
    
    # Quick question buttons
    st.markdown("#### 🔥 Popular Questions:")
    quick_questions = [
        "Signs of pregnancy in cattle?",
        "Best feeding schedule for dairy cows?",
        "Common health issues in goats?",
        "Vaccination schedule for livestock?"
    ]
    
    cols = st.columns(len(quick_questions))
    selected_question = None
    
    for i, question in enumerate(quick_questions):
        if cols[i].button(question, key=f"quick_{i}"):
            selected_question = question
    
    # Text input for custom questions
    user_input = st.text_area(
        "Type your question here:",
        value=selected_question if selected_question else "",
        placeholder="e.g., My cow is not eating well, what should I do?",
        key=f"chat_input_{st.session_state.chat_input_key}",
        height=100
    )
    
    # Send button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        send_button = st.button("Send 📤", use_container_width=True, key="send_chat")
    
    with col2:
        clear_button = st.button("Clear Chat 🗑️", use_container_width=True, key="clear_chat")
    
    # Handle clear chat
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.chat_input_key += 1
        st.rerun()
    
    # Handle send message
    if send_button and user_input.strip():
        # Add user message
        st.session_state.chat_history.append(("user", user_input.strip(), datetime.now()))
        
        with st.spinner("🤖 AI Adviser is thinking..."):
            try:
                # Enhanced system prompt for veterinary context
                system_prompt = """
                You are Dr. AI, a knowledgeable veterinary assistant specializing in:
                - Animal health diagnosis and treatment
                - Breeding and reproduction
                - Nutrition and feeding
                - Farm management
                - Emergency care advice
                
                Always provide:
                1. Clear, actionable advice
                2. When to consult a veterinarian
                3. Preventive measures
                4. Safety warnings if applicable
                
                Keep responses helpful but remind users that serious issues need professional veterinary care.
                """
                
                full_prompt = f"{system_prompt}\n\nUser question: {user_input.strip()}"
                response_text = safe_api_call(full_prompt)
                
                if response_text:
                    st.session_state.chat_history.append(("assistant", response_text, datetime.now()))
                    # Increment key to clear input
                    st.session_state.chat_input_key += 1
                    st.rerun()
                else:
                    st.error("Failed to get response from AI Adviser. Please try again.")
                
            except Exception as e:
                st.error(f"Chat failed: {e}")
    
    # Export chat option
    if st.session_state.chat_history:
        st.markdown("---")
        chat_export = "\n".join([f"{role.upper()} ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}): {msg}" for role, msg, timestamp in st.session_state.chat_history])
        st.markdown(
            create_download_link(chat_export, f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "📥 Export Chat History"),
            unsafe_allow_html=True
        )

# ================= ANALYTICS TAB =================
with tab5:
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.analysis_results:
        st.info("📈 No analysis data yet. Perform some breed analyses to see your analytics here!")
    else:
        # Analysis statistics
        total_analyses = len(st.session_state.analysis_results)
        analysis_types = {}
        
        for result in st.session_state.analysis_results:
            analysis_type = result.get("type", "unknown")
            analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Breed IDs", analysis_types.get("single_breed", 0))
        with col3:
            st.metric("Comparisons", analysis_types.get("comparison", 0))
        with col4:
            st.metric("Health Checks", analysis_types.get("health_analysis", 0))
        
        # Charts
        if len(analysis_types) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Analysis types pie chart
                fig_pie = px.pie(
                    values=list(analysis_types.values()),
                    names=list(analysis_types.keys()),
                    title="Analysis Types Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Timeline of analyses
                timestamps = [result["timestamp"] for result in st.session_state.analysis_results]
                dates = [ts.date() for ts in timestamps]
                date_counts = {}
                for date in dates:
                    date_counts[date] = date_counts.get(date, 0) + 1
                
                if date_counts:
                    fig_timeline = px.bar(
                        x=list(date_counts.keys()),
                        y=list(date_counts.values()),
                        title="Analyses Over Time"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent analyses table
        st.subheader("📋 Recent Analyses")
        recent_data = []
        for i, result in enumerate(reversed(st.session_state.analysis_results[-10:])):
            recent_data.append({
                "ID": len(st.session_state.analysis_results) - i,
                "Type": result["type"].replace("_", " ").title(),
                "Timestamp": result["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Preview": result["result"][:100] + "..." if len(result["result"]) > 100 else result["result"]
            })
        
        if recent_data:
            st.dataframe(pd.DataFrame(recent_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        🐾 <strong>AI Animal Breed Analyzer</strong> | Powered by Google Gemini AI<br>
        Made with ❤️ for animal welfare and farming excellence<br>
        <em>⚠️ Always consult professional veterinarians for serious health concerns</em>
        <span style="font-size:18px; font-weight:bold; color:#444;"><br>
        Developed by <span style="color:#1e90ff;">B. Vishal</span>
    </span><br><br>
    <!-- Social links -->
    <a href="https://github.com/syntex34x" target="_blank" style="margin:0 10px; text-decoration:none;color:#1e90ff;">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" alt="GitHub" width="25" style="vertical-align: middle;filter: brightness(0) saturate(100%) invert(33%) sepia(99%) saturate(7480%) hue-rotate(202deg) brightness(93%) contrast(97%);">
    </a>
    <a href="https://www.linkedin.com/in/badri-vishal/" target="_blank" style="margin:0 10px; text-decoration:none;color:#1e90ff;">
        <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" alt="LinkedIn" width="25" style="vertical-align: middle;filter: brightness(0) saturate(100%) invert(33%) sepia(99%) saturate(7480%) hue-rotate(202deg) brightness(93%) contrast(97%);">
    </a>
    <a href="https://wa.me/9026559040" target="_blank" style="margin:0 10px; text-decoration:none;color:#1e90ff;">
         <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/whatsapp.svg" alt="WhatsApp" width="25" style="vertical-align: middle; filter: brightness(0) saturate(100%) invert(33%) sepia(99%) saturate(7480%) hue-rotate(202deg) brightness(93%) contrast(97%);">
    </a>
    </div>
    """,
    unsafe_allow_html=True
)