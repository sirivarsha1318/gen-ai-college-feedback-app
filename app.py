# Enhanced College Feedback Classifier with Comprehensive Facility Classification
# Improved UI and detailed facility categorization

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import os
import sys

# =============================================================================
# PYTORCH CONFIGURATION
# =============================================================================

def setup_pytorch_cpu():
    """Setup PyTorch to use CPU only to avoid GPU dependency issues"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

setup_pytorch_cpu()

# =============================================================================
# MULTIPLE FEEDBACK PROCESSING
# =============================================================================

def classify_multiple_feedback(feedback_text, use_watson=True):
    """
    Process multiple feedback items from a single input
    """
    # Split feedback by numbered points
    feedback_items = []
    
    # Split by numbered points (1. 2. 3. etc.)
    import re
    pattern = r'\d+\.\s*'
    parts = re.split(pattern, feedback_text)
    
    # Remove empty first part and clean up
    feedback_items = [item.strip() for item in parts[1:] if item.strip()]
    
    if not feedback_items:
        # If no numbered items, treat as single feedback
        feedback_items = [feedback_text.strip()]
    
    results = []
    progress_bar = st.progress(0)
    
    for i, item in enumerate(feedback_items):
        if item:
            if use_watson:
                try:
                    result = classify_feedback_ibm(item)
                except Exception as e:
                    st.warning(f"Watson ML failed for item {i+1}: {str(e)}")
                    result = classify_feedback_enhanced(item)
            else:
                result = classify_feedback_enhanced(item)
            
            results.append({
                'feedback': item,
                'item_number': i + 1,
                **result
            })
        
        progress_bar.progress((i + 1) / len(feedback_items))
    
    progress_bar.empty()
    return results

# =============================================================================
# COMPREHENSIVE FACILITY CLASSIFICATION
# =============================================================================

FACILITY_CATEGORIES = {
    'Library': {
        'keywords': ['library', 'books', 'reading room', 'study hall', 'librarian', 'digital resources', 
                    'journal', 'database', 'reference', 'periodicals', 'circulation', 'quiet zone'],
        'subcategories': ['Study Spaces', 'Book Collection', 'Digital Resources', 'Staff Service', 'Operating Hours']
    },
    'Cafeteria/Food Services': {
        'keywords': ['cafeteria', 'canteen', 'food', 'dining', 'meal', 'menu', 'nutrition', 'hygiene',
                    'kitchen', 'restaurant', 'snacks', 'beverages', 'food court', 'catering'],
        'subcategories': ['Food Quality', 'Menu Variety', 'Hygiene Standards', 'Pricing', 'Service Quality']
    },
    'Hostel/Accommodation': {
        'keywords': ['hostel', 'dormitory', 'accommodation', 'residence', 'room', 'warden', 'mess',
                    'laundry', 'security', 'maintenance', 'furniture', 'bed', 'locker'],
        'subcategories': ['Room Conditions', 'Mess Services', 'Security', 'Maintenance', 'Amenities']
    },
    'Sports/Recreation': {
        'keywords': ['gym', 'sports', 'playground', 'fitness', 'recreation', 'games', 'court',
                    'swimming', 'athletics', 'equipment', 'coach', 'tournament'],
        'subcategories': ['Sports Equipment', 'Facilities', 'Coaching', 'Events', 'Maintenance']
    },
    'Laboratory': {
        'keywords': ['lab', 'laboratory', 'equipment', 'instruments', 'experiment', 'practical',
                    'chemicals', 'safety', 'technician', 'apparatus', 'microscope', 'computer lab'],
        'subcategories': ['Equipment Quality', 'Safety Measures', 'Technical Support', 'Availability', 'Maintenance']
    },
    'IT Infrastructure': {
        'keywords': ['wifi', 'internet', 'computer', 'network', 'server', 'connectivity', 'bandwidth',
                    'software', 'hardware', 'technical support', 'login', 'portal'],
        'subcategories': ['Internet Connectivity', 'Computer Systems', 'Software', 'Technical Support', 'Network Speed']
    },
    'Campus Infrastructure': {
        'keywords': ['building', 'classroom', 'auditorium', 'parking', 'road', 'maintenance', 'cleanliness',
                    'lighting', 'ventilation', 'air conditioning', 'heating', 'elevator', 'stairs'],
        'subcategories': ['Buildings', 'Classrooms', 'Parking', 'Maintenance', 'Utilities']
    },
    'Medical/Health': {
        'keywords': ['medical', 'health', 'clinic', 'doctor', 'nurse', 'first aid', 'emergency',
                    'dispensary', 'medicine', 'treatment', 'health center'],
        'subcategories': ['Medical Staff', 'Emergency Services', 'Medicine Availability', 'Health Programs', 'Facilities']
    },
    'Transportation': {
        'keywords': ['bus', 'transport', 'shuttle', 'vehicle', 'route', 'timing', 'driver',
                    'fare', 'safety', 'punctuality', 'commute'],
        'subcategories': ['Route Coverage', 'Timing', 'Vehicle Condition', 'Safety', 'Fare Structure']
    },
    'Security': {
        'keywords': ['security', 'guard', 'safety', 'cctv', 'surveillance', 'access', 'entry',
                    'gate', 'patrol', 'emergency', 'protection'],
        'subcategories': ['Campus Security', 'Access Control', 'Surveillance', 'Emergency Response', 'Safety Measures']
    }
}

def get_all_facility_classifications():
    """Return all facility categories and subcategories"""
    return FACILITY_CATEGORIES

def classify_facility_detailed(feedback_text):
    """
    Detailed facility classification with subcategories
    """
    if not feedback_text:
        return None, None, 0.0
    
    feedback_lower = feedback_text.lower()
    scores = {}
    
    # Score each facility category
    for facility, data in FACILITY_CATEGORIES.items():
        score = sum(1 for keyword in data['keywords'] if keyword in feedback_lower)
        if score > 0:
            scores[facility] = score
    
    if not scores:
        return None, None, 0.0
    
    # Get best match
    best_facility = max(scores, key=scores.get)
    confidence = min(0.95, scores[best_facility] / (sum(scores.values()) + 1))
    
    # Determine subcategory (simplified for now)
    subcategories = FACILITY_CATEGORIES[best_facility]['subcategories']
    best_subcategory = subcategories[0]  # Default to first subcategory
    
    return best_facility, best_subcategory, confidence

# =============================================================================
# IBM WATSONX INTEGRATION
# =============================================================================

def classify_feedback_ibm(feedback_text):
    """Enhanced IBM WatsonX classification with facility details"""
    try:
        setup_pytorch_cpu()
        
        from ibm_watson_machine_learning.foundation_models import Model
        from ibm_watson_machine_learning import APIClient
        
        if 'ibm' not in st.secrets:
            st.error("IBM Watson credentials not found in secrets!")
            return classify_feedback_enhanced(feedback_text)
        
        credentials = {
            "url": st.secrets["ibm"]["endpoint_url"],
            "apikey": st.secrets["ibm"]["api_key"]
        }
        
        client = APIClient(credentials)
        client.set.default_project(st.secrets["ibm"]["project_id"])
        
        model_id = st.session_state.get('model_choice', 'meta-llama/llama-2-70b-chat')
        
        try:
            model = Model(
                model_id=model_id,
                credentials=credentials,
                project_id=st.secrets["ibm"]["project_id"]
            )
        except Exception:
            model = Model(
                model_id="google/flan-t5-xxl",
                credentials=credentials,
                project_id=st.secrets["ibm"]["project_id"]
            )
        
        prompt = generate_enhanced_prompt(feedback_text)
        
        # Updated parameters for Watson ML API
        generation_params = {
            "max_new_tokens": st.session_state.get('max_tokens', 100),
            "temperature": st.session_state.get('temperature', 0.1),
            "top_p": 1.0,
            "repetition_penalty": 1.0
        }
        
        response = model.generate_text(prompt=prompt, params=generation_params)
        result = extract_enhanced_category(response)
        
        return result
        
    except Exception as e:
        st.warning(f"Watson ML error: {str(e)}")
        return classify_feedback_enhanced(feedback_text)

def generate_enhanced_prompt(feedback_text):
    """Generate enhanced prompt with facility classifications"""
    prompt = f"""You are an advanced feedback classifier for educational institutions. Classify the following feedback into one of these main categories with subcategories:

ACADEMICS:
- Course Content
- Teaching Quality  
- Assessment Methods
- Faculty Performance

FACILITIES:
- Library (Study Spaces, Book Collection, Digital Resources, Staff Service, Operating Hours)
- Cafeteria/Food Services (Food Quality, Menu Variety, Hygiene Standards, Pricing, Service Quality)
- Hostel/Accommodation (Room Conditions, Mess Services, Security, Maintenance, Amenities)
- Sports/Recreation (Sports Equipment, Facilities, Coaching, Events, Maintenance)
- Laboratory (Equipment Quality, Safety Measures, Technical Support, Availability, Maintenance)
- IT Infrastructure (Internet Connectivity, Computer Systems, Software, Technical Support, Network Speed)
- Campus Infrastructure (Buildings, Classrooms, Parking, Maintenance, Utilities)
- Medical/Health (Medical Staff, Emergency Services, Medicine Availability, Health Programs, Facilities)
- Transportation (Route Coverage, Timing, Vehicle Condition, Safety, Fare Structure)
- Security (Campus Security, Access Control, Surveillance, Emergency Response, Safety Measures)

ADMINISTRATION:
- Registration Process
- Financial Services
- Student Support
- Administrative Staff

Examples:
Input: "The library needs more study spaces and better WiFi"
Output: Facilities - Library - Study Spaces

Input: "Cafeteria food quality has improved but prices are high"
Output: Facilities - Cafeteria/Food Services - Food Quality

Input: "Course curriculum needs updating with industry standards"
Output: Academics - Course Content

Now classify this feedback:
Input: "{feedback_text}"
Output: """
    
    return prompt

def extract_enhanced_category(response_text):
    """Extract enhanced category information from response"""
    if not response_text:
        return classify_feedback_enhanced("")
    
    response_lower = response_text.lower().strip()
    
    # Try to extract structured response
    if ' - ' in response_text:
        parts = response_text.split(' - ')
        if len(parts) >= 2:
            main_category = parts[0].strip().title()
            if len(parts) >= 3:
                subcategory = parts[1].strip()
                detail = parts[2].strip()
                return {
                    'main_category': main_category,
                    'subcategory': subcategory,
                    'detail': detail,
                    'confidence': 0.9
                }
            else:
                return {
                    'main_category': main_category,
                    'subcategory': parts[1].strip(),
                    'detail': None,
                    'confidence': 0.85
                }
    
    # Fallback to enhanced classification
    return classify_feedback_enhanced(response_text)

# =============================================================================
# ENHANCED RULE-BASED CLASSIFIER
# =============================================================================

def classify_feedback_enhanced(feedback_text):
    """Enhanced rule-based classifier with detailed categorization"""
    if not feedback_text or not feedback_text.strip():
        return {
            'main_category': 'General',
            'subcategory': None,
            'detail': None,
            'confidence': 0.5
        }
    
    feedback_lower = feedback_text.lower()
    
    # Check if it's about facilities first
    facility_result = classify_facility_detailed(feedback_text)
    if facility_result[0]:  # If facility detected
        return {
            'main_category': 'Facilities',
            'subcategory': facility_result[0],
            'detail': facility_result[1],
            'confidence': facility_result[2]
        }
    
    # Academic keywords
    academic_keywords = {
        'Course Content': ['curriculum', 'syllabus', 'content', 'material', 'textbook', 'subject'],
        'Teaching Quality': ['professor', 'teacher', 'instructor', 'teaching', 'lecture', 'explanation'],
        'Assessment Methods': ['exam', 'test', 'assignment', 'quiz', 'grading', 'evaluation'],
        'Faculty Performance': ['faculty', 'staff', 'performance', 'knowledge', 'experience']
    }
    
    # Administration keywords
    admin_keywords = {
        'Registration Process': ['registration', 'enrollment', 'admission', 'application'],
        'Financial Services': ['fee', 'payment', 'financial', 'scholarship', 'billing'],
        'Student Support': ['support', 'counseling', 'guidance', 'help', 'assistance'],
        'Administrative Staff': ['office', 'staff', 'administration', 'service', 'bureaucracy']
    }
    
    # Score academic categories
    academic_scores = {}
    for category, keywords in academic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in feedback_lower)
        if score > 0:
            academic_scores[category] = score
    
    # Score admin categories
    admin_scores = {}
    for category, keywords in admin_keywords.items():
        score = sum(1 for keyword in keywords if keyword in feedback_lower)
        if score > 0:
            admin_scores[category] = score
    
    # Determine best match
    if academic_scores:
        best_academic = max(academic_scores, key=academic_scores.get)
        academic_confidence = academic_scores[best_academic] / (sum(academic_scores.values()) + 1)
    else:
        best_academic = None
        academic_confidence = 0
    
    if admin_scores:
        best_admin = max(admin_scores, key=admin_scores.get)
        admin_confidence = admin_scores[best_admin] / (sum(admin_scores.values()) + 1)
    else:
        best_admin = None
        admin_confidence = 0
    
    # Return best match
    if academic_confidence > admin_confidence and academic_confidence > 0:
        return {
            'main_category': 'Academics',
            'subcategory': best_academic,
            'detail': None,
            'confidence': min(0.9, academic_confidence + 0.6)
        }
    elif admin_confidence > 0:
        return {
            'main_category': 'Administration',
            'subcategory': best_admin,
            'detail': None,
            'confidence': min(0.9, admin_confidence + 0.6)
        }
    else:
        return {
            'main_category': 'General',
            'subcategory': None,
            'detail': None,
            'confidence': 0.5
        }

# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_classification_result(result):
    """Display classification result with enhanced UI"""
    if isinstance(result, dict):
        main_cat = result.get('main_category', 'Unknown')
        sub_cat = result.get('subcategory', 'N/A')
        detail = result.get('detail', 'N/A')
        confidence = result.get('confidence', 0.0)
    else:
        main_cat, confidence = result
        sub_cat = 'N/A'
        detail = 'N/A'
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Main Category", main_cat)
    
    with col2:
        st.metric("Subcategory", sub_cat if sub_cat != 'N/A' else '-')
    
    with col3:
        st.metric("Detail", detail if detail != 'N/A' else '-')
    
    with col4:
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Color-coded confidence indicator
    if confidence >= 0.8:
        st.success(f"🎯 **High Confidence Classification**")
    elif confidence >= 0.6:
        st.warning(f"⚠️ **Medium Confidence Classification**")
    else:
        st.error(f"❓ **Low Confidence Classification**")

def show_facility_categories():
    """Display all available facility categories"""
    st.markdown("### 🏢 Available Facility Categories")
    
    for facility, data in FACILITY_CATEGORIES.items():
        with st.expander(f"📍 {facility}"):
            st.markdown("**Subcategories:**")
            for subcat in data['subcategories']:
                st.write(f"• {subcat}")
            
            st.markdown("**Keywords:**")
            keywords_text = ", ".join(data['keywords'][:10])  # Show first 10 keywords
            if len(data['keywords']) > 10:
                keywords_text += f" ... (+{len(data['keywords']) - 10} more)"
            st.write(keywords_text)

def create_enhanced_dashboard(results_df):
    """Create enhanced analytics dashboard"""
    if results_df.empty:
        st.info("No data to display. Process feedback first.")
        return
    
    # Main category distribution
    main_cat_counts = results_df['main_category'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=main_cat_counts.values,
            names=main_cat_counts.index,
            title="Main Category Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig = px.histogram(
            results_df,
            x='confidence',
            title="Confidence Score Distribution",
            nbins=15
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Facilities breakdown (if any)
    facilities_df = results_df[results_df['main_category'] == 'Facilities']
    if not facilities_df.empty:
        st.markdown("### 🏢 Facilities Breakdown")
        
        facility_counts = facilities_df['subcategory'].value_counts()
        
        fig = px.bar(
            x=facility_counts.values,
            y=facility_counts.index,
            orientation='h',
            title="Facility Categories"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="College Feedback Classifier",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎓 College Feedback Classifier</h1>
        <p style="font-size: 18px; color: #666;">
            Advanced AI-powered classification system for educational feedback
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        use_watson = st.toggle("Use IBM Watson ML", value=True)
        
        st.markdown("---")
        
        # Model settings
        with st.expander("🤖 Model Settings"):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
            max_tokens = st.number_input("Max Tokens", 50, 500, 100)
            
            st.session_state.update({
                'temperature': temperature,
                'max_tokens': max_tokens
            })
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        if 'classification_history' not in st.session_state:
            st.session_state.classification_history = []
        
        total_classifications = len(st.session_state.classification_history)
        st.metric("Total Classifications", total_classifications)
        
        if total_classifications > 0:
            recent_categories = [h['main_category'] for h in st.session_state.classification_history[-10:]]
            most_common = Counter(recent_categories).most_common(1)[0][0]
            st.metric("Most Recent Category", most_common)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single/Multiple Classification", 
        "📊 Batch Processing", 
        "🏢 Facility Categories",
        "📈 Analytics"
    ])
    
    with tab1:
        st.markdown("### Enter Feedback for Classification")
        st.info("💡 **Tip:** You can enter multiple feedback items using numbered points (1. 2. 3. etc.) for batch classification!")
        
        # Input area
        feedback_input = st.text_area(
            "Student Feedback:",
            placeholder="""Enter feedback here... 

Examples:
- Single: "The library WiFi is very slow and needs improvement"
- Multiple: 
  1. The professors are helpful but need better teaching methods
  2. Cafeteria food quality is poor and unhygienic
  3. Campus WiFi connectivity issues in dormitories""",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
        
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            if clear_btn:
                st.rerun()
        
        if classify_btn and feedback_input.strip():
            with st.spinner("Analyzing feedback..."):
                # Check if input contains multiple numbered items
                if re.search(r'\d+\.\s*', feedback_input):
                    # Multiple feedback items
                    results = classify_multiple_feedback(feedback_input, use_watson)
                    
                    st.markdown("### 📊 Multiple Feedback Classification Results")
                    st.success(f"✅ Processed {len(results)} feedback items")
                    
                    # Display each result
                    for result in results:
                        with st.expander(f"📝 Item {result['item_number']}: {result['feedback'][:50]}..."):
                            display_classification_result(result)
                            
                            # Store in history
                            st.session_state.classification_history.append({
                                'feedback': result['feedback'],
                                'timestamp': datetime.now(),
                                'main_category': result['main_category'],
                                'subcategory': result.get('subcategory'),
                                'detail': result.get('detail'),
                                'confidence': result['confidence']
                            })
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary Statistics")
                    categories = [r['main_category'] for r in results]
                    category_counts = Counter(categories)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Items", len(results))
                    
                    with col2:
                        avg_conf = np.mean([r['confidence'] for r in results])
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    with col3:
                        most_common = category_counts.most_common(1)[0][0]
                        st.metric("Most Common", most_common)
                    
                    # Category breakdown
                    st.markdown("### 📊 Category Breakdown")
                    for category, count in category_counts.items():
                        percentage = (count / len(results)) * 100
                        st.write(f"**{category}:** {count} items ({percentage:.1f}%)")
                
                else:
                    # Single feedback item
                    if use_watson:
                        result = classify_feedback_ibm(feedback_input)
                    else:
                        result = classify_feedback_enhanced(feedback_input)
                    
                    # Store in history
                    st.session_state.classification_history.append({
                        'feedback': feedback_input,
                        'timestamp': datetime.now(),
                        **result
                    })
                    
                    st.markdown("### 📊 Classification Results")
                    display_classification_result(result)
                    
                    # Show facility details if it's a facility
                    if result.get('main_category') == 'Facilities' and result.get('subcategory'):
                        facility_name = result['subcategory']
                        if facility_name in FACILITY_CATEGORIES:
                            st.markdown(f"### 🏢 About {facility_name}")
                            st.info(f"**Subcategories:** {', '.join(FACILITY_CATEGORIES[facility_name]['subcategories'])}")
    
    with tab2:
        st.markdown("### Batch Processing")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should contain a column with feedback text"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                feedback_col = st.selectbox("Select feedback column:", text_columns)
                
                if st.button("🚀 Process All Feedback", type="primary"):
                    feedback_list = df[feedback_col].dropna().tolist()
                    
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, feedback in enumerate(feedback_list):
                        if use_watson:
                            result = classify_feedback_ibm(feedback)
                        else:
                            result = classify_feedback_enhanced(feedback)
                        
                        results.append({
                            'feedback': feedback,
                            **result
                        })
                        
                        progress_bar.progress((i + 1) / len(feedback_list))
                    
                    progress_bar.empty()
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    st.session_state.batch_results = results_df
                    
                    # Display summary
                    st.success(f"✅ Processed {len(results)} feedback items")
                    
                    # Quick metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Processed", len(results))
                    
                    with col2:
                        avg_conf = np.mean([r['confidence'] for r in results])
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    with col3:
                        facilities_count = sum(1 for r in results if r['main_category'] == 'Facilities')
                        st.metric("Facilities Issues", facilities_count)
                    
                    # Results preview
                    st.markdown("### 📋 Results Preview")
                    st.dataframe(results_df.head(10))
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "classification_results.csv",
                        "text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        show_facility_categories()
    
    with tab4:
        st.markdown("### 📈 Analytics Dashboard")
        
        if 'batch_results' in st.session_state:
            create_enhanced_dashboard(st.session_state.batch_results)
        elif st.session_state.classification_history:
            # Show history analytics
            history_df = pd.DataFrame(st.session_state.classification_history)
            create_enhanced_dashboard(history_df)
        else:
            st.info("📊 Process some feedback to see analytics here!")

if __name__ == "__main__":
    main()