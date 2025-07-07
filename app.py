# Enhanced College Feedback Classifier with Statistics and CSV Export
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re, os
import io

# =============================================================================
# PYTORCH SAFE SETUP (CPU ONLY)
# =============================================================================
def setup_pytorch_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
setup_pytorch_cpu()

# =============================================================================
# SIMPLIFIED FALLBACK CLASSIFIER (RULE BASED)
# =============================================================================
def classify_feedback_enhanced(feedback_text):
    if not feedback_text.strip():
        return {'main_category': 'General', 'subcategory': None, 'detail': None, 'confidence': 0.5}

    feedback_lower = feedback_text.lower()
    
    # Enhanced classification rules
    if any(word in feedback_lower for word in ['library', 'book', 'study room', 'reading']):
        return {'main_category': 'Facilities', 'subcategory': 'Library', 'detail': 'Study Spaces', 'confidence': 0.85}
    elif any(word in feedback_lower for word in ['food', 'canteen', 'mess', 'cafeteria', 'dining']):
        return {'main_category': 'Facilities', 'subcategory': 'Cafeteria', 'detail': 'Food Quality', 'confidence': 0.85}
    elif any(word in feedback_lower for word in ['wifi', 'internet', 'network', 'connection']):
        return {'main_category': 'Facilities', 'subcategory': 'IT Infrastructure', 'detail': 'Internet Connectivity', 'confidence': 0.9}
    elif any(word in feedback_lower for word in ['professor', 'teaching', 'lecturer', 'faculty', 'teacher']):
        return {'main_category': 'Academics', 'subcategory': 'Teaching Quality', 'detail': None, 'confidence': 0.85}
    elif any(word in feedback_lower for word in ['exam', 'test', 'assignment', 'grade', 'marks']):
        return {'main_category': 'Academics', 'subcategory': 'Assessment', 'detail': 'Examination', 'confidence': 0.8}
    elif any(word in feedback_lower for word in ['hostel', 'dormitory', 'accommodation', 'room']):
        return {'main_category': 'Facilities', 'subcategory': 'Accommodation', 'detail': 'Hostel Services', 'confidence': 0.8}
    elif any(word in feedback_lower for word in ['transport', 'bus', 'parking', 'travel']):
        return {'main_category': 'Facilities', 'subcategory': 'Transportation', 'detail': 'Campus Transport', 'confidence': 0.8}
    elif any(word in feedback_lower for word in ['sports', 'gym', 'playground', 'recreation']):
        return {'main_category': 'Facilities', 'subcategory': 'Sports & Recreation', 'detail': 'Athletic Facilities', 'confidence': 0.8}
    elif any(word in feedback_lower for word in ['club', 'event', 'festival', 'cultural']):
        return {'main_category': 'Student Life', 'subcategory': 'Extracurricular', 'detail': 'Cultural Activities', 'confidence': 0.8}
    elif any(word in feedback_lower for word in ['placement', 'job', 'career', 'internship']):
        return {'main_category': 'Career Services', 'subcategory': 'Placement', 'detail': 'Job Opportunities', 'confidence': 0.85}
    else:
        return {'main_category': 'General', 'subcategory': None, 'detail': None, 'confidence': 0.5}

# =============================================================================
# IBM WATSONX FALLBACK HANDLER
# =============================================================================
def classify_feedback_ibm(feedback_text):
    try:
        setup_pytorch_cpu()
        if 'ibm' not in st.secrets:
            raise RuntimeError("IBM Watson credentials missing.")
        from ibm_watson_machine_learning.foundation_models import Model
        from ibm_watson_machine_learning import APIClient
        credentials = {
            "url": st.secrets["ibm"]["endpoint_url"],
            "apikey": st.secrets["ibm"]["api_key"]
        }
        client = APIClient(credentials)
        client.set.default_project(st.secrets["ibm"]["project_id"])
        model = Model("google/flan-t5-xxl", credentials=credentials, project_id=st.secrets["ibm"]["project_id"])
        prompt = f"Classify this feedback:\n{feedback_text}\nOutput:"
        response = model.generate_text(prompt=prompt, params={"max_new_tokens": 50})
        return classify_feedback_enhanced(response)
    except Exception as e:
        st.warning(f"⚠️ IBM Watson fallback used: {e}")
        return classify_feedback_enhanced(feedback_text)

# =============================================================================
# STATISTICS AND VISUALIZATION FUNCTIONS
# =============================================================================
def create_stats_visualizations(results_df):
    """Create various statistics visualizations"""
    
    # Category distribution pie chart
    category_counts = results_df['main_category'].value_counts()
    fig_pie = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Distribution of Feedback Categories",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Subcategory bar chart
    subcategory_counts = results_df['subcategory'].value_counts().head(10)
    fig_bar = px.bar(
        x=subcategory_counts.values,
        y=subcategory_counts.index,
        orientation='h',
        title="Top 10 Subcategories",
        labels={'x': 'Count', 'y': 'Subcategory'},
        color=subcategory_counts.values,
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Confidence distribution
    fig_conf = px.histogram(
        results_df,
        x='confidence',
        nbins=20,
        title="Distribution of Classification Confidence",
        labels={'confidence': 'Confidence Score', 'count': 'Number of Feedback Items'}
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Feedback Items", len(results_df))
    with col2:
        st.metric("Unique Categories", results_df['main_category'].nunique())
    with col3:
        st.metric("Average Confidence", f"{results_df['confidence'].mean():.2f}")
    with col4:
        st.metric("High Confidence (>0.8)", len(results_df[results_df['confidence'] > 0.8]))

def create_detailed_stats_table(results_df):
    """Create detailed statistics table"""
    stats_summary = results_df.groupby(['main_category', 'subcategory']).agg({
        'confidence': ['count', 'mean', 'min', 'max'],
        'timestamp': ['min', 'max']
    }).round(3)
    
    # Flatten column names
    stats_summary.columns = ['Count', 'Avg_Confidence', 'Min_Confidence', 'Max_Confidence', 'First_Feedback', 'Last_Feedback']
    stats_summary = stats_summary.reset_index()
    
    return stats_summary

# =============================================================================
# CSV EXPORT FUNCTIONS
# =============================================================================
def prepare_csv_data(results_df):
    """Prepare data for CSV export"""
    export_df = results_df.copy()
    export_df['timestamp'] = export_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return export_df

def create_download_link(df, filename):
    """Create a download link for CSV file"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label=f"📊 Download {filename}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================
def main():
    st.set_page_config(page_title="Feedback Classifier", page_icon="🎓", layout="wide")
    st.title("🎓 Enhanced College Feedback Classifier")
    st.info("Enter feedback below. Use 1., 2., 3. for multiple feedback items.")

    # Initialize session state for storing results
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        use_watson = st.toggle("Use IBM Watson ML", value=False)
        
        st.header("📊 Statistics")
        if st.session_state.all_results:
            st.metric("Total Processed", len(st.session_state.all_results))
            if st.button("🗑️ Clear History"):
                st.session_state.all_results = []
                st.rerun()
        else:
            st.info("No feedback processed yet")

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Feedback Input")
        feedback_input = st.text_area("Enter feedback text:", height=150)
        classify_btn = st.button("🔍 Classify Feedback", type="primary")

        if classify_btn and feedback_input.strip():
            results = []
            pattern = r'\d+\.\s*'
            parts = re.split(pattern, feedback_input)
            items = [p.strip() for p in parts if p.strip()]

            with st.spinner("Processing feedback..."):
                for item in items:
                    result = classify_feedback_ibm(item) if use_watson else classify_feedback_enhanced(item)
                    timestamp = datetime.now()
                    
                    # Store result with timestamp
                    result_with_metadata = {
                        'feedback_text': item,
                        'main_category': result['main_category'],
                        'subcategory': result['subcategory'],
                        'detail': result['detail'],
                        'confidence': result['confidence'],
                        'timestamp': timestamp
                    }
                    results.append(result_with_metadata)
                    st.session_state.all_results.append(result_with_metadata)

            # Display current results
            st.subheader("🔍 Classification Results")
            for i, result in enumerate(results, start=1):
                with st.expander(f"📝 Feedback {i}", expanded=True):
                    st.write(f"**Text:** {result['feedback_text']}")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Main Category", result['main_category'])
                    with col_b:
                        st.metric("Subcategory", result['subcategory'] or "-")
                    with col_c:
                        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

    with col2:
        st.subheader("📊 Statistics Dashboard")
        
        if st.session_state.all_results:
            # Create DataFrame from all results
            results_df = pd.DataFrame(st.session_state.all_results)
            
            # Show statistics
            create_stats_visualizations(results_df)
            
            # Detailed statistics table
            with st.expander("📋 Detailed Statistics Table"):
                detailed_stats = create_detailed_stats_table(results_df)
                st.dataframe(detailed_stats, use_container_width=True)
            
            # Download section
            st.subheader("💾 Download Options")
            
            # Prepare data for export
            export_df = prepare_csv_data(results_df)
            detailed_stats_export = create_detailed_stats_table(results_df)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                create_download_link(export_df, "feedback_classifications.csv")
            with col_dl2:
                create_download_link(detailed_stats_export, "feedback_statistics.csv")
            
            # Show recent feedback
            st.subheader("🕒 Recent Feedback")
            recent_df = results_df.tail(5)[['feedback_text', 'main_category', 'subcategory', 'confidence']]
            st.dataframe(recent_df, use_container_width=True)
            
        else:
            st.info("No feedback data to display. Please classify some feedback first.")

if __name__ == "__main__":
    main()
