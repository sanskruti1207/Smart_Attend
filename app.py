import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime, date
import matplotlib.pyplot as plt

# Set page configuration first
st.set_page_config(
    page_title="SmartAttend AI - Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, light-mode minimal aesthetics (from Oakridge reference)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* Glassmorphic/Minimal White card design */
.card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 22px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    color: #0f172a;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    border-color: #cbd5e1;
}

/* Custom premium metrics cards from reference */
.metric-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}
.metric-card.primary::before { background-color: #3b82f6; }
.metric-card.success::before { background-color: #10b981; }
.metric-card.warning::before { background-color: #f59e0b; }
.metric-card.danger::before { background-color: #ef4444; }

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -5px rgba(0, 0, 0, 0.08);
    border-color: #cbd5e1;
}

.metric-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.6rem;
    font-weight: 700;
    color: #0f172a !important;
    margin: 4px 0;
    line-height: 1.1;
}
.metric-subtitle {
    font-size: 0.75rem;
    color: #94a3b8 !important;
    margin-top: 4px;
}

.glow-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f172a !important;
    letter-spacing: -0.02em;
    margin-bottom: 24px;
    background: none;
    -webkit-text-fill-color: initial;
    text-shadow: none;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Premium Button Customization */
div.stButton > button[kind="primary"] {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 10px rgba(59, 130, 246, 0.15) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button[kind="primary"]:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 14px rgba(59, 130, 246, 0.25) !important;
}
div.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div.stButton > button[kind="secondary"]:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Webcam container aesthetics */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 4px solid #ffffff !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.08) !important;
}

/* Badges */
.badge-safe {
    background-color: #dcfce7;
    color: #15803d !important;
    border: 1px solid #bbf7d0;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
}
.badge-warning {
    background-color: #fef9c3;
    color: #a16207 !important;
    border: 1px solid #fef08a;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
}
.badge-critical {
    background-color: #fee2e2;
    color: #b91c1c !important;
    border: 1px solid #fecaca;
    padding: 5px 14px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
}

/* --- EXPLICIT VISIBILITY OVERRIDES --- */
/* Forces dark text on Streamlit headings, labels, body, and sidebar list */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label {
    color: #0f172a !important;
}
.stApp p, .stApp span, .stApp li, [data-testid="stMarkdownContainer"] {
    color: #334155 !important;
}

/* Sidebar label and navigation override */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h2 {
    color: #0f172a !important;
}

/* Text Inputs, Selectboxes, and Dropdowns override */
input[type="text"], input[type="number"], textarea, select {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}

/* Streamlit baseweb widgets (Selectbox container, Selectbox items) */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

div[data-baseweb="select"] span {
    color: #0f172a !important;
}

/* Override dropdown option text colors and backgrounds */
div[role="listbox"], div[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}

div[role="listbox"] ul li, div[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

div[role="listbox"] ul li:hover, div[data-baseweb="menu"] li:hover {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}

/* Tabs styling overrides */
button[data-baseweb="tab"] p {
    color: #64748b !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] p {
    color: #3b82f6 !important;
    font-weight: 700 !important;
}

/* Database Tables / DataFrames text visibility */
.stDataFrame td, .stDataFrame th, [data-testid="stTable"] td, [data-testid="stTable"] th {
    color: #334155 !important;
}

/* Sub-headers inside cards */
.card h3, .card h4 {
    color: #0f172a !important;
}

/* Sidebar Radio Navigation Custom Styling */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #64748b !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
    margin-top: 10px !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 8px !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease-in-out !important;
    width: 100% !important;
    margin-bottom: 0px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: #f8fafc !important;
    border-color: #cbd5e1 !important;
}

/* Selected item styling */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
    background-color: #eff6ff !important;
    border-color: #3b82f6 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {
    color: #1d4ed8 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p {
    margin: 0 !important;
    font-size: 0.95rem !important;
    color: #334155 !important;
}

/* Hide the native radio dot/circle completely */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div[role="presentation"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Import local modules
from src.database_helper import DatabaseHelper
from src.recognition.face_classifier import FaceClassifier
from src.anti_spoofing.liveness import LivenessDetector
from src.registration.collector import FaceCollector
from src.registration.manager import StudentManager
from src.attendance.attendance_marker import AttendanceMarker
from src.prediction.predictor import AttendancePredictor

# Initialize helpers in session state to avoid reloading on every rerun
if 'db_helper' not in st.session_state:
    st.session_state.db_helper = DatabaseHelper()
if 'face_classifier' not in st.session_state:
    st.session_state.face_classifier = FaceClassifier()
if 'liveness_detector' not in st.session_state:
    st.session_state.liveness_detector = LivenessDetector()
if 'face_collector' not in st.session_state:
    st.session_state.face_collector = FaceCollector(detector=st.session_state.face_classifier.detector)
if 'student_manager' not in st.session_state:
    st.session_state.student_manager = StudentManager(
        db_helper=st.session_state.db_helper,
        face_classifier=st.session_state.face_classifier
    )
if 'attendance_marker' not in st.session_state:
    st.session_state.attendance_marker = AttendanceMarker(db_helper=st.session_state.db_helper)
if 'predictor' not in st.session_state:
    st.session_state.predictor = AttendancePredictor()

# Setup UI and Authentication states
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'registering' not in st.session_state:
    st.session_state.registering = False
if 'last_marked' not in st.session_state:
    st.session_state.last_marked = ""
if 'last_unknown_time' not in st.session_state:
    st.session_state.last_unknown_time = 0.0

# ----------------- LOGIN SCREEN GATE -----------------
if not st.session_state.authenticated:
    # Minimal sidebar welcome when not logged in
    st.sidebar.markdown("<h2 style='text-align: center; color: #0f172a; font-weight:700;'>SmartAttend AI</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='height: 1px; background-color: #cbd5e1; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.sidebar.info("🔑 Please log in with your credentials to access the dashboards and features.")
    
    # Main area Login form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card' style='margin-top: 50px;'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #0f172a; margin-bottom: 24px; font-weight:700;'>SmartAttend Login</h2>", unsafe_allow_html=True)
        
        tab_teacher, tab_student = st.tabs(["👨‍🏫 Teacher Portal", "🎓 Student Portal"])
        
        with tab_teacher:
            t_user = st.text_input("Teacher Username", placeholder="Enter username", key="teacher_user_in")
            t_pass = st.text_input("Teacher Password", type="password", placeholder="Enter password", key="teacher_pass_in")
            if st.button("Login as Teacher", type="primary", key="teacher_login_btn"):
                if t_user == "teacher" and t_pass == "teacher123":
                    st.session_state.authenticated = True
                    st.session_state.user_role = "teacher"
                    st.session_state.username = "teacher"
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        with tab_student:
            s_user = st.text_input("Student Username", placeholder="Enter username", key="student_user_in")
            s_pass = st.text_input("Student Password", type="password", placeholder="Enter password", key="student_pass_in")
            if st.button("Login as Student", type="primary", key="student_login_btn"):
                if s_user == "student" and s_pass == "student123":
                    st.session_state.authenticated = True
                    st.session_state.user_role = "student"
                    st.session_state.username = "student"
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ----------------- AUTHENTICATED SIDEBAR & ROUTING -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #0f172a; font-weight:700;'>SmartAttend AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='height: 1px; background-color: #cbd5e1; margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# User Session Welcome Info & Logout
username_val = st.session_state.username.capitalize() if st.session_state.username else "Guest"
user_role_val = st.session_state.user_role.capitalize() if st.session_state.user_role else "User"

st.sidebar.markdown(f"""
<div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; margin-bottom: 15px; text-align: center;">
    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.05em;">Logged in as</div>
    <div style="font-size: 1rem; color: #0f172a; font-weight: 700; margin: 4px 0;">{username_val}</div>
    <div style="font-size: 0.8rem; color: #3b82f6; font-weight: 600;">{user_role_val} Portal</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🔓 Logout", type="secondary", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.rerun()

st.sidebar.markdown("<div style='height: 1px; background-color: #cbd5e1; margin-top: 10px; margin-bottom: 15px;'></div>", unsafe_allow_html=True)

# Role-based page routing
if st.session_state.user_role == "teacher":
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard & Analytics",
            "📝 Student Registration",
            "🗃️ Manage Student Records",
            "🔍 System Audit Logs"
        ]
    )
    # Strict gate check
    if page not in ["📊 Dashboard & Analytics", "📝 Student Registration", "🗃️ Manage Student Records", "🔍 System Audit Logs"]:
        page = "📊 Dashboard & Analytics"
else: # student
    page = st.sidebar.radio(
        "Navigation",
        [
            "🎓 Student Portal",
            "📹 Live Attendance Session"
        ]
    )
    # Strict gate check
    if page not in ["🎓 Student Portal", "📹 Live Attendance Session"]:
        page = "🎓 Student Portal"

# Folder for unknown faces
os.makedirs("unknown_faces", exist_ok=True)

# ----------------- PAGE: DASHBOARD & ANALYTICS -----------------
if page == "📊 Dashboard & Analytics":
    st.markdown("<h1 class='glow-title'>Teacher Dashboard & Analytics</h1>", unsafe_allow_html=True)
    
    # Retrieve data
    students = st.session_state.db_helper.get_all_students()
    total_students = len(students)
    
    attendance_records = st.session_state.db_helper.get_all_attendance()
    total_records = len(attendance_records)
    
    today_str = date.today().strftime("%Y-%m-%d")
    today_records = st.session_state.db_helper.get_attendance_for_date(today_str)
    present_today = sum(1 for r in today_records if r['status'] == 'Present')
    late_today = sum(1 for r in today_records if r['status'] == 'Late')
    absent_today = total_students - (present_today + late_today)
    
    # Key Performance Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card primary">
            <div class="metric-title">Total Registered</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-subtitle">Active Student Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        attendance_rate = (present_today + late_today) / total_students * 100 if total_students > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card success">
            <div class="metric-title">Attendance Rate</div>
            <div class="metric-value">{attendance_rate:.1f}%</div>
            <div class="metric-subtitle">Today's Punctuality Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card warning">
            <div class="metric-title">Late Arrivals</div>
            <div class="metric-value">{late_today}</div>
            <div class="metric-subtitle">Arrived Post 9:35 AM Today</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card danger">
            <div class="metric-title">Absentees</div>
            <div class="metric-value">{absent_today if absent_today > 0 else 0}</div>
            <div class="metric-subtitle">Unaccounted Students Today</div>
        </div>
        """, unsafe_allow_html=True)

    # Plot analytics
    if total_students > 0 and total_records > 0:
        st.markdown("### Visual Reports")
        chart_col1, chart_col2 = st.columns(2)
        
        df_att = pd.DataFrame(attendance_records)
        
        with chart_col1:
            st.markdown("<div class='card'><h4>Attendance Status Distribution</h4>", unsafe_allow_html=True)
            status_counts = df_att['status'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            colors = ['#3b82f6', '#f59e0b', '#ef4444']
            ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, 
                   textprops={'color': '#0f172a'}, colors=colors[:len(status_counts)])
            ax.axis('equal')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with chart_col2:
            st.markdown("<div class='card'><h4>Department-wise Present Count</h4>", unsafe_allow_html=True)
            if 'department' in df_att.columns:
                dept_counts = df_att[df_att['status'].isin(['Present', 'Late'])].groupby('department').size()
                if not dept_counts.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#ffffff')
                    ax.set_facecolor('#ffffff')
                    dept_counts.plot(kind='bar', color='#3b82f6', ax=ax)
                    ax.tick_params(colors='#0f172a')
                    ax.spines['bottom'].set_color('#cbd5e1')
                    ax.spines['left'].set_color('#cbd5e1')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_ylabel("Count", color='#0f172a')
                    ax.set_xlabel("Department", color='#0f172a')
                    plt.xticks(rotation=45, color='#0f172a')
                    plt.yticks(color='#0f172a')
                    st.pyplot(fig)
                else:
                    st.info("No present/late entries found for departments.")
            else:
                st.info("No department details found in attendance records.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Frequently Absent Students (<75%)
        st.markdown("### Risk Analysis: Students with Attendance &lt; 75%")
        # Calculate percentage per student
        student_history = {}
        # Count total days in database matching logs (simulate at least some sessions)
        all_dates = df_att['date'].nunique()
        if all_dates == 0:
            all_dates = 1
            
        for s in students:
            roll = s['roll_number']
            s_records = [r for r in attendance_records if r['roll_number'] == roll]
            presents = sum(1 for r in s_records if r['status'] in ['Present', 'Late'])
            rate = (presents / all_dates) * 100
            student_history[roll] = {
                'name': s['name'],
                'department': s['department'],
                'classes_attended': presents,
                'total_sessions': all_dates,
                'rate': round(rate, 2)
            }
            
        df_history = pd.DataFrame.from_dict(student_history, orient='index').reset_index()
        df_history = df_history.rename(columns={'index': 'Roll Number', 'name': 'Name', 'department': 'Department', 'classes_attended': 'Classes Attended', 'total_sessions': 'Total Sessions', 'rate': 'Attendance Rate (%)'})
        
        low_att_df = df_history[df_history['Attendance Rate (%)'] < 75.0]
        
        if not low_att_df.empty:
            st.dataframe(low_att_df, use_container_width=True)
        else:
            st.success("Great job! All students are currently above the 75% attendance threshold.")
    else:
        st.info("Add students, mark attendance, and records will show up here.")

# ----------------- PAGE: LIVE ATTENDANCE SESSION -----------------
elif page == "📹 Live Attendance Session":
    st.markdown("<h1 class='glow-title'>Live Face Recognition & Attendance</h1>", unsafe_allow_html=True)
    
    # Check if system is trained
    if st.session_state.face_classifier.knn is None:
        st.warning("⚠️ Face Recognition classifier is not trained. Please register students and verify training first.")
    else:
        st.markdown("Verify your identity. Complete the liveness challenges displayed on screen.")
        
        col_ctrl, col_cam = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Session Controls")
            
            # Dynamic Class Start Time selection
            from datetime import time as datetime_time
            class_start = st.time_input("Class Start Time", datetime_time(9, 30))
            
            if st.button("Start Attendance Stream", key="start_att", type="primary"):
                st.session_state.session_active = True
                st.session_state.liveness_detector.reset_challenge_state()
                st.session_state.last_marked = ""
                st.rerun()
                
            if st.button("Stop Stream", key="stop_att", type="secondary"):
                st.session_state.session_active = False
                st.rerun()
                
            st.write("### System Status")
            if st.session_state.session_active:
                st.markdown("🟢 **Camera active**")
                challenge = st.session_state.liveness_detector.challenge_list
                curr_idx = st.session_state.liveness_detector.current_challenge_idx
                if curr_idx < len(challenge):
                    st.info(f"👉 **Challenge:** {challenge[curr_idx]}")
                    # Progress bar
                    progress_pct = float(curr_idx) / len(challenge)
                    st.progress(progress_pct)
                else:
                    st.success("✅ **Liveness Verified!**")
            else:
                st.markdown("⚪ **Camera idle**")
                
            if st.session_state.last_marked:
                st.success(st.session_state.last_marked)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_cam:
            frame_placeholder = st.empty()
            
            if st.session_state.session_active:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Failed to open camera. Make sure no other app is using it.")
                    st.session_state.session_active = False
                else:
                    try:
                        while st.session_state.session_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to capture frame.")
                                break
                            
                            frame = cv2.flip(frame, 1) # Mirror flip
                            
                            # Process frame for liveness
                            frame, liveness_status = st.session_state.liveness_detector.process_frame(frame)
                            
                            # Face Recognition trigger after liveness passes
                            if liveness_status["liveness_passed"]:
                                num_faces, faces = st.session_state.face_classifier.detect_faces(frame)
                                if num_faces > 0:
                                    for face in faces:
                                        bbox = face[0:4].astype(int)
                                        roll_number, name, confidence = st.session_state.face_classifier.match_face(frame, face)
                                        
                                        # Draw bounding box
                                        if roll_number != "Unknown":
                                            color = (0, 255, 0) # Green for recognized
                                            label = f"{name} ({confidence:.1f}%)"
                                            
                                            # Try marking attendance
                                            marked, status, time_str = st.session_state.attendance_marker.mark_student_attendance(roll_number, name, start_time=class_start)
                                            if marked:
                                                st.session_state.last_marked = f"✅ Attendance Marked: {name} ({roll_number}) - {status} at {time_str}"
                                                # Small delay and reset for next student
                                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 3)
                                                cv2.putText(frame, "ACCESS GRANTED", (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                                time.sleep(1.5)
                                                st.session_state.liveness_detector.reset_challenge_state()
                                                st.rerun()
                                            else:
                                                st.session_state.last_marked = f"ℹ️ {name} ({roll_number}) is already marked today."
                                                
                                        else:
                                            color = (0, 0, 255) # Red for unknown
                                            label = "Unknown Face"
                                            
                                            # Unknown Face Anti-Spoofing/Logging Cooldown (5 seconds)
                                            curr_time = time.time()
                                            if curr_time - st.session_state.last_unknown_time > 5.0:
                                                st.session_state.last_unknown_time = curr_time
                                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                img_path = f"unknown_faces/unknown_{timestamp}.jpg"
                                                cv2.imwrite(img_path, frame)
                                                
                                                # Log failed attempt
                                                st.session_state.db_helper.add_log(
                                                    roll_number=None,
                                                    name="Unknown",
                                                    confidence=confidence,
                                                    status="Failed",
                                                    details=f"Unregistered face detected. Image saved to {img_path}"
                                                )
                                                logger.warning("Logged unregistered face attempt.")
                                        
                                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
                                        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            else:
                                # Overlays liveness instructions directly on output frame
                                challenge = liveness_status["current_challenge"]
                                progress = liveness_status["progress"]
                                cv2.putText(frame, f"Liveness Verification Steps: {progress}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(frame, f"DO THIS: {challenge}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            
                            # Render frame
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                            time.sleep(0.03)
                    finally:
                        cap.release()
                        st.session_state.session_active = False

# ----------------- PAGE: STUDENT REGISTRATION -----------------
elif page == "📝 Student Registration":
    st.markdown("<h1 class='glow-title'>Register New Student</h1>", unsafe_allow_html=True)
    
    col_form, col_feed = st.columns([1, 1])
    
    with col_form:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Student Details")
        roll_number = st.text_input("Roll Number", placeholder="e.g. 101")
        name = st.text_input("Full Name", placeholder="e.g. John Doe")
        department = st.selectbox("Department", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"])
        year = st.selectbox("Year", ["First Year", "Second Year", "Third Year", "Final Year"])
        division = st.selectbox("Division", ["A", "B", "C"])
        
        if st.button("Start Camera & Capture Face (50 Frames)", key="start_register_btn", type="primary"):
            if not roll_number or not name:
                st.error("Please fill in both Roll Number and Full Name.")
            else:
                # Check if roll number already exists
                existing = st.session_state.db_helper.get_student(roll_number)
                if existing:
                    st.error(f"Student with Roll Number {roll_number} is already registered!")
                else:
                    st.session_state.face_collector.start_registration(roll_number, name)
                    st.session_state.registering = True
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_feed:
        frame_placeholder = st.empty()
        progress_bar = st.empty()
        status_msg = st.empty()
        
        if st.session_state.registering:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open camera.")
                st.session_state.registering = False
            else:
                try:
                    while st.session_state.registering:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Frame read failure.")
                            break
                        
                        frame = cv2.flip(frame, 1)
                        
                        # Process frame
                        frame, count, is_complete, msg = st.session_state.face_collector.process_frame(frame)
                        
                        # Render frame & update progress bar
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                        progress_bar.progress(float(count) / 50.0)
                        status_msg.info(msg)
                        
                        if is_complete:
                            st.session_state.registering = False
                            
                            # 1. Add to Database
                            success = st.session_state.db_helper.add_student(
                                roll_number=roll_number,
                                name=name,
                                department=department,
                                year=year,
                                division=division
                            )
                            if success:
                                # 2. Retrain Classifier
                                status_msg.info("Captures finished. Training classifier (KNN)...")
                                trained, train_msg = st.session_state.student_manager.retrain_system()
                                if trained:
                                    st.success(f"🎉 Student {name} registered and face classifier retrained successfully!")
                                else:
                                    st.warning(f"Student registered, but classifier training failed: {train_msg}")
                            else:
                                st.error("Failed to register student in database.")
                            time.sleep(2.0)
                            st.rerun()
                            
                        time.sleep(0.03)
                finally:
                    cap.release()
                    st.session_state.registering = False

# ----------------- PAGE: MANAGE STUDENT RECORDS -----------------
elif page == "🗃️ Manage Student Records":
    st.markdown("<h1 class='glow-title'>Manage Student Records</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["View Records & Export", "Update/Delete Records"])
    
    with tab1:
        st.markdown("### Registered Students")
        students = st.session_state.db_helper.get_all_students()
        if students:
            df_stud = pd.DataFrame(students)
            st.dataframe(df_stud, use_container_width=True)
            
            # Export controls
            st.markdown("### Export Logs / Reports")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.write("#### Download Attendance Sheet")
                if st.button("Generate Reports (CSV/Excel)", type="primary"):
                    csv_p, xls_p = st.session_state.attendance_marker.export_attendance(filter_type='all')
                    if csv_p and xls_p:
                        st.success(f"Reports exported to attendance/ folder!")
                        # Download buttons
                        with open(csv_p, 'r') as f:
                            st.download_button("Download CSV Report", f, file_name=os.path.basename(csv_p), mime="text/csv")
                        with open(xls_p, 'rb') as f:
                            st.download_button("Download Excel Report", f, file_name=os.path.basename(xls_p), mime="application/vnd.ms-excel")
                    else:
                        st.error("No attendance data found to export.")
            with col_exp2:
                # Custom date search
                st.write("#### Search Daily Attendance")
                search_date = st.date_input("Select Date", date.today())
                if st.button("Search Date", type="secondary"):
                    date_records = st.session_state.db_helper.get_attendance_for_date(search_date.strftime("%Y-%m-%d"))
                    if date_records:
                        st.dataframe(pd.DataFrame(date_records), use_container_width=True)
                    else:
                        st.info("No records found for the selected date.")
        else:
            st.info("No students registered yet.")

    with tab2:
        st.markdown("### Modify Student Data")
        if students:
            student_roll_list = [s['roll_number'] for s in students]
            select_roll = st.selectbox("Select Student Roll Number to Modify", student_roll_list)
            
            student_data = st.session_state.db_helper.get_student(select_roll)
            
            if student_data:
                # Edit Form
                new_name = st.text_input("Name", value=student_data['name'])
                new_dept = st.selectbox("Department", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"], 
                                      index=["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil"].index(student_data['department']))
                new_year = st.selectbox("Year", ["First Year", "Second Year", "Third Year", "Final Year"], 
                                      index=["First Year", "Second Year", "Third Year", "Final Year"].index(student_data['year']))
                new_div = st.selectbox("Division", ["A", "B", "C"], 
                                     index=["A", "B", "C"].index(student_data['division']))
                
                col_mod1, col_mod2 = st.columns(2)
                with col_mod1:
                    if st.button("Update Details", key="update_btn", type="primary"):
                        success, msg = st.session_state.student_manager.update_student_record(
                            select_roll, new_name, new_dept, new_year, new_div
                        )
                        if success:
                            st.success(msg)
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            st.error(msg)
                with col_mod2:
                    if st.button("❌ Delete Student Account", key="delete_btn", type="secondary"):
                        success, msg = st.session_state.student_manager.delete_student_record(select_roll)
                        if success:
                            st.success(msg)
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No records to manage.")

# ----------------- PAGE: SYSTEM AUDIT LOGS -----------------
elif page == "🔍 System Audit Logs":
    st.markdown("<h1 class='glow-title'>System & Security Logs</h1>", unsafe_allow_html=True)
    st.write("Logs of all recognition attempts, liveness failures, camera issues, and spoofing attempts.")
    
    logs = st.session_state.db_helper.get_logs()
    if logs:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            log_filter = st.selectbox("Filter Logs by Status", ["All", "Success", "Failed", "Spoof Attempt"])
        with col_f2:
            search_log_name = st.text_input("Search Logs by Name / Roll Number", placeholder="e.g. John")
            
        filtered_logs = logs
        if log_filter != "All":
            filtered_logs = [l for l in filtered_logs if l['status'] == log_filter]
        if search_log_name:
            filtered_logs = [l for l in filtered_logs if (l['name'] and search_log_name.lower() in l['name'].lower()) or (l['roll_number'] and search_log_name in l['roll_number'])]
            
        if filtered_logs:
            st.dataframe(pd.DataFrame(filtered_logs), use_container_width=True)
        else:
            st.info("No logs match the selected filters.")
    else:
        st.info("No system logs recorded yet.")

# ----------------- PAGE: STUDENT PORTAL -----------------
elif page == "🎓 Student Portal":
    st.markdown("<h1 class='glow-title'>Student Portal</h1>", unsafe_allow_html=True)
    st.markdown("Search by your Roll Number to view your attendance history, trends, and risk analysis.")
    
    search_roll = st.text_input("Enter Roll Number", placeholder="e.g. 101")
    if st.button("Retrieve Portfolio", type="primary"):
        if not search_roll:
            st.error("Please enter a roll number.")
        else:
            student = st.session_state.db_helper.get_student(search_roll)
            if not student:
                st.error("Student Roll Number not registered.")
            else:
                col_info, col_predict = st.columns([1, 1])
                
                # Fetch attendance history
                history = st.session_state.db_helper.get_attendance_for_student(search_roll)
                
                # We need all unique dates for attendance percentage calculation
                all_attendance = st.session_state.db_helper.get_all_attendance()
                if all_attendance:
                    total_classes = pd.DataFrame(all_attendance)['date'].nunique()
                else:
                    total_classes = 0
                
                presents = sum(1 for r in history if r['status'] in ['Present', 'Late'])
                
                current_rate = (presents / total_classes * 100) if total_classes > 0 else 100.0
                
                with col_info:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"### Profile: {student['name']}")
                    st.write(f"**Roll Number:** {student['roll_number']}")
                    st.write(f"**Department:** {student['department']}")
                    st.write(f"**Division:** {student['division']} | **Year:** {student['year']}")
                    
                    st.metric("Overall Attendance Rate", f"{current_rate:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_predict:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### Attendance Risk Predictor")
                    
                    # Convert history to a binary list for the predictor trend check (1=present/late, 0=absent)
                    recent_history = []
                    # We can rebuild the chronological array of student attendance
                    history_sorted = sorted(history, key=lambda x: x['date'])
                    for h in history_sorted:
                        recent_history.append(1 if h['status'] in ['Present', 'Late'] else 0)
                    
                    # Call risk predictor
                    predicted_pct, risk_level = st.session_state.predictor.predict_risk(
                        current_pct=current_rate,
                        days_attended=presents,
                        total_days_passed=total_classes,
                        recent_history_list=recent_history
                    )
                    
                    st.metric("Predicted Semester Attendance", f"{predicted_pct:.1f}%")
                    
                    # Risk badge display
                    if risk_level == "Safe":
                        st.markdown("Risk Status: <span class='badge-safe'>Safe (≥ 75%)</span>", unsafe_allow_html=True)
                    elif risk_level == "Warning":
                        st.markdown("Risk Status: <span class='badge-warning'>Warning (65% - 75%)</span>", unsafe_allow_html=True)
                        st.warning("⚠️ Your attendance is close to falling below the 75% threshold. Attend future classes to avoid warnings.")
                    else:
                        st.markdown("Risk Status: <span class='badge-critical'>Critical (&lt; 65%)</span>", unsafe_allow_html=True)
                        st.error("🚨 Critical Alert: Your attendance trend indicates you may fall below 75% by the end of the semester.")
                        
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Render Trajectory Chart
                st.markdown("<div class='card'><h4>Projected Attendance Semester Path</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff')
                
                # Required rate line
                ax.axhline(y=75, color='#ef4444', linestyle='--', alpha=0.8, label='Required Rate (75%)')
                
                # Historic percentage path
                historic_pcts = []
                running_presents = 0
                for idx, val in enumerate(recent_history):
                    running_presents += val
                    historic_pcts.append((running_presents / (idx + 1)) * 100)
                    
                historic_days = np.arange(1, len(historic_pcts) + 1)
                if len(historic_pcts) > 0:
                    ax.plot(historic_days, historic_pcts, color='#3b82f6', linewidth=2.5, marker='o', markersize=4, label='Actual Attendance')
                
                # Projected path
                if len(historic_pcts) > 0:
                    projected_days = np.arange(len(historic_pcts), 91)
                    start_val = historic_pcts[-1]
                    projected_pcts = np.linspace(start_val, predicted_pct, len(projected_days))
                    ax.plot(projected_days, projected_pcts, color='#8b5cf6', linewidth=2, linestyle=':', label='Projected Path')
                else:
                    ax.plot([1, 90], [100, 100], color='#8b5cf6', linewidth=2, linestyle=':', label='Projected Path')
                    
                ax.set_ylim(0, 105)
                ax.set_xlim(1, 90)
                ax.set_xlabel('Semester Day', color='#0f172a')
                ax.set_ylabel('Attendance Rate (%)', color='#0f172a')
                ax.tick_params(colors='#0f172a')
                ax.spines['bottom'].set_color('#cbd5e1')
                ax.spines['left'].set_color('#cbd5e1')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(facecolor='#ffffff', edgecolor='#cbd5e1', labelcolor='#0f172a')
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display history table
                st.markdown("### Attendance History")
                if history:
                    df_hist = pd.DataFrame(history)
                    df_hist_show = df_hist[['date', 'time', 'status']].rename(columns={'date': 'Date', 'time': 'Time', 'status': 'Status'})
                    st.dataframe(df_hist_show, use_container_width=True)
                    
                    # Download personal report
                    csv_data = df_hist_show.to_csv(index=False)
                    st.download_button(
                        label="Download Personal Attendance History",
                        data=csv_data,
                        file_name=f"attendance_report_{search_roll}.csv",
                        mime='text/csv'
                    )
                else:
                    st.info("No attendance records found.")
