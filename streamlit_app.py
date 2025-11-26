
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Force CPU for Streamlit deployment reliability
DEVICE = torch.device('cpu')
IMG_SIZE = 224
NUM_CLASSES = 7

# Define paths for two image folders
DATA_DIR_PART1 = Path('data/raw/HAM10000_images_part_1')
DATA_DIR_PART2 = Path('data/raw/HAM10000_images_part_2')

CLASS_NAMES = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}

CLASS_DESCRIPTIONS = {
    0: 'Precancerous skin lesions caused by sun damage',
    1: 'Most common type of skin cancer, rarely spreads',
    2: 'Non-cancerous skin growth, harmless',
    3: 'Benign skin nodule, usually harmless',
    4: 'Serious skin cancer, can spread to other organs',
    5: 'Common moles, usually benign',
    6: 'Blood vessel abnormalities in the skin'
}

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def find_image_path(image_id):
    """Find image in either part_1 or part_2 folder"""
    path1 = DATA_DIR_PART1 / f"{image_id}.jpg"
    path2 = DATA_DIR_PART2 / f"{image_id}.jpg"

    if path1.exists():
        return path1
    elif path2.exists():
        return path2
    else:
        return None

@st.cache_resource
def load_models():
    """Load all trained models"""
    models_dict = {}

    try:
        # ResNet50 Baseline
        if Path('resnet50_baseline_best.pth').exists():
            resnet50_base = models.resnet50(weights=None)
            resnet50_base.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(2048, 512), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
            )
            resnet50_base.load_state_dict(torch.load('resnet50_baseline_best.pth', 
                                                      map_location=DEVICE, weights_only=False))
            resnet50_base = resnet50_base.to(DEVICE)
            resnet50_base.eval()
            models_dict['ResNet50 Baseline'] = resnet50_base

        # ResNet50 Fine-tuned
        if Path('resnet50_finetuned_best.pth').exists():
            resnet50_ft = models.resnet50(weights=None)
            resnet50_ft.fc = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(2048, 512), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(512, NUM_CLASSES)
            )
            resnet50_ft.load_state_dict(torch.load('resnet50_finetuned_best.pth', 
                                                   map_location=DEVICE, weights_only=False))
            resnet50_ft = resnet50_ft.to(DEVICE)
            resnet50_ft.eval()
            models_dict['ResNet50 Fine-tuned'] = resnet50_ft

        # EfficientNet Baseline - ADDED
        if Path('efficientnet_baseline_best.pth').exists():
            effnet_base = models.efficientnet_b0(weights=None)
            effnet_base.classifier = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(1280, 512), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES)
            )
            effnet_base.load_state_dict(torch.load('efficientnet_baseline_best.pth', 
                                                   map_location=DEVICE, weights_only=False))
            effnet_base = effnet_base.to(DEVICE)
            effnet_base.eval()
            models_dict['EfficientNet Baseline'] = effnet_base

        # EfficientNet Fine-tuned
        if Path('efficientnet_ensemble.pth').exists():
            effnet_ft = models.efficientnet_b0(weights=None)
            effnet_ft.classifier = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(1280, 512), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(512, NUM_CLASSES)
            )
            effnet_ft.load_state_dict(torch.load('efficientnet_ensemble.pth', 
                                                 map_location=DEVICE, weights_only=False))
            effnet_ft = effnet_ft.to(DEVICE)
            effnet_ft.eval()
            models_dict['EfficientNet Fine-tuned'] = effnet_ft

        # ResNet101 - REMOVED "(Best)" from name
        if Path('resnet101_ensemble.pth').exists():
            resnet101 = models.resnet101(weights=None)
            resnet101.fc = nn.Sequential(
                nn.Dropout(0.4), nn.Linear(2048, 512), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(512, NUM_CLASSES)
            )
            resnet101.load_state_dict(torch.load('resnet101_ensemble.pth', 
                                                 map_location=DEVICE, weights_only=False))
            resnet101 = resnet101.to(DEVICE)
            resnet101.eval()
            models_dict['ResNet101'] = resnet101

        if len(models_dict) > 0:
            st.sidebar.success(f"✓ Loaded {len(models_dict)} models")

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure all model files (.pth) are in the same directory")

    return models_dict

@st.cache_data
def load_test_images():
    """Load test dataset for selection"""
    try:
        test_df = pd.read_csv('test_data.csv')

        def get_image_path(image_id):
            path = find_image_path(image_id)
            return str(path) if path else ""

        test_df['image_path'] = test_df['image_id'].apply(get_image_path)
        test_df = test_df[test_df['image_path'] != ""]

        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return test_df, le
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return None, None

@st.cache_data
def load_evaluation_results():
    """Load pre-computed evaluation results"""
    try:
        with open('final_evaluation_results.json', 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")
        return None

def predict_image(image, models_dict):
    """Predict on a single image with all models"""
    predictions = {}

    try:
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            for name, model in models_dict.items():
                model = model.to(DEVICE)
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predictions[name] = {
                    'probabilities': probs,
                    'predicted_class': int(np.argmax(probs)),
                    'confidence': float(np.max(probs))
                }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return {}

    return predictions

# Initialize
st.markdown('<p class="main-header">🔬 Skin Lesion Classification System</p>', unsafe_allow_html=True)
st.markdown("**HAM10000 Medical Image Classification using Deep Learning**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Image source selection
    image_source = st.radio(
        "Select Image Source:",
        ["Upload Image", "Select from Test Set"]
    )

    # Model selection
    st.subheader("Select Models")
    models_dict = load_models()

    if models_dict:
        # CHANGED: Default to all models
        selected_models = st.multiselect(
            "Choose models to compare:",
            list(models_dict.keys()),
            default=list(models_dict.keys()),  # Select all by default
            help="All models are selected by default. Deselect any you don't want to use."
        )
    else:
        selected_models = []
        st.error("No models loaded")

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info(f"""
    This system classifies 7 types of skin lesions using deep learning models trained on the HAM10000 dataset.

    **Models Available:** {len(models_dict)}
    - ResNet50 (Baseline & Fine-tuned)
    - EfficientNet (Baseline & Fine-tuned)
    - ResNet101
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Model Performance", "📈 Visualizations", "📋 About Dataset"])

# TAB 1: PREDICTION
with tab1:
    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Input Image")

            image = None
            true_label = None

            if image_source == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload a skin lesion image (JPG/PNG)",
                    type=['jpg', 'jpeg', 'png']
                )
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_container_width=True)

            else:  # Select from test set
                test_df, le = load_test_images()

                if test_df is not None and len(test_df) > 0:
                    # Filter options
                    class_filter = st.selectbox(
                        "Filter by class (optional):",
                        ['All'] + list(CLASS_NAMES.values())
                    )

                    if class_filter != 'All':
                        label_idx = [k for k, v in CLASS_NAMES.items() if v == class_filter][0]
                        filtered_df = test_df[test_df['label'] == label_idx]
                    else:
                        filtered_df = test_df

                    if len(filtered_df) == 0:
                        st.warning("No images found for selected class")
                    else:
                        # Random selection
                        if st.button("🎲 Select Random Image"):
                            st.session_state.selected_idx = np.random.randint(0, len(filtered_df))

                        if 'selected_idx' not in st.session_state:
                            st.session_state.selected_idx = 0

                        # Image selector
                        selected_row = filtered_df.iloc[st.session_state.selected_idx % len(filtered_df)]
                        image_path = selected_row['image_path']
                        true_label = selected_row['label']

                        if image_path and Path(image_path).exists():
                            image = Image.open(image_path).convert('RGB')
                            st.image(image, caption=f"Test Image (True: {CLASS_NAMES[true_label]})", 
                                    use_container_width=True)
                        else:
                            st.error(f"Image not found: {image_path}")
                            image = None

                        # Navigation
                        col_prev, col_next = st.columns(2)
                        with col_prev:
                            if st.button("⬅️ Previous"):
                                st.session_state.selected_idx = (st.session_state.selected_idx - 1) % len(filtered_df)
                                st.rerun()
                        with col_next:
                            if st.button("➡️ Next"):
                                st.session_state.selected_idx = (st.session_state.selected_idx + 1) % len(filtered_df)
                                st.rerun()
                else:
                    st.error("Test data not available or empty")

        with col2:
            st.subheader("🎯 Prediction Results")

            if image is not None:
                with st.spinner("Making predictions..."):
                    # Filter selected models
                    selected_models_dict = {k: v for k, v in models_dict.items() if k in selected_models}
                    predictions = predict_image(image, selected_models_dict)

                if len(predictions) > 0:
                    st.subheader("🤝 Model Consensus Analysis")

                    pred_classes = [pred['predicted_class'] for pred in predictions.values()]
                    most_common = max(set(pred_classes), key=pred_classes.count)
                    agreement = pred_classes.count(most_common) / len(pred_classes)
                    agreement_percent = agreement * 100

                    # Determine color based on agreement level
                    if agreement_percent > 70:
                        color = "#28a745"  # Green
                        reliability = "Strong"
                        icon = "✅"
                    elif agreement_percent >= 50:
                        color = "#007bff"  # Blue
                        reliability = "Moderate"
                        icon = "ℹ️"
                    else:
                        color = "#dc3545"  # Red
                        reliability = "Weak"
                        icon = "⚠️"

                    # Custom styled metrics with uniform font size
                    col_cons1, col_cons2, col_cons3 = st.columns(3)

                    with col_cons1:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                            <p style='font-size: 0.9rem; color: #666; margin: 0;'>Consensus Class</p>
                            <p style='font-size: 0.9rem; font-weight: bold; color: #1f77b4; margin: 0.5rem 0;'>
                                {CLASS_NAMES[most_common]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_cons2:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                            <p style='font-size: 0.9rem; color: #666; margin: 0;'>Agreement Level</p>
                            <p style='font-size: 0.9rem; font-weight: bold; color: {color}; margin: 0.5rem 0;'>
                                {agreement_percent:.0f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_cons3:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                            <p style='font-size: 0.9rem; color: #666; margin: 0;'>Reliability</p>
                            <p style='font-size: 0.9rem; font-weight: bold; color: {color}; margin: 0.5rem 0;'>
                                {icon} {reliability}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Warning message for low agreement
                    if agreement_percent < 50:
                        st.error("⚠️ **Low model agreement** - Consider additional diagnostic tests")
                    elif agreement_percent < 70:
                        st.info("ℹ️ **Moderate agreement** - Results are reasonably consistent")
                    else:
                        st.success("✅ **Strong agreement** - Models are highly consistent")

                    st.markdown("---")




                    # Display individual predictions
                    for model_name, pred in predictions.items():
                        predicted_class = pred['predicted_class']
                        confidence = pred['confidence']

                        # Color code: green if correct, red if wrong, blue if unknown
                        if true_label is not None:
                            if predicted_class == true_label:
                                icon = "✅"
                            else:
                                icon = "❌"
                        else:
                            icon = "🔹"

                        with st.expander(f"{icon} **{model_name}** - {CLASS_NAMES[predicted_class]} ({confidence*100:.1f}%)", 
                                        expanded=True):
                            st.markdown(f"**Predicted:** {CLASS_NAMES[predicted_class]}")
                            st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                            st.markdown(f"*{CLASS_DESCRIPTIONS[predicted_class]}*")

                            # Probability bar chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=pred['probabilities'],
                                    y=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
                                    orientation='h',
                                    marker=dict(
                                        color=pred['probabilities'],
                                        colorscale='Blues',
                                        showscale=False
                                    )
                                )
                            ])
                            fig.update_layout(
                                title="Class Probabilities",
                                xaxis_title="Probability",
                                yaxis_title="Class",
                                height=300,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Upload an image or select from test set to begin")

# TAB 2: MODEL PERFORMANCE
with tab2:
    st.header("📊 Model Performance Comparison")

    results = load_evaluation_results()

    if results:
        # Create comparison dataframe
        perf_data = []
        for model_name, metrics in results['test_results'].items():
            perf_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC': f"{metrics['auc']:.4f}",
                'Inference (ms)': f"{metrics['inference_time_ms']:.2f}"
            })

        perf_df = pd.DataFrame(perf_data)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        best_model = max(results['test_results'].items(), key=lambda x: x[1]['accuracy'])

        with col1:
            st.metric("Best Model", best_model[0].split()[0], 
                     f"{best_model[1]['accuracy']*100:.1f}%")
        with col2:
            st.metric("Best F1-Score", "", f"{best_model[1]['f1']:.4f}")
        with col3:
            st.metric("Best AUC", "", f"{best_model[1]['auc']:.4f}")
        with col4:
            avg_acc = np.mean([m['accuracy'] for m in results['test_results'].values()])
            st.metric("Average Accuracy", "", f"{avg_acc*100:.1f}%")

        st.markdown("---")

        # Performance table
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Interactive comparison chart
        st.subheader("Performance Metrics Comparison")

        metric_choice = st.selectbox(
            "Select metric to visualize:",
            ['Accuracy', 'F1-Score', 'AUC', 'Inference Time']
        )

        metric_map = {
            'Accuracy': 'accuracy',
            'F1-Score': 'f1',
            'AUC': 'auc',
            'Inference Time': 'inference_time_ms'
        }

        models = list(results['test_results'].keys())
        values = [results['test_results'][m][metric_map[metric_choice]] for m in models]

        if metric_choice != 'Inference Time':
            values = [v * 100 for v in values]

        fig = go.Figure(data=[
            go.Bar(x=models, y=values, marker_color='steelblue')
        ])
        fig.update_layout(
            title=f"{metric_choice} by Model",
            xaxis_title="Model",
            yaxis_title=metric_choice + (' (%)' if metric_choice != 'Inference Time' else ' (ms)'),
            height=400
        )

        if metric_choice == 'Accuracy':
            fig.add_hline(y=75, line_dash="dash", line_color="red", 
                         annotation_text="Target (75%)")

        st.plotly_chart(fig, use_container_width=True)

# TAB 3: VISUALIZATIONS
with tab3:
    st.header("📈 Model Visualizations")

    viz_type = st.selectbox(
        "Select Visualization:",
        ["Confusion Matrices", "Per-Class Performance", "ROC Curves", "Overall Comparison"]
    )

    try:
        if viz_type == "Confusion Matrices":
            st.image('confusion_matrices_comparison.png', 
                    caption="Confusion Matrices for All Models", 
                    use_container_width=True)
            st.info("Darker blue = higher proportion of predictions. Diagonal = correct predictions.")

        elif viz_type == "Per-Class Performance":
            st.image('per_class_performance.png', 
                    caption="Precision, Recall, and F1-Score by Class", 
                    use_container_width=True)
            st.info("Shows how each model performs on individual skin lesion types.")

        elif viz_type == "ROC Curves":
            st.image('roc_curves_by_class.png', 
                    caption="ROC Curves for Each Class", 
                    use_container_width=True)
            st.info("Higher AUC (area under curve) = better class discrimination ability.")

        elif viz_type == "Overall Comparison":
            st.image('overall_metrics_comparison.png', 
                    caption="Overall Model Performance Metrics", 
                    use_container_width=True)
            st.info("Comprehensive comparison of accuracy, F1, AUC, and inference speed.")
    except:
        st.error("Visualization files not found. Make sure all PNG files are in the same directory.")

# TAB 4: ABOUT DATASET
with tab4:
    st.header("📋 HAM10000 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.markdown("""
        **HAM10000** (Human Against Machine with 10000 training images)

        - **Total Images:** 10,015
        - **Classes:** 7 types of skin lesions
        - **Source:** Dermatoscopic images
        - **Purpose:** Early skin cancer detection

        **Data Split:**
        - Training: 7,010 images (70%)
        - Validation: 1,502 images (15%)
        - Test: 1,503 images (15%)
        """)

        st.markdown("---")

        st.subheader("Class Imbalance Handling")
        st.markdown("""
        The dataset exhibits severe class imbalance with a **57.9:1 ratio** between the most and least frequent classes.

        **Imbalance Statistics:**
        - Majority class (nv): 4,693 samples (66.9%)
        - Minority classes: df (81), vasc (99), akiec (229)

        **Techniques Applied:**
        1. **Weighted Random Sampling** - Balances class distribution per epoch
        2. **Class-weighted Loss Function** - Penalizes minority class errors more
        3. **Adaptive Augmentation** - Strong augmentation for minority classes
        4. **Medical-specific Augmentations** - Hair artifacts, color constancy, elastic deformation

        These techniques improved minority class performance significantly while maintaining overall accuracy.
        """)

    with col2:
        st.subheader("Class Distribution")

        # Load training data for distribution
        try:
            train_df = pd.read_csv('train_data.csv')
            class_counts = train_df['label'].value_counts().sort_index()

            fig = go.Figure(data=[
                go.Pie(
                    labels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
                    values=class_counts.values,
                    hole=0.3
                )
            ])
            fig.update_layout(title="Training Set Class Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Bar chart showing imbalance
            fig2 = go.Figure(data=[
                go.Bar(
                    x=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
                    y=class_counts.values,
                    marker_color=['red' if count < 300 else 'orange' if count < 1000 else 'green' 
                                  for count in class_counts.values]
                )
            ])
            fig2.update_layout(
                title="Class Imbalance (Training Set)",
                xaxis_title="Class",
                yaxis_title="Number of Samples",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        except:
            st.error("Training data not available for visualization")

    st.markdown("---")

    st.subheader("Lesion Type Descriptions")

    for class_id, class_name in CLASS_NAMES.items():
        with st.expander(f"**{class_name}**"):
            st.markdown(f"*{CLASS_DESCRIPTIONS[class_id]}*")

            # Add clinical info
            if class_id == 4:  # Melanoma
                st.warning("⚠️ **Most dangerous** - Requires immediate medical attention")
            elif class_id in [0, 1]:  # akiec, bcc
                st.info("ℹ️ **Potentially cancerous** - Should be monitored")
            else:
                st.success("✅ **Generally benign** - Usually harmless")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with PyTorch, Streamlit | HAM10000 Medical Image Classification</p>
    <p><small>⚠️ Disclaimer: This is a demonstration system. Always consult medical professionals for diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)
