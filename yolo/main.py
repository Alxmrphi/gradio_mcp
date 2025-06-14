import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import pandas as pd
import os
from pathlib import Path

# Available YOLO models
MODELS = {
    'YOLOv8n (fastest)': 'yolov8n.pt',
    'YOLOv8s (balanced)': 'yolov8s.pt', 
    'YOLOv8m (accurate)': 'yolov8m.pt',
    'YOLOv8l (very accurate)': 'yolov8l.pt',
    'YOLOv8x (most accurate)': 'yolov8x.pt'
}

# Global model cache
model_cache = {}

def load_model(model_name):
    """Load and cache YOLO model"""
    model_path = MODELS[model_name]
    if model_path not in model_cache:
        print(f"Loading {model_name}...")
        model_cache[model_path] = YOLO(model_path)
    return model_cache[model_path]

def detect_objects(image, model_name, confidence_threshold, iou_threshold, show_labels, show_confidence):
    """
    Perform object detection with customizable parameters
    """
    if image is None:
        return None, pd.DataFrame({'Message': ['Please upload an image']}), "No image provided"
    
    try:
        # Load selected model
        model = load_model(model_name)
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        original_height, original_width = img_array.shape[:2]
        
        # Run inference
        results = model(img_array, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        
        # Process results
        detections = []
        annotated_img = img_array.copy()
        
        # Color palette for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 165, 0)
        ]
        
        detection_count = 0
        class_counts = {}
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Count detections
                    detection_count += 1
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Calculate box dimensions
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    area = width * height
                    
                    # Store detection info
                    detections.append({
                        'ID': detection_count,
                        'Class': class_name,
                        'Confidence': f"{confidence:.3f}",
                        'BBox (x1,y1,x2,y2)': f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                        'Width': width,
                        'Height': height,
                        'Area': area,
                        'Center X': int((x1 + x2) / 2),
                        'Center Y': int((y1 + y2) / 2)
                    })
                    
                    # Choose color based on class
                    color = colors[class_id % len(colors)]
                    
                    # Draw bounding box with thicker lines
                    cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Prepare label text
                    label_parts = []
                    if show_labels:
                        label_parts.append(class_name)
                    if show_confidence:
                        label_parts.append(f"{confidence:.2f}")
                    
                    if label_parts:
                        label = ": ".join(label_parts)
                        
                        # Calculate text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_img,
                            (int(x1), int(y1) - text_height - 10),
                            (int(x1) + text_width, int(y1)),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_img, 
                            label, 
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (255, 255, 255), 
                            2
                        )
        
        # Convert back to PIL
        result_image = Image.fromarray(annotated_img)
        
        # Create DataFrame for results table
        if detections:
            df = pd.DataFrame(detections)
            # Sort by confidence descending
            df = df.sort_values('Confidence', ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame({'Message': ['No objects detected above confidence threshold']})
        
        # Create summary text
        if detection_count > 0:
            summary_lines = [
                f"🎯 **Detection Summary**",
                f"📊 Total objects detected: **{detection_count}**",
                f"📏 Image dimensions: **{original_width} × {original_height}**",
                f"🤖 Model used: **{model_name}**",
                f"⚡ Confidence threshold: **{confidence_threshold:.2f}**",
                f"🔗 IoU threshold: **{iou_threshold:.2f}**",
                "",
                "📋 **Class Distribution:**"
            ]
            
            for class_name, count in sorted(class_counts.items()):
                summary_lines.append(f"   • {class_name}: {count}")
            
            summary = "\n".join(summary_lines)
        else:
            summary = f"❌ No objects detected above confidence threshold of {confidence_threshold:.2f}"
        
        return result_image, df, summary
        
    except Exception as e:
        error_msg = f"Error during detection: {str(e)}"
        return None, pd.DataFrame({'Error': [error_msg]}), error_msg

def clear_all():
    """Clear all inputs and outputs"""
    return None, None, pd.DataFrame(), "", 0.5, 0.45, "YOLOv8n (fastest)", True, True

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1200px !important;
    margin: auto;
}

.title {
    text-align: center;
    color: #2563eb;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.description {
    text-align: center;
    font-size: 1.2em;
    color: #6b7280;
    margin-bottom: 2em;
}

.output-image {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.settings-panel {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}

/* Highlight detected objects table */
.detection-table {
    font-size: 0.9em;
    border-radius: 8px;
    overflow: hidden;
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    padding: 0.75rem 2rem !important;
    border-radius: 25px !important;
    transition: all 0.3s ease !important;
}

.secondary-button {
    background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%) !important;
    border: none !important;
    color: #2d3436 !important;
    font-weight: bold !important;
    padding: 0.75rem 2rem !important;
    border-radius: 25px !important;
}
"""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css=custom_css, title="🎯 YOLO Object Detection Studio") as demo:
        
        # Header
        gr.HTML("""
        <div class="title">🎯 YOLO Object Detection Studio</div>
        <div class="description">
            Upload an image and let AI detect and identify objects with bounding boxes
        </div>
        """)
        
        with gr.Row():
            # Left column - Input controls
            with gr.Column(scale=1):
                gr.HTML('<div style="font-size: 1.3em; font-weight: bold; margin-bottom: 1em;">📤 Upload & Settings</div>')
                
                # Image input
                input_image = gr.Image(
                    type="pil", 
                    label="📸 Upload Image",
                    height=300
                )
                
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="YOLOv8n (fastest)",
                    label="🤖 Select YOLO Model",
                    info="Larger models are more accurate but slower"
                )
                
                # Detection parameters
                gr.HTML('<div style="font-weight: bold; margin: 1em 0 0.5em 0;">⚙️ Detection Parameters</div>')
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="🎯 Confidence Threshold",
                    info="Minimum confidence for detections"
                )
                
                iou_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="🔗 IoU Threshold", 
                    info="Overlap threshold for duplicate removal"
                )
                
                # Display options
                gr.HTML('<div style="font-weight: bold; margin: 1em 0 0.5em 0;">🎨 Display Options</div>')
                
                with gr.Row():
                    show_labels = gr.Checkbox(
                        value=True,
                        label="Show Labels"
                    )
                    show_confidence = gr.Checkbox(
                        value=True,
                        label="Show Confidence"
                    )
                
                # Action buttons
                with gr.Row():
                    detect_btn = gr.Button(
                        "🔍 Detect Objects", 
                        variant="primary",
                        elem_classes=["primary-button"]
                    )
                    clear_btn = gr.Button(
                        "🧹 Clear All",
                        variant="secondary", 
                        elem_classes=["secondary-button"]
                    )
            
            # Right column - Results
            with gr.Column(scale=2):
                gr.HTML('<div style="font-size: 1.3em; font-weight: bold; margin-bottom: 1em;">📊 Detection Results</div>')
                
                # Output image
                output_image = gr.Image(
                    label="🎯 Detected Objects",
                    height=400,
                    elem_classes=["output-image"]
                )
                
                # Summary text
                summary_text = gr.Markdown(
                    label="📋 Summary",
                    value="Upload an image and click 'Detect Objects' to see results"
                )
        
        # Full-width results table
        gr.HTML('<div style="font-size: 1.3em; font-weight: bold; margin: 2em 0 1em 0;">📄 Detailed Detection Results</div>')
        results_table = gr.Dataframe(
            label="Detection Details",
            elem_classes=["detection-table"],
            wrap=True
        )
        
        # Event handlers
        detect_btn.click(
            fn=detect_objects,
            inputs=[
                input_image, model_dropdown, confidence_slider, 
                iou_slider, show_labels, show_confidence
            ],
            outputs=[output_image, results_table, summary_text]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[
                input_image, output_image, results_table, summary_text,
                confidence_slider, iou_slider, model_dropdown, 
                show_labels, show_confidence
            ]
        )
        
        # Auto-detect on image upload
        input_image.change(
            fn=detect_objects,
            inputs=[
                input_image, model_dropdown, confidence_slider,
                iou_slider, show_labels, show_confidence  
            ],
            outputs=[output_image, results_table, summary_text]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2em; padding: 1em; 
                    background-color: #f8fafc; border-radius: 10px; color: #6b7280;">
            <p>🚀 Powered by YOLOv8 and Gradio | Built with ❤️ for Object Detection</p>
            <p><small>💡 Tip: Try different models and thresholds for optimal results</small></p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create example images directory if it doesn't exist
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Launch the application
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public link
        debug=True,
        show_error=True,
        favicon_path=None,
        # ssl_keyfile=None,  # Add SSL cert path if needed
        # ssl_certfile=None, # Add SSL cert path if needed
    )