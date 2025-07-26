#!/usr/bin/env python3
"""
Gradio Playground - Exploring Different Components and Demos
============================================================

This file contains various Gradio demos and experiments to understand
how different components work together. Each demo is self-contained
and demonstrates specific functionality.

Requirements:
pip install gradio numpy pillow requests matplotlib

Author: Your Name
Date: 2024
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import requests
import json
import random
import time

# =============================================================================
# DEMO 1: Text Processing Playground
# =============================================================================

def text_analyzer(text, operation):
    """
    Analyze text with different operations.
    
    Args:
        text (str): Input text to analyze
        operation (str): Type of analysis to perform
    
    Returns:
        str: Analysis results
    """
    if not text.strip():
        return "Please enter some text to analyze."
    
    results = []
    
    if operation == "Word Count":
        word_count = len(text.split())
        char_count = len(text)
        char_no_spaces = len(text.replace(" ", ""))
        results.append(f"📊 **Text Statistics:**")
        results.append(f"• Words: {word_count}")
        results.append(f"• Characters (with spaces): {char_count}")
        results.append(f"• Characters (no spaces): {char_no_spaces}")
        
    elif operation == "Reverse Text":
        reversed_text = text[::-1]
        results.append(f"🔄 **Reversed Text:**")
        results.append(f"{reversed_text}")
        
    elif operation == "Case Analysis":
        results.append(f"🔤 **Case Variations:**")
        results.append(f"• Uppercase: {text.upper()}")
        results.append(f"• Lowercase: {text.lower()}")
        results.append(f"• Title Case: {text.title()}")
        results.append(f"• Sentence case: {text.capitalize()}")
        
    elif operation == "Word Frequency":
        words = text.lower().split()
        word_freq = {}
        for word in words:
            # Remove basic punctuation
            clean_word = word.strip('.,!?";')
            word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        results.append(f"📈 **Word Frequency (Top 10):**")
        for word, freq in sorted_words[:10]:
            results.append(f"• {word}: {freq}")
    
    return "\n".join(results)

# =============================================================================
# DEMO 2: Image Processing Playground
# =============================================================================

def process_image(image, effect, intensity):
    """
    Apply various effects to an uploaded image.
    
    Args:
        image: PIL Image object
        effect (str): Type of effect to apply
        intensity (float): Effect intensity (0-100)
    
    Returns:
        PIL.Image: Processed image
    """
    if image is None:
        return None
    
    # Convert intensity from 0-100 to appropriate range
    factor = intensity / 50.0  # Range 0-2, with 1 being neutral
    
    if effect == "Brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    elif effect == "Contrast":
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    elif effect == "Saturation":
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    elif effect == "Blur":
        # For blur, use intensity directly as radius
        blur_radius = intensity / 10.0
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    elif effect == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    
    elif effect == "Edge Detect":
        return image.filter(ImageFilter.FIND_EDGES)
    
    return image

# =============================================================================
# DEMO 3: Data Visualization Generator
# =============================================================================

def generate_chart(chart_type, data_points, title):
    """
    Generate different types of charts with random or custom data.
    
    Args:
        chart_type (str): Type of chart to generate
        data_points (int): Number of data points
        title (str): Chart title
    
    Returns:
        matplotlib.figure.Figure: Generated chart
    """
    # Generate sample data
    x = range(1, data_points + 1)
    y = [random.randint(10, 100) for _ in range(data_points)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "Line Chart":
        ax.plot(x, y, marker='o', linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3)
        
    elif chart_type == "Bar Chart":
        bars = ax.bar(x, y, color='skyblue', edgecolor='navy', alpha=0.7)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom')
    
    elif chart_type == "Scatter Plot":
        colors = np.random.rand(data_points)
        sizes = np.random.randint(50, 200, data_points)
        ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    
    elif chart_type == "Histogram":
        ax.hist(y, bins=min(10, data_points//3), alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Value Range')
        ax.set_ylabel('Frequency')
    
    # Customize chart
    ax.set_title(title if title else f"Sample {chart_type}", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if chart_type != "Histogram":
        ax.set_xlabel('Data Points', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
    
    plt.tight_layout()
    return fig

# =============================================================================
# DEMO 4: Simple Calculator with History
# =============================================================================

# Global variable to store calculation history
calc_history = []

def calculator(num1, operation, num2, show_history):
    """
    Perform basic calculations and maintain history.
    
    Args:
        num1 (float): First number
        operation (str): Mathematical operation
        num2 (float): Second number
        show_history (bool): Whether to show calculation history
    
    Returns:
        tuple: (result, history_text)
    """
    global calc_history
    
    try:
        if operation == "Add (+)":
            result = num1 + num2
            op_symbol = "+"
        elif operation == "Subtract (-)":
            result = num1 - num2
            op_symbol = "-"
        elif operation == "Multiply (×)":
            result = num1 * num2
            op_symbol = "×"
        elif operation == "Divide (÷)":
            if num2 == 0:
                return "Error: Division by zero!", get_history_text(show_history)
            result = num1 / num2
            op_symbol = "÷"
        else:
            return "Invalid operation!", get_history_text(show_history)
        
        # Add to history
        calculation = f"{num1} {op_symbol} {num2} = {result}"
        calc_history.append(calculation)
        
        # Keep only last 10 calculations
        if len(calc_history) > 10:
            calc_history = calc_history[-10:]
        
        return f"Result: {result}", get_history_text(show_history)
        
    except Exception as e:
        return f"Error: {str(e)}", get_history_text(show_history)

def get_history_text(show_history):
    """Get formatted history text."""
    if not show_history or not calc_history:
        return ""
    
    history_text = "📜 **Calculation History:**\n"
    for i, calc in enumerate(reversed(calc_history), 1):
        history_text += f"{i}. {calc}\n"
    
    return history_text

def clear_history():
    """Clear calculation history."""
    global calc_history
    calc_history = []
    return "History cleared!", ""

# =============================================================================
# DEMO 5: Random Quote Generator with Categories
# =============================================================================

# Sample quotes database
QUOTES_DB = {
    "Motivational": [
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Innovation distinguishes between a leader and a follower. - Steve Jobs",
        "Your time is limited, don't waste it living someone else's life. - Steve Jobs",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "It is during our darkest moments that we must focus to see the light. - Aristotle"
    ],
    "Technology": [
        "Technology is best when it brings people together. - Matt Mullenweg",
        "The science of today is the technology of tomorrow. - Edward Teller",
        "Any sufficiently advanced technology is indistinguishable from magic. - Arthur C. Clarke",
        "The real problem is not whether machines think but whether men do. - B.F. Skinner",
        "We are stuck with technology when what we really want is just stuff that works. - Douglas Adams"
    ],
    "Wisdom": [
        "The only true wisdom is in knowing you know nothing. - Socrates",
        "In the middle of difficulty lies opportunity. - Albert Einstein",
        "A person who never made a mistake never tried anything new. - Albert Einstein",
        "The only impossible journey is the one you never begin. - Tony Robbins",
        "Life is what happens to you while you're busy making other plans. - John Lennon"
    ]
}

def generate_quote(category, auto_refresh):
    """
    Generate a random quote from the selected category.
    
    Args:
        category (str): Quote category
        auto_refresh (bool): Whether this is an auto-refresh call
    
    Returns:
        str: Formatted quote
    """
    if category not in QUOTES_DB:
        return "Category not found!"
    
    quote = random.choice(QUOTES_DB[category])
    
    # Add some formatting
    formatted_quote = f"💭 **{category} Quote:**\n\n"
    formatted_quote += f"*\"{quote}\"*\n\n"
    formatted_quote += f"🎲 Click 'Generate New Quote' for another {category.lower()} quote!"
    
    return formatted_quote

# =============================================================================
# DEMO 6: Color Palette Generator
# =============================================================================

def generate_color_palette(base_color, palette_type, num_colors):
    """
    Generate color palettes based on color theory.
    
    Args:
        base_color (str): Base color in hex format
        palette_type (str): Type of color palette
        num_colors (int): Number of colors to generate
    
    Returns:
        tuple: (HTML color display, color codes text)
    """
    try:
        # Remove # if present
        base_color = base_color.lstrip('#')
        
        # Convert hex to RGB
        r = int(base_color[0:2], 16)
        g = int(base_color[2:4], 16)
        b = int(base_color[4:6], 16)
        
        colors = []
        color_info = []
        
        if palette_type == "Monochromatic":
            # Generate variations in brightness
            for i in range(num_colors):
                factor = 0.3 + (0.7 * i / (num_colors - 1)) if num_colors > 1 else 1
                new_r = int(r * factor)
                new_g = int(g * factor)
                new_b = int(b * factor)
                
                # Ensure values stay in valid range
                new_r = max(0, min(255, new_r))
                new_g = max(0, min(255, new_g))
                new_b = max(0, min(255, new_b))
                
                color_hex = f"#{new_r:02x}{new_g:02x}{new_b:02x}"
                colors.append(color_hex)
                color_info.append(f"{color_hex} - RGB({new_r}, {new_g}, {new_b})")
        
        elif palette_type == "Complementary":
            # Add base color
            colors.append(f"#{base_color}")
            color_info.append(f"#{base_color} - RGB({r}, {g}, {b}) [Base]")
            
            # Add complementary color (opposite on color wheel)
            comp_r = 255 - r
            comp_g = 255 - g
            comp_b = 255 - b
            comp_hex = f"#{comp_r:02x}{comp_g:02x}{comp_b:02x}"
            colors.append(comp_hex)
            color_info.append(f"{comp_hex} - RGB({comp_r}, {comp_g}, {comp_b}) [Complementary]")
            
            # Add variations if more colors needed
            for i in range(2, num_colors):
                if i % 2 == 0:
                    # Lighter version of base
                    factor = 1.3
                    new_r = min(255, int(r * factor))
                    new_g = min(255, int(g * factor))
                    new_b = min(255, int(b * factor))
                else:
                    # Lighter version of complementary
                    factor = 1.3
                    new_r = min(255, int(comp_r * factor))
                    new_g = min(255, int(comp_g * factor))
                    new_b = min(255, int(comp_b * factor))
                
                color_hex = f"#{new_r:02x}{new_g:02x}{new_b:02x}"
                colors.append(color_hex)
                color_info.append(f"{color_hex} - RGB({new_r}, {new_g}, {new_b})")
        
        elif palette_type == "Random":
            # Generate random colors
            colors.append(f"#{base_color}")
            color_info.append(f"#{base_color} - RGB({r}, {g}, {b}) [Base]")
            
            for i in range(1, num_colors):
                rand_r = random.randint(0, 255)
                rand_g = random.randint(0, 255)
                rand_b = random.randint(0, 255)
                color_hex = f"#{rand_r:02x}{rand_g:02x}{rand_b:02x}"
                colors.append(color_hex)
                color_info.append(f"{color_hex} - RGB({rand_r}, {rand_g}, {rand_b})")
        
        # Generate HTML for color display
        html_output = "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;'>"
        for color in colors:
            html_output += f"""
            <div style='
                width: 100px; 
                height: 100px; 
                background-color: {color}; 
                border: 2px solid #333;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: {"white" if sum([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) < 384 else "black"};
                font-weight: bold;
                font-size: 12px;
            '>
                {color.upper()}
            </div>
            """
        html_output += "</div>"
        
        # Generate text output
        text_output = f"🎨 **{palette_type} Color Palette:**\n\n"
        for info in color_info:
            text_output += f"• {info}\n"
        
        return html_output, text_output
        
    except Exception as e:
        return f"<p>Error generating palette: {str(e)}</p>", "Invalid color format. Please use hex format (e.g., #FF5733)"

# =============================================================================
# BUILD THE GRADIO INTERFACE
# =============================================================================

def create_gradio_app():
    """Create and configure the main Gradio application."""
    
    with gr.Blocks(
        title="Gradio Playground",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as app:
        
        gr.Markdown("""
        # 🎮 Gradio Playground
        
        Welcome to my Gradio exploration! This app demonstrates various Gradio components 
        and functionalities through interactive demos. Each tab showcases different 
        capabilities and use cases.
        
        **Features:**
        - Text processing and analysis
        - Image manipulation and filters  
        - Data visualization generation
        - Interactive calculator with history
        - Random quote generator
        - Color palette creator
        """)
        
        with gr.Tabs():
            
            # =================================================================
            # TAB 1: Text Processing
            # =================================================================
            with gr.Tab("📝 Text Processor"):
                gr.Markdown("### Analyze and manipulate text with different operations")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Enter your text",
                            placeholder="Type or paste text here...",
                            lines=5
                        )
                        operation_choice = gr.Dropdown(
                            choices=["Word Count", "Reverse Text", "Case Analysis", "Word Frequency"],
                            label="Choose Analysis Type",
                            value="Word Count"
                        )
                        analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
                    
                    with gr.Column(scale=2):
                        text_output = gr.Markdown(label="Analysis Results")
                
                # Example texts for quick testing
                gr.Markdown("**Quick Examples:**")
                with gr.Row():
                    example_btns = [
                        gr.Button("📖 Lorem Ipsum", size="sm"),
                        gr.Button("🎭 Shakespeare Quote", size="sm"),
                        gr.Button("💻 Tech Text", size="sm")
                    ]
                
                # Connect examples
                example_btns[0].click(
                    lambda: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                    outputs=text_input
                )
                example_btns[1].click(
                    lambda: "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
                    outputs=text_input
                )
                example_btns[2].click(
                    lambda: "Artificial intelligence and machine learning are transforming the way we interact with technology and process information.",
                    outputs=text_input
                )
                
                analyze_btn.click(
                    text_analyzer,
                    inputs=[text_input, operation_choice],
                    outputs=text_output
                )
            
            # =================================================================
            # TAB 2: Image Processing
            # =================================================================
            with gr.Tab("🖼️ Image Editor"):
                gr.Markdown("### Upload an image and apply various effects and filters")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil"
                        )
                        effect_choice = gr.Dropdown(
                            choices=["Brightness", "Contrast", "Saturation", "Blur", "Sharpen", "Edge Detect"],
                            label="Choose Effect",
                            value="Brightness"
                        )
                        intensity_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Effect Intensity"
                        )
                        process_btn = gr.Button("✨ Apply Effect", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Image(label="Processed Image")
                
                process_btn.click(
                    process_image,
                    inputs=[image_input, effect_choice, intensity_slider],
                    outputs=image_output
                )
            
            # =================================================================
            # TAB 3: Data Visualization
            # =================================================================
            with gr.Tab("📊 Chart Generator"):
                gr.Markdown("### Generate different types of charts with customizable parameters")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        chart_type = gr.Dropdown(
                            choices=["Line Chart", "Bar Chart", "Scatter Plot", "Histogram"],
                            label="Chart Type",
                            value="Line Chart"
                        )
                        data_points_slider = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=15,
                            step=1,
                            label="Number of Data Points"
                        )
                        chart_title = gr.Textbox(
                            label="Chart Title",
                            placeholder="Enter chart title (optional)"
                        )
                        generate_chart_btn = gr.Button("📈 Generate Chart", variant="primary")
                    
                    with gr.Column(scale=2):
                        chart_output = gr.Plot(label="Generated Chart")
                
                generate_chart_btn.click(
                    generate_chart,
                    inputs=[chart_type, data_points_slider, chart_title],
                    outputs=chart_output
                )
            
            # =================================================================
            # TAB 4: Calculator
            # =================================================================
            with gr.Tab("🧮 Calculator"):
                gr.Markdown("### Simple calculator with operation history")
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            num1_input = gr.Number(label="First Number", value=0)
                            operation_dropdown = gr.Dropdown(
                                choices=["Add (+)", "Subtract (-)", "Multiply (×)", "Divide (÷)"],
                                label="Operation",
                                value="Add (+)"
                            )
                            num2_input = gr.Number(label="Second Number", value=0)
                        
                        with gr.Row():
                            calculate_btn = gr.Button("🔢 Calculate", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        
                        show_history_checkbox = gr.Checkbox(
                            label="Show Calculation History",
                            value=True
                        )
                    
                    with gr.Column():
                        calc_result = gr.Textbox(label="Result", interactive=False)
                        calc_history_display = gr.Markdown(label="History")
                
                calculate_btn.click(
                    calculator,
                    inputs=[num1_input, operation_dropdown, num2_input, show_history_checkbox],
                    outputs=[calc_result, calc_history_display]
                )
                
                clear_btn.click(
                    clear_history,
                    outputs=[calc_result, calc_history_display]
                )
            
            # =================================================================
            # TAB 5: Quote Generator
            # =================================================================
            with gr.Tab("💭 Quote Generator"):
                gr.Markdown("### Get inspired with random quotes from different categories")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        quote_category = gr.Dropdown(
                            choices=list(QUOTES_DB.keys()),
                            label="Quote Category",
                            value="Motivational"
                        )
                        generate_quote_btn = gr.Button("🎲 Generate New Quote", variant="primary")
                        
                        gr.Markdown("**Available Categories:**")
                        gr.Markdown("""
                        - **Motivational**: Inspiring quotes to boost your day
                        - **Technology**: Insights about tech and innovation  
                        - **Wisdom**: Timeless wisdom from great thinkers
                        """)
                    
                    with gr.Column(scale=2):
                        quote_output = gr.Markdown(label="Your Quote")
                
                # Generate initial quote
                app.load(
                    lambda: generate_quote("Motivational", False),
                    outputs=quote_output
                )
                
                generate_quote_btn.click(
                    generate_quote,
                    inputs=[quote_category, gr.State(False)],
                    outputs=quote_output
                )
            
            # =================================================================
            # TAB 6: Color Palette Generator
            # =================================================================
            with gr.Tab("🎨 Color Palettes"):
                gr.Markdown("### Create beautiful color palettes from a base color")
                
                with gr.Row():
                    with gr.Column():
                        base_color_input = gr.Textbox(
                            label="Base Color (Hex)",
                            value="#3498db",
                            placeholder="e.g., #FF5733"
                        )
                        palette_type_dropdown = gr.Dropdown(
                            choices=["Monochromatic", "Complementary", "Random"],
                            label="Palette Type",
                            value="Monochromatic"
                        )
                        num_colors_slider = gr.Slider(
                            minimum=2,
                            maximum=8,
                            value=5,
                            step=1,
                            label="Number of Colors"
                        )
                        generate_palette_btn = gr.Button("🎨 Generate Palette", variant="primary")
                        
                        gr.Markdown("""
                        **Palette Types:**
                        - **Monochromatic**: Variations of the same hue
                        - **Complementary**: Colors opposite on the color wheel
                        - **Random**: Randomly generated color combinations
                        """)
                    
                    with gr.Column():
                        color_display = gr.HTML(label="Color Palette")
                        color_codes = gr.Markdown(label="Color Information")
                
                generate_palette_btn.click(
                    generate_color_palette,
                    inputs=[base_color_input, palette_type_dropdown, num_colors_slider],
                    outputs=[color_display, color_codes]
                )
                
                # Generate initial palette
                app.load(
                    lambda: generate_color_palette("#3498db", "Monochromatic", 5),
                    outputs=[color_display, color_codes]
                )
        
        # Footer
        gr.Markdown("""
        ---
        
        **🚀 Gradio Playground** - Built with [Gradio](https://gradio.app/)
        
        This playground demonstrates various Gradio components including:
        `gr.Textbox`, `gr.Dropdown`, `gr.Slider`, `gr.Image`, `gr.Plot`, `gr.Button`, 
        `gr.Markdown`, `gr.HTML`, `gr.Number`, `gr.Checkbox`, and `gr.Tabs`.
        
        Perfect for learning how to build interactive ML and data science applications! 🎯
        """)
    
    return app

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create and launch the app
    app = create_gradio_app()
    
    # Launch with custom settings
    app.launch(
        share=False,          # Set to True to get a public link
        server_name="0.0.0.0", # Allow external connections
        server_port=7860,     # Default Gradio port
        show_tips=True,       # Show helpful tips
        show_error=True,      # Show errors in browser
        quiet=False           # Print startup messages
    )
    
    # Alternative launch options you can try:
    # app.launch(share=True)  # Get a public shareable link
    # app.launch(debug=True)  # Enable debug mode
    # app.launch(auth=("username", "password"))  # Add authentication