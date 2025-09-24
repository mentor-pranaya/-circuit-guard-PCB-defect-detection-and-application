import streamlit as st
from PIL import Image, ImageDraw
import io
import time

def process_images_mock(ref_image, defected_image):
    """
    This function simulates the image processing and defect detection.
    
    In a real application, this is where you would integrate your
    image subtraction, contour detection, and EfficientNet model.
    The function would return the final image with defect labels.
    """
    
    # Use the defected image as the base for the output
    output_image = defected_image.copy()
    
    # Convert to RGB to ensure drawing works correctly
    if output_image.mode != 'RGB':
        output_image = output_image.convert('RGB')
        
    draw = ImageDraw.Draw(output_image)
    
    # Simulate defect detection by drawing bounding boxes
    width, height = output_image.size
    
    # Mock defect labels and coordinates
    defects = [
        {"label": "Missing Component", "box": (int(width * 0.2), int(height * 0.3), int(width * 0.35), int(height * 0.4))},
        {"label": "Solder Bridge", "box": (int(width * 0.55), int(height * 0.6), int(width * 0.7), int(height * 0.7))},
    ]

    # Draw a semi-transparent overlay
    overlay = Image.new('RGBA', output_image.size, (255, 0, 0, 50))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw the bounding boxes and text
    for defect in defects:
        box = defect["box"]
        label = defect["label"]
        
        # Draw the semi-transparent red box
        overlay_draw.rectangle(box, fill=(255, 0, 0, 100))
        
        # Draw a red border
        draw.rectangle(box, outline="red", width=3)
        
        # Draw the label text
        text_position = (box[0], box[1] - 15)
        draw.text(text_position, label, fill="red")

    # Combine the overlay and the original image
    output_image = Image.alpha_composite(output_image.convert("RGBA"), overlay).convert("RGB")
        
    return output_image, defects

def main():
    st.set_page_config(page_title="AI Circuit Guard", layout="wide")
    st.title("AI Circuit Guard")
    st.markdown(
        """
        <style>
        .stButton>button {
            border-radius: 20px;
            color: white;
            background-color: #6366f1;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #4f46e5;
        }
        .stFileUploader>div {
            border-radius: 12px;
            border: 2px dashed #424242;
            padding: 2rem;
            text-align: center;
        }
        .stFileUploader>div p {
            color: #9e9e9e;
        }
        .container {
            background-color: #1e1e1e;
            border-radius: 1.5rem;
            padding: 2.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="container">
            <h2 style='text-align: center; color: #E0E0E0;'>Upload PCB Images</h2>
            <p style='text-align: center; color: #A0A0A0;'>Please upload a golden (reference) image and a defected image to begin the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Reference Image")
        ref_image_file = st.file_uploader("Golden PCB", type=["jpg", "jpeg", "png"], key="ref_uploader")
        if ref_image_file is not None:
            ref_image = Image.open(ref_image_file)
            st.image(ref_image, caption="Reference Image", use_column_width=True)

    with col2:
        st.header("Upload Defected Image")
        defected_image_file = st.file_uploader("PCB to Test", type=["jpg", "jpeg", "png"], key="defected_uploader")
        if defected_image_file is not None:
            defected_image = Image.open(defected_image_file)
            st.image(defected_image, caption="Defected Image", use_column_width=True)

    if ref_image_file and defected_image_file:
        st.divider()
        if st.button("Detect Defects", use_container_width=True):
            with st.spinner("Detecting defects... Please wait."):
                # Simulate a delay for the processing
                time.sleep(2)
                
                # Process the images
                ref_image = Image.open(ref_image_file)
                defected_image = Image.open(defected_image_file)
                
                output_image, defects = process_images_mock(ref_image, defected_image)
            
            st.success("Defect analysis complete!")
            st.header("Defect Analysis Output")
            
            st.image(output_image, caption="Defected Image with Labels", use_column_width=True)

            # Convert the output image to bytes for download
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Output Image",
                data=byte_im,
                file_name="aicircuit_guard_output.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Display a summary of detected defects
            st.subheader("Summary of Defects")
            for i, defect in enumerate(defects):
                st.markdown(f"**Defect {i+1}:** {defect['label']}")
                
    else:
        st.info("Please upload both images to enable the 'Detect Defects' button.")

if __name__ == "__main__":
    main()

