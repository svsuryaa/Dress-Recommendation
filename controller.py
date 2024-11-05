import streamlit as st
import os
import shutil
from PIL import Image
from get_image_text import get_text_for_image
from RAG_fusion_2 import rrf_retriever
from gemini import generate_context
from RAG_fusion import dress_retriever
from stable_diffusion import generate_image

def display_results_streamlit(result_dir, suggested_outfits=None):
    # Read and display images from the specified directory (either result_wardrobe or result_image)
    image_paths = [os.path.join(result_dir, img) for img in os.listdir(result_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_paths:
        st.write(f"Images from {result_dir}:")
        # Display as a carousel
        for img_path in image_paths:
            st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
    else:
        st.warning(f"No images found in the {result_dir} directory.")
    
    # Display suggested outfits (if provided)

def process_results(image_path, customer_context):
    similiar_names = set()

    # Step 1: Extract text from the image
    extracted_text = get_text_for_image(image_path)
    print("Extracted Text from Image:")
    print(extracted_text)

    # Step 2: Call RAG fusion using the extracted text
    retrieved_documents = rrf_retriever(extracted_text)

    # Step 3: Output retrieved documents
    profiles = []
    for doc in retrieved_documents:
        content = doc.page_content[:100]
        start = content.find('"') + 1
        end = content.find('":')
        name = content[start:end].strip()
        similiar_names.add(name)

        # Load the actress profile from the corresponding text file
        profile_path = os.path.join("actress_profile", f"{name}.txt")
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as file:
                profile_content = file.read()
                profiles.append(profile_content)
    
    # Step 4: Combine all profiles into a single string
    all_profiles_content = "\n\n".join(profiles)
    celebrity_context, suggested_outfits = generate_context(all_profiles_content, customer_context)
    total_context = f"{customer_context}, {celebrity_context}"
    retrieved_dress = dress_retriever(total_context)

    for doc in retrieved_dress:
        content = doc.page_content[:100]
        start = content.find('"') + 1
        end = content.find('":')
        name = content[start:end].strip()
        similiar_names.add(name)

    input_images_dir = 'input_images'
    result_wardrobe_dir = 'result_wardrobe'
    
    # Ensure the result_wardrobe directory exists
    if not os.path.exists(result_wardrobe_dir):
        os.makedirs(result_wardrobe_dir)

    for sim_name in similiar_names:
        # Search for image files with the similar name in the input_images directory
        found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(input_images_dir, sim_name + ext)
            if os.path.exists(img_path):
                print(f"Found image for {sim_name}: {img_path}")
                shutil.copy(img_path, result_wardrobe_dir)
                found = True
                break

        if not found:
            print(f"No image found for {sim_name} in {input_images_dir}")

    # Generate image and save to result_image directory
    generate_image(extracted_text, suggested_outfits)
    
    # Ensure the result_image directory exists
    result_image_dir = 'result_image'
    if not os.path.exists(result_image_dir):
        os.makedirs(result_image_dir)
    
    return suggested_outfits

# Streamlit app
def main():
    st.title("Celebrity Attire Design")
    
    # Text input for customer context
    customer_context = st.text_input("Enter customer context:")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded image to a temporary location
        temp_image_path = os.path.join("new_input", uploaded_file.name)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
        
        # Separate buttons for processing and asking AI designer
        if st.button("Process Image"):
            suggested_outfits = process_results(temp_image_path, customer_context)
            print(suggested_outfits)
            display_results_streamlit('result_wardrobe', suggested_outfits)

        if st.button("Ask AI Designer"):
            suggested_outfits = process_results(temp_image_path, customer_context)
            print(suggested_outfits)
            display_results_streamlit('result_image', suggested_outfits)
if __name__ == "__main__":
    main()
