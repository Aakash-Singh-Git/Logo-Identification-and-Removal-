import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
import threading
import torch
from io import BytesIO
import json
from datetime import datetime
import requests
import base64
from io import BytesIO
from PIL import Image
import openai
import tempfile
import requests
from dotenv import load_dotenv
load_dotenv()

class Config:
    """Configuration class for the application"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "train", "logo_final", "weights", "best.pt")
    OUTPUT_FOLDER   = os.path.join(BASE_DIR, "mask and output")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Default settings
    DEFAULT_MASK_EXPANSION = 10
    IMAGE_PREVIEW_SIZE = (300, 300)
    RESULT_PREVIEW_SIZE = (600, 400)
    COMPARISON_SIZE = (350, 350)
    
    # AI API Settings (configure these with your API keys)
    STABILITYAI_API_KEY = os.getenv("STABILITYAI_API_KEY", "")
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

class AIInpainter:
    """Handles AI inpainting using various APIs"""
    def __init__(self):
        self.stability_available = bool(Config.STABILITYAI_API_KEY)
        self.replicate_available = bool(Config.REPLICATE_API_TOKEN)
        self.openai_available = bool(Config.OPENAI_API_KEY)
    
    def stability_ai_inpaint(self, image, mask):
        """Use Stability AI for inpainting - FIXED ENDPOINT"""
        if not self.stability_available:
            return None
    
        try:
            # Convert PIL images to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
        
            mask_buffer = BytesIO()
            mask.save(mask_buffer, format='PNG')
            mask_buffer.seek(0)
        
            # FIXED: Use correct endpoint and field names
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {Config.STABILITYAI_API_KEY}"
                },
                files={
                    "image": ("image.png", img_buffer, "image/png"),  # FIXED: correct field name
                    "mask": ("mask.png", mask_buffer, "image/png")     # FIXED: correct field name
                },
                data={
                    "prompt": "remove the object, fill with appropriate background",
                    "mode": "inpaint"
                }
            )
        
            if response.status_code == 200:
                data = response.json()
                if data.get("image"):
                    # Decode the base64 image
                    image_data = base64.b64decode(data["image"])
                    return Image.open(BytesIO(image_data))
            else:
                print(f"Stability AI error: {response.status_code} - {response.text}")
            
        except Exception as e:
            print(f"Stability AI inpainting error: {e}")
    
        return None

    def replicate_inpaint(self, image, mask):
        """Use Replicate for inpainting - FIXED MODEL VERSION"""
        if not self.replicate_available:
            return None
        
        try:
            import replicate
            
            # Set API token
            os.environ["REPLICATE_API_TOKEN"] = Config.REPLICATE_API_TOKEN
            
            # Convert images to base64
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            mask_buffer = BytesIO()
            mask.save(mask_buffer, format='PNG')
            mask_str = base64.b64encode(mask_buffer.getvalue()).decode()
            
            # FIXED: Use working Stable Diffusion inpainting model
            output = replicate.run(
                "andreasjansson/stable-diffusion-inpainting:c28b92a7ecd66eee4aefcd8a94eb9e7f6c3805d5f06038165407fb5cb355ba67",
                input={
                    "image": f"data:image/png;base64,{img_str}",
                    "mask": f"data:image/png;base64,{mask_str}",
                    "prompt": "clean background, remove object completely",
                    "num_inference_steps": 20,
                    "guidance_scale": 10
                }
            )
            
            if output:
                # Download the result image
                response = requests.get(output)
                return Image.open(BytesIO(response.content))
                
        except Exception as e:
            print(f"Replicate inpainting error: {e}")
        
        return None
    
    def openai_inpaint(self, image, mask):
        if not self.openai_available:
            return None

        try:
            import openai
            from io import BytesIO
            from PIL import Image
            import requests
            import tempfile
            import os

            # Set API key based on OpenAI version
            if hasattr(openai, 'api_key'):
                openai.api_key = Config.OPENAI_API_KEY
            else:
                # For newer versions of openai library
                client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

            # Resize and square the image
            if image.width != image.height:
                size = min(image.width, image.height)
                image = image.crop((0, 0, size, size))
            if image.width > 1024:
                image = image.resize((1024, 1024), Image.LANCZOS)

            # Resize and convert the mask
            mask = mask.resize(image.size, Image.LANCZOS)
            mask = mask.convert("RGB")

            # Use temporary files approach for better compatibility
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_temp:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as mask_temp:
                    # Save images to temp files
                    image.save(img_temp.name, format='PNG')
                    mask.save(mask_temp.name, format='PNG')
                    
                    # Wait for files to be written
                    img_temp.flush()
                    mask_temp.flush()
                    
                    try:
                        # Open files and make API call
                        with open(img_temp.name, 'rb') as img_file:
                            with open(mask_temp.name, 'rb') as mask_file:
                                if hasattr(openai, 'api_key'):
                                    # Old API version
                                    response = openai.Image.create_edit(
                                        image=img_file,
                                        mask=mask_file,
                                        prompt="clean empty space, remove logo",
                                        n=1,
                                        size="1024x1024"
                                    )
                                else:
                                    # New API version
                                    response = client.images.edit(
                                        image=img_file,
                                        mask=mask_file,
                                        prompt="clean empty space, remove logo",
                                        n=1,
                                        size="1024x1024"
                                    )
                                
                                if response and 'data' in response and len(response['data']) > 0:
                                    image_url = response['data'][0]['url']
                                    result_response = requests.get(image_url)
                                    if result_response.status_code == 200:
                                        return Image.open(BytesIO(result_response.content))
                                    else:
                                        print(f"Failed to download result: {result_response.status_code}")
                                else:
                                    print("No data returned from OpenAI API")
                                    
                    finally:
                        # Clean up temp files
                        try:
                            os.unlink(img_temp.name)
                            os.unlink(mask_temp.name)
                        except:
                            pass

        except Exception as e:
            print(f"OpenAI inpainting error: {e}")

        return None


class CloudVisionInpainter:
    """Alternative inpainting solution using AI services"""
    def __init__(self):
        self.services = {
            'clipdrop': {
                'name': 'ClipDrop',
                'api_key': '',#add your clipdrop API key
                'endpoint': 'https://clipdrop-api.co/cleanup/v1'
            }
        }
    
    def clipdrop_inpaint(self, image, mask):
        """Use ClipDrop API for inpainting"""
        try:
            if not self.services['clipdrop']['api_key']:
                return None
            
            # Convert images to bytes
            img_buffer = BytesIO()
            # ClipDrop works better with JPG for cleanup
            image_rgb = image.convert('RGB')
            image_rgb.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            
            mask_buffer = BytesIO()
            # Ensure mask is proper format (white areas will be removed)
            mask_rgb = mask.convert('RGB')
            mask_rgb.save(mask_buffer, format='JPEG', quality=95)
            mask_buffer.seek(0)
            
            # Make API request
            response = requests.post(
                self.services['clipdrop']['endpoint'],
                files={
                    'image_file': ('image.jpg', img_buffer, 'image/jpeg'),
                    'mask_file': ('mask.jpg', mask_buffer, 'image/jpeg')
                },
                headers={
                    'x-api-key': self.services['clipdrop']['api_key']
                }
            )
            
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                print(f"ClipDrop error: {response.status_code}")
                if response.status_code == 402:
                    print("Payment required - check your ClipDrop API credits")
                
        except Exception as e:
            print(f"ClipDrop inpainting error: {e}")
        
        return None
    
    def huggingface_inpaint(self, image, mask):
        """Use Hugging Face inference API for inpainting - FIXED"""
        try:
            if not Config.HUGGINGFACE_TOKEN or Config.HUGGINGFACE_TOKEN == "hf_your_token_here":
                print("HuggingFace token not configured properly")
                return None
            
            # FIXED: Use the correct API endpoint and format
            API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
            
            headers = {
                "Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Convert to base64 - this is the format HF expects
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            mask_buffer = BytesIO()
            mask.save(mask_buffer, format='PNG')
            mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode()
            
            # FIXED: Proper JSON payload format
            payload = {
                "inputs": {
                    "image": img_b64,
                    "mask": mask_b64,
                    "prompt": "clean background, remove object"
                },
                "parameters": {
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if result and isinstance(result, list) and len(result) > 0:
                    # HF returns base64 encoded image
                    img_data = base64.b64decode(result[0])
                    return Image.open(BytesIO(img_data))
            else:
                print(f"Hugging Face error: {response.status_code} - {response.text}")
                if response.status_code == 403:
                    print("Access denied - check if your token has permission to access this model")
                
        except Exception as e:
            print(f"Hugging Face inpainting error: {e}")
    
        return None

# Alternative HuggingFace approach if the above doesn't work
def huggingface_inpaint_alternative(image, mask):
    """Alternative HuggingFace inpainting using requests-toolbelt for multipart"""
    try:
        from requests_toolbelt.multipart.encoder import MultipartEncoder
        
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
        
        # Prepare images as files
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        mask_buffer = BytesIO()
        mask.save(mask_buffer, format='PNG')
        mask_buffer.seek(0)
        
        # Create multipart form data
        multipart_data = MultipartEncoder(
            fields={
                'image': ('image.png', img_buffer, 'image/png'),
                'mask': ('mask.png', mask_buffer, 'image/png'),
                'prompt': 'remove the marked object and fill with background'
            }
        )
        
        response = requests.post(
            API_URL,
            data=multipart_data,
            headers={
                'Authorization': f'Bearer {Config.HUGGINGFACE_TOKEN}',
                'Content-Type': multipart_data.content_type
            }
        )
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Hugging Face alternative error: {response.status_code} - {response.text}")
            
    except ImportError:
        print("requests-toolbelt not installed. Install with: pip install requests-toolbelt")
    except Exception as e:
        print(f"Hugging Face alternative error: {e}")
    
    return None
    
class YOLODetector:
    """Handles YOLO model operations"""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Successfully loaded YOLO model from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            return False
    
    def detect(self, image_path):
        """Detect logos in image"""
        if self.model is None:
            return None
        
        try:
            results = self.model(image_path)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            return boxes
        except Exception as e:
            print(f"Error detecting logos: {e}")
            return None

class LogoRemover:
    """Main application class"""
    def __init__(self, root):
        self.root = root
        self.root.title("Logo Remover using YOLO + AI APIs")
        self.root.geometry("1000x700")
        
        # Initialize models
        self.yolo_detector = YOLODetector(Config.YOLO_MODEL_PATH)
        self.ai_inpainter = AIInpainter()
        self.cloud_inpainter = CloudVisionInpainter()
        
        # Application state
        self.current_image_path = None
        self.current_img = None
        self.current_boxes = None
        self.processing = False
        
        # Setup UI
        self.setup_ui()
        self.check_ai_availability()
    
    def check_ai_availability(self):
        """Check which AI services are available"""
        available_services = []
        
        if self.ai_inpainter.stability_available:
            available_services.append("Stability AI")
        if self.ai_inpainter.replicate_available:
            available_services.append("Replicate")
        if self.ai_inpainter.openai_available:
            available_services.append("OpenAI DALL-E")
        if self.cloud_inpainter.services['clipdrop']['api_key']:
            available_services.append("ClipDrop")
        
        if available_services:
            self.status_var.set(f"Status: Available services: {', '.join(available_services)}")
        else:
            self.status_var.set("Status: No AI services configured. Only OpenCV available.")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create UI sections
        self.create_controls_section(main_frame)
        self.create_method_section(main_frame)
        self.create_api_config_section(main_frame)
        self.create_mask_section(main_frame)
        self.create_images_section(main_frame)
        self.create_progress_section(main_frame)
    
    def create_controls_section(self, parent):
        """Create the top controls section"""
        controls_frame = tk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Select image button
        self.select_btn = tk.Button(
            controls_frame, 
            text="Select Image", 
            command=self.select_image, 
            width=15,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_btn = tk.Button(
            controls_frame, 
            text="Process", 
            command=self.process_image, 
            state=tk.DISABLED, 
            width=15,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = tk.Button(
            controls_frame,
            text="Clear",
            command=self.clear_all,
            width=15,
            bg='#f44336',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Status: Loading...")
        self.status_label = tk.Label(
            controls_frame, 
            textvariable=self.status_var, 
            anchor="w",
            font=('Arial', 9)
        )
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
    
    def create_method_section(self, parent):
        """Create the inpainting method selection section"""
        self.method_frame = tk.LabelFrame(parent, text="Inpainting Method", font=('Arial', 10, 'bold'))
        self.method_frame.pack(fill=tk.X, pady=5)
        
        self.method_var = tk.StringVar(value="auto")
        
        methods = [
            ("Auto (AI if available, else OpenCV)", "auto"),
            ("Stability AI", "stability"),
            ("Replicate", "replicate"),
            ("OpenAI DALL-E", "openai"),
            ("ClipDrop", "clipdrop"),
            ("Hugging Face", "huggingface"),
            ("OpenCV Only", "opencv")
        ]
        
        for text, value in methods:
            radio = tk.Radiobutton(
                self.method_frame, 
                text=text, 
                variable=self.method_var, 
                value=value,
                font=('Arial', 9)
            )
            radio.pack(side=tk.LEFT, padx=5)
    
    def create_api_config_section(self, parent):
        """Create API configuration section"""
        api_frame = tk.LabelFrame(parent, text="API Configuration", font=('Arial', 10, 'bold'))
        api_frame.pack(fill=tk.X, pady=5)
        
        # Add button to configure API keys
        config_btn = tk.Button(
            api_frame,
            text="Configure API Keys",
            command=self.show_api_config,
            font=('Arial', 9)
        )
        config_btn.pack(side=tk.LEFT, padx=5)
        
        # API status
        self.api_status_var = tk.StringVar(value="No API keys configured")
        api_status_label = tk.Label(
            api_frame,
            textvariable=self.api_status_var,
            font=('Arial', 9),
            fg='gray'
        )
        api_status_label.pack(side=tk.LEFT, padx=10)
    
    def show_api_config(self):
        """Show API configuration dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("API Key Configuration")
        dialog.geometry("500x400")
        dialog.grab_set()
        
        # Create input fields for API keys
        fields = {
            'Stability AI': 'STABILITYAI_API_KEY',
            'Replicate': 'REPLICATE_API_TOKEN',
            'OpenAI': 'OPENAI_API_KEY',
        }
        
        entries = {}
        
        for i, (name, key) in enumerate(fields.items()):
            frame = tk.Frame(dialog)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(frame, text=f"{name}:", width=15, anchor='w').pack(side=tk.LEFT)
            entry = tk.Entry(frame, show='*', width=40)
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, getattr(Config, key, ''))
            entries[key] = entry
        
        # Instructions
        instructions = tk.Text(dialog, height=6, wrap=tk.WORD)
        instructions.pack(fill=tk.X, padx=10, pady=5)
        instructions.insert('1.0', """
API Key Sources:
• Stability AI: Get from https://platform.stability.ai/
• Replicate: Get from https://replicate.com/
• OpenAI: Get from https://openai.com/

Note: You only need to configure the APIs you want to use.
Leave others blank if not needed.
        """)
        instructions.config(state=tk.DISABLED)
        
        # Buttons
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=10)
        
        def save_config():
            for key, entry in entries.items():
                setattr(Config, key, entry.get().strip())
            
            # Reinitialize AI services
            self.ai_inpainter = AIInpainter()
            self.cloud_inpainter = CloudVisionInpainter()
            self.check_ai_availability()
            dialog.destroy()
        
        tk.Button(btn_frame, text="Save", command=save_config, 
                 bg='#4CAF50', fg='white').pack(side=tk.RIGHT, padx=10)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
    
    def create_mask_section(self, parent):
        """Create the mask expansion control section"""
        mask_frame = tk.LabelFrame(parent, text="Mask Settings", font=('Arial', 10, 'bold'))
        mask_frame.pack(fill=tk.X, pady=5)
        
        # Mask expansion control
        tk.Label(mask_frame, text="Mask Expansion:", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        self.mask_expansion = tk.IntVar(value=Config.DEFAULT_MASK_EXPANSION)
        
        expansion_spinbox = tk.Spinbox(
            mask_frame, 
            from_=0, 
            to=50, 
            width=10, 
            textvariable=self.mask_expansion,
            command=self.update_mask_preview
        )
        expansion_spinbox.pack(side=tk.LEFT, padx=5)
        tk.Label(mask_frame, text="pixels", font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Auto-adjust mask button
        auto_mask_btn = tk.Button(
            mask_frame,
            text="Auto-adjust Mask",
            command=self.auto_adjust_mask,
            font=('Arial', 9)
        )
        auto_mask_btn.pack(side=tk.LEFT, padx=10)
    
    def create_images_section(self, parent):
        """Create the images display section"""
        self.images_frame = tk.Frame(parent)
        self.images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Original image frame
        orig_frame = tk.LabelFrame(self.images_frame, text="Original Image with Bounding Boxes", font=('Arial', 10, 'bold'))
        orig_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.original_img_label = tk.Label(orig_frame, bg='white')
        self.original_img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Mask frame
        mask_frame = tk.LabelFrame(self.images_frame, text="Generated Mask", font=('Arial', 10, 'bold'))
        mask_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.mask_img_label = tk.Label(mask_frame, bg='white')
        self.mask_img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Result frame
        result_frame = tk.LabelFrame(self.images_frame, text="Result (Logo Removed)", font=('Arial', 10, 'bold'))
        result_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        self.result_img_label = tk.Label(result_frame, bg='white')
        self.result_img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure grid weights
        self.images_frame.columnconfigure(0, weight=1)
        self.images_frame.columnconfigure(1, weight=1)
        self.images_frame.rowconfigure(0, weight=1)
        self.images_frame.rowconfigure(1, weight=1)
    
    def create_progress_section(self, parent):
        """Create the progress bar section"""
        self.progress_frame = tk.Frame(parent)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = tk.Label(self.progress_frame, textvariable=self.progress_var, font=('Arial', 9))
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.progress_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def clear_all(self):
        """Clear all images and reset the application state"""
        self.current_image_path = None
        self.current_img = None
        self.current_boxes = None
        
        # Clear image displays
        self.original_img_label.config(image="")
        self.mask_img_label.config(image="")
        self.result_img_label.config(image="")
        
        # Reset UI state
        self.process_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Ready")
        self.progress_var.set("Ready")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        
        self.current_image_path = file_path
        self.status_var.set(f"Status: Loaded image {os.path.basename(file_path)}")
        
        # Display original image
        self.current_img = Image.open(file_path).convert("RGB")
        
        # Detect logo with YOLO
        self.current_boxes = self.yolo_detector.detect(file_path)
        
        if self.current_boxes is not None and len(self.current_boxes) > 0:
            # Draw bounding boxes on the original image for visualization
            img_with_boxes = self.current_img.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            
            for box in self.current_boxes:
                x1, y1, x2, y2 = box
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                # Add label
                draw.text((x1, y1-25), "Logo", fill="red", font=None)
            
            self.display_image(img_with_boxes, self.original_img_label, Config.IMAGE_PREVIEW_SIZE)
            
            # Generate and display mask preview
            self.update_mask_preview()
            
            self.process_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Status: Detected {len(self.current_boxes)} logo(s). Ready to process.")
        else:
            self.display_image(self.current_img, self.original_img_label, Config.IMAGE_PREVIEW_SIZE)
            self.mask_img_label.config(image="")
            self.process_btn.config(state=tk.DISABLED)
            self.status_var.set("Status: No logos detected in this image.")
    
    def update_mask_preview(self):
        """Update the mask preview when expansion value changes"""
        if self.current_boxes is None or len(self.current_boxes) == 0:
            return
        
        mask = self.create_mask_from_boxes(self.mask_expansion.get())
        self.display_image(mask, self.mask_img_label, Config.IMAGE_PREVIEW_SIZE)
    
    def auto_adjust_mask(self):
        """Auto-adjust mask expansion based on logo size"""
        if self.current_boxes is None or len(self.current_boxes) == 0:
            return
        
        # Calculate average logo size
        total_area = 0
        for box in self.current_boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            total_area += area
        
        avg_area = total_area / len(self.current_boxes)
        avg_size = np.sqrt(avg_area)
        
        # Set expansion based on logo size
        expansion = max(5, min(50, int(avg_size * 0.1)))
        self.mask_expansion.set(expansion)
        self.update_mask_preview()
    
    def display_image(self, img, label, size=None):
        """Display PIL image in a tkinter label with optional resizing"""
        if size:
            # Calculate aspect ratio
            aspect_ratio = img.width / img.height
            if img.width > img.height:
                new_width = size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = size[1]
                new_width = int(new_height * aspect_ratio)
            
            display_img = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            display_img = img
        
        tk_img = ImageTk.PhotoImage(display_img)
        label.config(image=tk_img)
        label.image = tk_img  # Keep a reference to prevent garbage collection
    
    def process_image(self):
        """Process the image to remove the logo"""
        if not self.current_img or self.current_boxes is None or len(self.current_boxes) == 0:
            messagebox.showerror("Error", "No image or detections available")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.progress_var.set("Processing...")
        
        # Run processing in a separate thread
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        """Process image in a separate thread"""
        try:
            method = self.method_var.get()
            self.root.after(0, lambda: self.status_var.set(f"Status: Processing with {method} method..."))
            
            # Generate unique filenames
            img_uuid = uuid.uuid4().hex
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"output_{timestamp}_{img_uuid}"
            output_path = os.path.join(Config.OUTPUT_FOLDER, f"{base_name}.png")
            
            # Create mask from bounding boxes
            expansion = self.mask_expansion.get()
            mask = self.create_mask_from_boxes(expansion)
            
            # Save mask for reference
            mask_path = os.path.join(Config.OUTPUT_FOLDER, f"mask_{base_name}.png")
            mask.save(mask_path)
            
            # Apply inpainting based on selected method
            result = None
            
            # Update progress UI
            self.root.after(0, lambda: self.progress_var.set("Applying inpainting..."))
            
            if method == "auto":
                # Try AI methods first, fall back to OpenCV if not available
                if self.ai_inpainter.stability_available:
                    result = self.ai_inpainter.stability_ai_inpaint(self.current_img, mask)
                elif self.ai_inpainter.replicate_available:
                    result = self.ai_inpainter.replicate_inpaint(self.current_img, mask)
                elif self.ai_inpainter.openai_available:
                    result = self.ai_inpainter.openai_inpaint(self.current_img, mask)
                
                # If all AI methods failed, use OpenCV
                if result is None:
                    result = self.apply_opencv_inpainting(self.current_img, mask)
            elif method == "stability":
                result = self.ai_inpainter.stability_ai_inpaint(self.current_img, mask)
            elif method == "replicate":
                result = self.ai_inpainter.replicate_inpaint(self.current_img, mask)
            elif method == "openai":
                result = self.ai_inpainter.openai_inpaint(self.current_img, mask)
            elif method == "clipdrop":
                result = self.cloud_inpainter.clipdrop_inpaint(self.current_img, mask)
            elif method == "huggingface":
                result = self.cloud_inpainter.huggingface_inpaint(self.current_img, mask)
            else:  # Default to OpenCV
                result = self.apply_opencv_inpainting(self.current_img, mask)
            
            # If inpainting was successful
            if result:
                # Save the result
                result.save(output_path)
                
                # Update UI with the result
                self.root.after(0, lambda: self.display_image(result, self.result_img_label, Config.RESULT_PREVIEW_SIZE))
                self.root.after(0, lambda: self.status_var.set(f"Status: Done! Result saved to {output_path}"))
                
                # Show comparison dialog
                self.root.after(0, lambda: self.show_comparison_dialog(self.current_img, result))
            else:
                self.root.after(0, lambda: self.status_var.set("Status: Inpainting failed. Try another method."))
                
        except Exception as e:
            print(f"Processing error: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Status: Error: {str(e)}"))
        finally:
            # Reset UI state
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.processing = False
    
    def create_mask_from_boxes(self, expansion=0):
        """Create a mask from detected bounding boxes with optional expansion"""
        if self.current_img is None or self.current_boxes is None:
            return None
        
        # Create a blank mask with the same size as the image
        mask = Image.new('L', self.current_img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw white rectangles on the mask for each logo detection
        for box in self.current_boxes:
            x1, y1, x2, y2 = box
            
            # Expand the box if requested
            if expansion > 0:
                x1 = max(0, x1 - expansion)
                y1 = max(0, y1 - expansion)
                x2 = min(self.current_img.width, x2 + expansion)
                y2 = min(self.current_img.height, y2 + expansion)
            
            draw.rectangle([(x1, y1), (x2, y2)], fill=255)
        
        return mask
    
    def apply_opencv_inpainting(self, image, mask):
        """Apply OpenCV inpainting method"""
        try:
            # Convert PIL images to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            mask_cv = np.array(mask.convert('L'))
            
            # Apply inpainting
            result_cv = cv2.inpaint(img_cv, mask_cv, 3, cv2.INPAINT_TELEA)
            
            # Convert back to PIL
            result = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
            return result
        except Exception as e:
            print(f"OpenCV inpainting error: {e}")
            return None
    
    def show_comparison_dialog(self, original, result):
        """Show a before/after comparison dialog"""
        if original is None or result is None:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Before/After Comparison")
        dialog.geometry("800x500")
        
        # Create comparison frame
        comparison_frame = tk.Frame(dialog)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original image
        orig_frame = tk.LabelFrame(comparison_frame, text="Original Image")
        orig_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        orig_label = tk.Label(orig_frame)
        orig_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_image(original, orig_label, Config.COMPARISON_SIZE)
        
        # Result image
        result_frame = tk.LabelFrame(comparison_frame, text="Logo Removed")
        result_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        result_label = tk.Label(result_frame)
        result_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_image(result, result_label, Config.COMPARISON_SIZE)
        
        # Configure grid weights
        comparison_frame.columnconfigure(0, weight=1)
        comparison_frame.columnconfigure(1, weight=1)
        comparison_frame.rowconfigure(0, weight=1)
        
        # Add save buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_to_clipboard():
            # Convert PIL image to a format suitable for clipboard
            output = BytesIO()
            result.save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove BMP header
            output.close()
            
            # Copy to clipboard (Windows only)
            if os.name == 'nt':
                import win32clipboard
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                messagebox.showinfo("Success", "Image copied to clipboard")
            else:
                messagebox.showerror("Error", "Clipboard functionality is only available on Windows")
        
        def save_as():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG Image", "*.png"),
                    ("JPEG Image", "*.jpg"),
                    ("All Files", ".")
                ]
            )
            if file_path:
                result.save(file_path)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
        
        tk.Button(button_frame, text="Copy to Clipboard", command=save_to_clipboard).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save As...", command=save_as).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)


def main():
    root = tk.Tk()
    app = LogoRemover(root)
    root.mainloop()

if __name__ == "__main__":
    main()