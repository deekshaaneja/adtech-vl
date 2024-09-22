# Vision-Language (VL) Model UI (Based on Qwen2 VL)

This project is a web application that allows users to upload images and get predictions from a Vision-Language (VL) model fine-tuned on custom images. The model is based on **Qwen2 VL** and has been trained on data stored in `combined_conversations.json`.

## Features
- Upload an image via the UI.
- Send the image to a fine-tuned Qwen2 VL model for inference.
- Display the model's prediction/output.

## Technologies Used
- **Streamlit**: For building the UI.
- **Transformers**: For loading and using the Qwen2 VL model.
- **Pillow (PIL)**: To handle image uploads and processing.
- **PyTorch**: To run the Qwen2 VL model inference.

## Setup and Installation

### Prerequisites
- Python 3.8 or above
- pip (Python package manager)
- GPU (Optional but recommended for faster inference)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vl-model-ui.git
   cd vl-model-ui
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include:
   ```text
   streamlit
   torch
   transformers
   Pillow
   ```

4. **Prepare the Qwen2 VL model:**
   Make sure your fine-tuned Qwen2 VL model is accessible. Either place it in a local directory or load it from a model hub. Update the model loading path in the `app.py` file:
   ```python
   model = Qwen2VLForConditionalGeneration.from_pretrained('path_to_your_trained_model')
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

   This will launch the app on `localhost:8501` by default. You can access the app via your web browser.

## Project Structure
```
vl-model-ui/
│
├── app.py                   # Main Streamlit application
├── model_inference.py        # Inference logic for the fine-tuned Qwen2 VL model
├── images/                   # Directory containing custom training images
├── combined_conversations.json # JSON file with learning data for fine-tuning
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Usage

1. Launch the application by running the following command:
   ```bash
   streamlit run app.py
   ```

2. On the web interface:
   - Upload an image by clicking the "Choose an image..." button.
   - Click on the "Get Prediction" button to receive a response from the Qwen2 VL model.
   - The model’s output will be displayed below the image.

## Example

Once the app is running, the workflow is as follows:
1. **Image Upload**: Select an image from your local machine.
2. **Prediction**: Click the "Get Prediction" button to send the image to the fine-tuned Qwen2 VL model and get a response.

## Customization

- **Model Configuration**: If you want to fine-tune or replace the model, update the `model_inference.py` script to load the new model and adjust the input/output processing accordingly.
- **Training Data**: Your model has been trained on custom images stored in the `images` folder and the learning data from `combined_conversations.json`.

## Future Enhancements

- Add support for batch image uploads.
- Deploy the application to cloud platforms such as Streamlit Cloud, Heroku, or AWS.
- Implement additional features such as image classification or caption generation.

## Contact
For any queries or support, contact [deekshaaneja@gmail.com].
