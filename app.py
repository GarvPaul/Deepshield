
import streamlit as st
import torch
import cv2
import numpy as np
import lpips
import io

from PIL import Image, ImageDraw
from pathlib import Path
from fgsm import generate_protected_image


# PAGE CONFIG
st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="wide"
)


# METRICS
def calculate_psnr(original, protected):
    mse = np.mean(
        (original.astype(np.float64) -
         protected.astype(np.float64)) ** 2
    )

    if mse == 0:
        return float("inf")

    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(original, protected):
    original = original.astype(np.float64)
    protected = protected.astype(np.float64)

    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(protected, (11, 11), 1.5)

    sigma1 = cv2.GaussianBlur(
        original * original, (11, 11), 1.5
    ) - mu1 * mu1

    sigma2 = cv2.GaussianBlur(
        protected * protected, (11, 11), 1.5
    ) - mu2 * mu2

    sigma12 = cv2.GaussianBlur(
        original * protected, (11, 11), 1.5
    ) - mu1 * mu2

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (
        (2 * mu1 * mu2 + c1) *
        (2 * sigma12 + c2)
    )

    denominator = (
        (mu1 * mu1 + mu2 * mu2 + c1) *
        (sigma1 + sigma2 + c2)
    )

    return float(np.mean(numerator / denominator))


@st.cache_resource
def load_lpips_model():
    model = lpips.LPIPS(net="alex")
    model.eval()
    return model


def calculate_lpips(original, protected):
    original = original.resize((64, 64)).convert("RGB")
    protected = protected.resize((64, 64)).convert("RGB")

    a = np.array(original).astype(np.float32) / 127.5 - 1
    b = np.array(protected).astype(np.float32) / 127.5 - 1

    a = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)

    model = load_lpips_model()

    with torch.no_grad():
        score = model(a, b)

    return float(score.item())


# DATASET
BASE_DIR = Path(__file__).resolve().parent

dataset_path = BASE_DIR / "datasets" / "FFHQ" / "images"

if not dataset_path.exists():
    dataset_path = BASE_DIR / "image" / "images"

image_files = sorted(
    list(dataset_path.glob("*.jpg")) +
    list(dataset_path.glob("*.jpeg")) +
    list(dataset_path.glob("*.png"))
) if dataset_path.exists() else []


# SESSION STATE
if "processed_results" not in st.session_state:
    st.session_state.processed_results = {}


# HEADER
st.title("🛡️ DeepShield")

st.subheader(
    "AI-Powered Defence System Against Deepfake Generation"
)

st.write(
    "Experimental adversarial perturbation of facial images."
)

st.sidebar.title("DeepShield")
st.sidebar.info("Major Project-I | Group RG21")

st.sidebar.success(
    f"Dataset images found: {len(image_files):,}"
)

st.divider()


# IMAGE SELECTION
st.header("1. Select FFHQ Image")

if not image_files:
    st.error("FFHQ dataset folder not found.")
    st.stop()

total_images = len(image_files)

st.success(
    f"Total dataset images available: {total_images:,}"
)

# SEARCH ALL DATASET IMAGES
search_query = st.text_input(
    "🔍 Search image by filename",
    placeholder="Enter image name, e.g. 00001"
)

if search_query.strip():

    filtered_images = [
        path for path in image_files
        if search_query.lower() in path.name.lower()
    ]

    if not filtered_images:
        st.warning("No matching images found.")
        st.stop()

    st.caption(
        f"Search results: {len(filtered_images):,} images"
    )

else:

    # PAGINATION
    page_size = 1000

    total_pages = (
        total_images + page_size - 1
    ) // page_size

    page_number = st.selectbox(
        "Browse dataset pages",
        options=list(range(1, total_pages + 1)),
        format_func=lambda page: (
            f"Page {page} "
            f"({(page - 1) * page_size + 1} - "
            f"{min(page * page_size, total_images)})"
        )
    )

    start_index = (page_number - 1) * page_size
    end_index = min(
        start_index + page_size,
        total_images
    )

    filtered_images = image_files[start_index:end_index]

# SELECT IMAGE
selected_path = st.selectbox(
    "Choose your image",
    options=filtered_images,
    format_func=lambda path: path.name
)

st.caption(
    f"Selected image: {selected_path.name}"
)

try:
    original = Image.open(selected_path).convert("RGB")

except Exception as e:
    st.error(f"Could not open image: {e}")
    st.stop()


# ORIGINAL IMAGE
st.subheader("Original Image")

st.image(
    original,
    width=400
)


# FACE DETECTION
st.header("2. Face Detection")

image_array = np.array(original)

gray = cv2.cvtColor(
    image_array,
    cv2.COLOR_RGB2GRAY
)

cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(cascade_path)

faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

face_output = original.copy()

draw = ImageDraw.Draw(face_output)

for x, y, w, h in faces:

    draw.rectangle(
        [x, y, x + w, y + h],
        outline="lime",
        width=4
    )

st.image(
    face_output,
    caption="Face Detection Output",
    width=400
)

st.success(
    f"Faces detected: {len(faces)}"
)


# FGSM PROCESSING
st.divider()

st.header("3. Experimental FGSM Processing")

epsilon = st.slider(
    "Perturbation Strength (Epsilon)",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001
)

result_key = f"{selected_path}_{epsilon:.3f}"

if st.button(
    "Generate FGSM Result",
    use_container_width=True
):

    with st.spinner(
        "Processing image using trained model..."
    ):

        resized = original.resize((64, 64))

        image_np = (
            np.array(resized).astype(np.float32) / 255.0
        )

        image_tensor = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        protected_tensor = generate_protected_image(
            image_tensor,
            epsilon
        )

        result_np = (
            protected_tensor.squeeze(0)
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy() * 255
        ).clip(0, 255).astype(np.uint8)

        result_image = Image.fromarray(result_np)

        original_array = np.array(resized)

        protected_array = np.array(result_image)

        psnr = calculate_psnr(
            original_array,
            protected_array
        )

        ssim = calculate_ssim(
            original_array,
            protected_array
        )

        try:

            lpips_score = calculate_lpips(
                resized,
                result_image
            )

            lpips_error = None

        except Exception as e:

            lpips_score = None
            lpips_error = str(e)

        st.session_state.processed_results[result_key] = {
            "image": result_image,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_score,
            "lpips_error": lpips_error
        }


# DISPLAY RESULT
result = st.session_state.processed_results.get(
    result_key
)

if result:

    st.header("4. Image Quality Evaluation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "PSNR",
        f"{result['psnr']:.2f} dB"
    )

    c2.metric(
        "SSIM",
        f"{result['ssim']:.4f}"
    )

    if result["lpips"] is not None:

        c3.metric(
            "LPIPS",
            f"{result['lpips']:.4f}"
        )

    else:

        c3.metric(
            "LPIPS",
            "Unavailable"
        )

        st.warning(
            "LPIPS could not be calculated."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original.resize((64, 64)),
            caption="Original Image",
            width="stretch"
        )

    with col2:

        st.image(
            result["image"],
            caption="Experimental FGSM Output",
            width="stretch"
        )

    buffer = io.BytesIO()

    result["image"].save(
        buffer,
        format="PNG"
    )

    st.download_button(
        "Download FGSM Result",
        data=buffer.getvalue(),
        file_name=(
            f"deepshield_{selected_path.stem}_"
            f"epsilon_{epsilon:.3f}.png"
        ),
        mime="image/png",
        use_container_width=True
    )

else:

    st.info(
        "Select an image and click Generate FGSM Result."
    )


# FOOTER
st.divider()

st.header("5. Processing Workflow")

st.markdown(
    "FFHQ Image Selection → Face Detection → "
    "Trained Experimental Classifier → FGSM Perturbation → "
    "Image Quality Evaluation"
)

st.caption(
    "Experimental prototype. Deepfake resistance has not "
    "yet been validated against actual deepfake generation systems."
)

st.caption(
    "DeepShield | Major Project-I | Group RG21"
)
