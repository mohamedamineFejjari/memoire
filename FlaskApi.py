import os
import traceback
import uuid
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.utils import to_categorical, img_to_array, load_img
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from werkzeug.utils import secure_filename
import io
from PIL import Image
import shutil
import requests
from urllib.parse import urlparse

# --- Config ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def configure_tensorflow_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("TensorFlow GPU check: no GPU detected. Training will run on CPU.")
        return []

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not set memory growth for {gpu.name}: {e}")

    print(f"TensorFlow GPU check: detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    return gpus


AVAILABLE_GPUS = configure_tensorflow_gpu()

# --- Utility function to handle path or URL ---
def get_file_from_path_or_url(identifier, upload_folder, allowed_extensions=None):
    if not identifier:
        return None, "Identifier is empty"

    if identifier.startswith('http://') or identifier.startswith('https://'):
        # It's a URL, download it
        try:
            response = requests.get(identifier, stream=True)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Get filename from URL or generate one
            parsed_url = urlparse(identifier)
            original_filename = os.path.basename(parsed_url.path)
            if not original_filename or '.' not in original_filename:
                # If no clear filename in URL, generate a unique one
                ext = '.tmp' # Default extension
                if allowed_extensions:
                    for allowed_ext in allowed_extensions:
                        if identifier.lower().endswith(allowed_ext):
                            ext = allowed_ext
                            break
                filename = f"{uuid.uuid4()}{ext}"
            else:
                filename = secure_filename(original_filename)

            temp_filepath = os.path.join(upload_folder, filename)
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {identifier} to {temp_filepath}")
            return temp_filepath, None
        except requests.exceptions.RequestException as e:
            return None, f"Failed to download file from URL {identifier}: {e}"
        except Exception as e:
            return None, f"Error processing URL {identifier}: {e}"
    else:
        # It's a local path
        if os.path.exists(identifier):
            return identifier, None
        else:
            return None, f"File not found at local path: {identifier}"

# --- Attack Functions (unchanged, omit for brevity) ---
def fgsm_attack(image, label, model, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    label_tensor = tf.convert_to_tensor([label])
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
    gradient = tape.gradient(loss, image_tensor)
    if gradient is None:
        raise ValueError("Gradient is None. FGSM attack failed.")
    signed_grad = tf.sign(gradient)
    adv_image = image_tensor + epsilon * signed_grad
    return tf.clip_by_value(adv_image[0], 0, 1).numpy()


def pgd_attack(image, label, model, epsilon=0.03, alpha=0.007, iterations=10):
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([label])
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image[np.newaxis, ...])
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        gradient = tape.gradient(loss, adv_image)
        if gradient is None:
            raise ValueError("Gradient is None. PGD attack failed.")
        adv_image = adv_image + alpha * tf.sign(gradient)
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()


def bim_attack(image, label, model, epsilon=0.03, alpha=0.005, iterations=10):
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([label])
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image[np.newaxis, ...])
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        gradient = tape.gradient(loss, adv_image)
        if gradient is None:
            raise ValueError("Gradient is None. BIM attack failed.")
        adv_image = adv_image + alpha * tf.sign(gradient)
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()


def mim_attack(image, label, model, epsilon=0.03, alpha=0.005, iterations=10, decay=1.0):
    adv_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adv_image)
    label_tensor = tf.convert_to_tensor([label])
    momentum = tf.zeros_like(image, dtype=tf.float32)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image[np.newaxis, ...])
            loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
        gradient = tape.gradient(loss, adv_image)
        if gradient is None:
            raise ValueError("Gradient is None. MIM attack failed.")
        gradient = gradient / (tf.reduce_mean(tf.abs(gradient)) + 1e-8)
        momentum = decay * momentum + gradient
        adv_image = adv_image + alpha * tf.sign(momentum)
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image.numpy()


def tanh_space(x):
    return tf.tanh(x) / 2 + 0.5


def cw_attack(image, label, model, confidence=10, learning_rate=0.01, max_iterations=500):
    image_tensor = tf.cast(image, tf.float32)
    initial_w = tf.atanh(tf.clip_by_value(image_tensor * 2 - 1, -1 + 1e-6, 1 - 1e-6))
    w = tf.Variable(initial_w, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate)
    for i in range(max_iterations):
        with tf.GradientTape() as tape:
            tape.watch(w)
            adv_image = tanh_space(w)
            perturbation = adv_image - image_tensor
            loss_l2 = tf.reduce_sum(tf.square(perturbation))
            preds = model(adv_image[np.newaxis, ...])
            target_one_hot = tf.one_hot([label], preds.shape[-1])
            real = tf.reduce_sum(preds * target_one_hot, axis=1)
            other = tf.reduce_max(preds * (1 - target_one_hot) - (target_one_hot * 10000.0), axis=1)
            loss_misclassification = tf.maximum(0.0, other - real + confidence)
            loss = tf.reduce_sum(loss_l2) + tf.reduce_sum(loss_misclassification)
        gradients = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(gradients, [w]))
    return tf.clip_by_value(tanh_space(w).numpy(), 0, 1)


# --- Utility Functions (unchanged, omit for brevity) ---
def load_images_from_folder(folder_path, target_size=(128, 128)):
    images, labels = [], []
    for label_name in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_name)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(label_path, file)
                    try:
                        img = load_img(img_path, target_size=target_size)
                        img_array = img_to_array(img) / 255.0
                        if img_array.shape[-1] == 4:
                            img_array = img_array[..., :3]
                        elif img_array.shape[-1] == 1:
                            img_array = np.concatenate([img_array] * 3, axis=-1)
                        images.append(img_array)
                        labels.append(label_name)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)


def label_to_index(labels):
    mapping = {
        'propre': 0, 'FGSM': 1, 'BIM': 2, 'MIM': 3, 'PGD': 4, 'CW': 5
    }
    indexes = []
    for l in labels:
        if isinstance(l, str) and l in mapping:
            indexes.append(mapping[l])
        elif isinstance(l, (int, np.integer)) and l >= 0 and l <= 5:
            indexes.append(l)
        else:
            raise ValueError(f"Unknown or invalid label: '{l}' (type: {type(l)}). "
                             "Please ensure labels are 'propre', 'FGSM', 'BIM', 'MIM', 'PGD', 'CW', or corresponding integers 0-5.")
    return np.array(indexes)


def build_detector_model(input_shape, num_classes=6):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_adversarial_dataset(clean_images, clean_labels, classifier_model_instance, output_path):
    if classifier_model_instance is None:
        raise ValueError("Classifier model instance is None. Required for attack generation.")

    processed_clean_images = []
    for img in clean_images:
        if img.shape[:2] != (128, 128):
            img = Image.fromarray((img * 255).astype(np.uint8)).resize((128, 128))
            img = np.array(img) / 255.0

        if img.shape[-1] == 1:
            processed_clean_images.append(np.concatenate([img] * 3, axis=-1))
        elif img.shape[-1] == 4:
            processed_clean_images.append(img[..., :3])
        else:
            processed_clean_images.append(img)
    clean_images = np.array(processed_clean_images, dtype=np.float32)

    adv_data = []
    adv_data += [('propre', img) for img in clean_images]

    attack_types = [
        ('FGSM', fgsm_attack),
        ('BIM', bim_attack),
        ('MIM', mim_attack),
        ('PGD', pgd_attack),
        ('CW', cw_attack)
    ]

    for attack_name, attack_fn in attack_types:
        print(f"Generating {attack_name} attacks...")
        for i, (img, lbl) in enumerate(zip(clean_images, clean_labels)):
            try:
                adv_img = attack_fn(img, lbl, classifier_model_instance)
                adv_data.append((attack_name, adv_img))
            except Exception as e:
                print(f"Error generating {attack_name} attack for label {lbl} (image {i}): {e}")
                continue

    out_df = pd.DataFrame({
        'label': [lbl for lbl, _ in adv_data],
        'image': [img for _, img in adv_data]
    })
    out_df.to_parquet(output_path)


def train_detector(dataset_path, save_path):
    df = pd.read_parquet(dataset_path)
    print(f"--- Debugging `train_detector` data preparation ---")
    print(f"TensorFlow devices available for training: {tf.config.list_logical_devices()}")
    print(f"DataFrame 'image' column head type: {type(df['image'].iloc[0])}")
    if isinstance(df['image'].iloc[0], np.ndarray):
        print(f"First image array shape: {df['image'].iloc[0].shape}, dtype: {df['image'].iloc[0].dtype}")
    else:
        print(f"First image data (NOT numpy array): {df['image'].iloc[0]}")

    x = np.stack(df['image'].to_numpy())
    y = to_categorical(label_to_index(df['label'].to_numpy()), num_classes=6)

    model = build_detector_model(x.shape[1:])
    print(f"Training detector model with input shape: {x.shape[1:]}")
    model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save(save_path)
    print(f"Detector model saved to: {save_path}")


# --- Flask Routes ---
@app.route('/train', methods=['POST'])
def train_endpoint_json(): # Renaming this function might be good, e.g., train_endpoint
    dataset_local_path = None
    classifier_local_path = None
    extracted_data_path = None

    try:
        # Use request.form for regular fields and request.files for file uploads
        dataset_method = request.form.get('dataset_method')
        classifier_method = request.form.get('classifier_method')

        dataset_identifier = None
        if dataset_method == 'upload':
            dataset_file = request.files.get('dataset_file')
            if dataset_file and dataset_file.filename != '':
                dataset_identifier = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dataset_file.filename))
                dataset_file.save(dataset_identifier)
                print(f"Uploaded dataset file to: {dataset_identifier}")
            else:
                return jsonify({'error': 'No dataset file uploaded.'}), 400
        elif dataset_method == 'path' or dataset_method == 'url':
            dataset_identifier = request.form.get('dataset_identifier') # Use the name from JS
            if not dataset_identifier:
                return jsonify({'error': 'Dataset identifier (path/URL) is missing.'}), 400
        else:
            return jsonify({'error': 'Invalid dataset method chosen.'}), 400

        classifier_identifier = None
        if classifier_method == 'upload':
            classifier_file = request.files.get('classifier_file') # Use the name from JS
            if classifier_file and classifier_file.filename != '':
                classifier_identifier = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(classifier_file.filename))
                classifier_file.save(classifier_identifier)
                print(f"Uploaded classifier file to: {classifier_identifier}")
            else:
                return jsonify({'error': 'No classifier file uploaded.'}), 400
        elif classifier_method == 'path' or classifier_method == 'url':
            classifier_identifier = request.form.get('classifier_identifier') # Use the name from JS
            if not classifier_identifier:
                return jsonify({'error': 'Classifier identifier (path/URL) is missing.'}), 400
        else:
            return jsonify({'error': 'Invalid classifier method chosen.'}), 400


        # Now, proceed with resolving the paths/URLs and training as you intended
        # The get_file_from_path_or_url function can still be used for URLs/paths
        final_dataset_path, dataset_error = get_file_from_path_or_url(dataset_identifier, app.config['UPLOAD_FOLDER'], allowed_extensions=['.zip', '.parquet'])
        if dataset_error:
            # If the dataset was an upload, final_dataset_path is already set from the save operation
            # This handles cases where path/URL was used
            if dataset_method != 'upload':
                return jsonify({'error': f'Dataset error: {dataset_error}'}), 400

        final_classifier_path, classifier_error = get_file_from_path_or_url(classifier_identifier, app.config['UPLOAD_FOLDER'], allowed_extensions=['.h5'])
        if classifier_error:
            if classifier_method != 'upload':
                return jsonify({'error': f'Classifier model error: {classifier_error}'}), 400

        # Ensure that the variables used later (dataset_local_path, classifier_local_path)
        # reflect the correct path regardless of upload or path/url.
        dataset_local_path = final_dataset_path if dataset_method != 'upload' else dataset_identifier
        classifier_local_path = final_classifier_path if classifier_method != 'upload' else classifier_identifier


        adv_dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], f"adv_dataset_{uuid.uuid4()}.parquet")
        detector_model_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detector_model_{uuid.uuid4()}.h5")
        

        loaded_classifier_model = tf.keras.models.load_model(
            classifier_local_path,
            compile=False,
            safe_mode=False
        )
        print("Classifier model loaded successfully.")

        images = None
        labels = None
        target_image_size = (128, 128)

        # Your logic for processing ZIP/Parquet remains largely the same
        if dataset_local_path.lower().endswith('.zip'):
            print("Processing ZIP file...")
            extracted_data_path = os.path.join(app.config['UPLOAD_FOLDER'], f"extracted_{uuid.uuid4()}")
            os.makedirs(extracted_data_path, exist_ok=True)
            with zipfile.ZipFile(dataset_local_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_data_path)
            print(f"Dataset extracted to: {extracted_data_path}")
            images, labels = load_images_from_folder(extracted_data_path, target_size=target_image_size)
            numeric_labels = label_to_index(labels)

        elif dataset_local_path.lower().endswith('.parquet'):
            print("Processing Parquet file...")
            df = pd.read_parquet(dataset_local_path)
            loaded_images = []
            loaded_labels = []

            for index, row in df.iterrows():
                image_data = row['image']
                label = row['label']

                processed_img = None
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    img_bytes = image_data['bytes']
                    img = Image.open(io.BytesIO(img_bytes)).resize(target_image_size)
                    processed_img = np.array(img) / 255.0
                elif isinstance(image_data, np.ndarray):
                    processed_img = image_data.astype(np.float32)
                    if processed_img.max() > 1.0:
                        processed_img /= 255.0
                    if processed_img.shape[:2] != target_image_size:
                        pil_img = Image.fromarray((processed_img * 255).astype(np.uint8)).resize(target_image_size)
                        processed_img = np.array(pil_img) / 255.0
                else:
                    print(f"Unsupported image data format in parquet row {index}: {type(image_data)}")
                    continue

                if processed_img is not None:
                    if processed_img.shape[-1] == 4:
                        processed_img = processed_img[..., :3]
                    elif processed_img.ndim == 2:
                        processed_img = np.stack([processed_img]*3, axis=-1)

                    loaded_images.append(processed_img)
                    loaded_labels.append(label)

            if not loaded_images:
                return jsonify({'error': 'No valid images loaded from Parquet.'}), 400

            images = np.stack(loaded_images)
            labels = np.array(loaded_labels)
            numeric_labels = label_to_index(labels)
            print("Dataset loaded from Parquet file.")

        else:
            return jsonify({'error': 'Unsupported dataset file type. Please provide a .zip or .parquet dataset.'}), 400

        if images is None or len(images) == 0:
            return jsonify({'error': 'No images found or processed from the dataset.'}), 400

        print(f"Images shape: {images.shape}, dtype: {images.dtype}")

        # Ensure numeric_labels is defined and correct for generate_adversarial_dataset
        if 'numeric_labels' not in locals():
            numeric_labels = label_to_index(labels) # Calculate if not already from ZIP

        generate_adversarial_dataset(images, numeric_labels, loaded_classifier_model, adv_dataset_path)
        print(f"Adversarial dataset saved to: {adv_dataset_path}")

        train_detector(adv_dataset_path, detector_model_path)
        print(f"Detector model saved to: {detector_model_path}")

        # Correction: Use os.path.basename consistently
        adversarial_dataset_url = f"/uploads/{os.path.basename(adv_dataset_path)}"
        detector_model_url = f"/uploads/{os.path.basename(detector_model_path)}" # Corrected this line

        return jsonify({
            'message': 'Training completed successfully',
            'adversarial_dataset_url': adversarial_dataset_url,
            'detector_model_url': detector_model_url
        }), 200

    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files (existing logic is fine here)
        if dataset_local_path and dataset_local_path.startswith(app.config['UPLOAD_FOLDER']):
            # Ensure not to delete the generated adv_dataset_*.parquet
            if not os.path.basename(dataset_local_path).startswith("adv_dataset_"):
                try:
                    os.remove(dataset_local_path)
                    print(f"Deleted temporary dataset file: {dataset_local_path}")
                except OSError as e:
                    print(f"Error deleting temporary dataset file {dataset_local_path}: {e}")

        if classifier_local_path and classifier_local_path.startswith(app.config['UPLOAD_FOLDER']):
            # Ensure not to delete the generated detector_model_*.h5
            if not os.path.basename(classifier_local_path).startswith("detector_model_"):
                try:
                    os.remove(classifier_local_path)
                    print(f"Deleted temporary classifier file: {classifier_local_path}")
                except OSError as e:
                    print(f"Error deleting temporary classifier file {classifier_local_path}: {e}")

        if extracted_data_path and os.path.exists(extracted_data_path):
            try:
                shutil.rmtree(extracted_data_path)
                print(f"Deleted extracted dataset folder: {extracted_data_path}")
            except OSError as e:
                print(f"Error deleting extracted dataset folder {extracted_data_path}: {e}")
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    detector_local_path = None
    classifier_local_path = None
    image_local_path = None

    try:
        detector_method = request.form.get('detector_method')
        classifier_method = request.form.get('classifier_method')
        image_method = request.form.get('image_method')

        print(f"Predict: Detector Method: {detector_method}")
        print(f"Predict: Classifier Method: {classifier_method}")
        print(f"Predict: Image Method: {image_method}")

        # --- Resolve Detector Model Path ---
        if detector_method == 'upload':
            detector_file = request.files.get('detector_file')
            if detector_file and detector_file.filename != '':
                detector_local_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(detector_file.filename))
                detector_file.save(detector_local_path)
                print(f"Uploaded detector file to: {detector_local_path}")
            else:
                return jsonify({'error': 'No detector file uploaded.'}), 400
        elif detector_method == 'path' or detector_method == 'url':
            detector_identifier = request.form.get('detector_identifier_text')
            if not detector_identifier:
                return jsonify({'error': 'Detector model path/URL is missing.'}), 400
            detector_local_path, error = get_file_from_path_or_url(detector_identifier, app.config['UPLOAD_FOLDER'],
                                                                   allowed_extensions=['.h5'])
            if error:
                return jsonify({'error': f'Detector model error: {error}'}), 400
        else:
            return jsonify({'error': 'Invalid detector method chosen.'}), 400

        # --- Resolve Classifier Model Path ---
        if classifier_method == 'upload':
            classifier_file = request.files.get('classifier_file')
            if classifier_file and classifier_file.filename != '':
                classifier_local_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                                     secure_filename(classifier_file.filename))
                classifier_file.save(classifier_local_path)
                print(f"Uploaded classifier file to: {classifier_local_path}")
            else:
                return jsonify({'error': 'No classifier file uploaded.'}), 400
        elif classifier_method == 'path' or classifier_method == 'url':
            classifier_identifier = request.form.get('classifier_identifier_text')
            if not classifier_identifier:
                return jsonify({'error': 'Classifier model path/URL is missing.'}), 400
            classifier_local_path, error = get_file_from_path_or_url(classifier_identifier, app.config['UPLOAD_FOLDER'],
                                                                     allowed_extensions=['.h5'])
            if error:
                return jsonify({'error': f'Classifier model error: {error}'}), 400
        else:
            return jsonify({'error': 'Invalid classifier method chosen.'}), 400

        # --- Resolve Image Path ---
        if image_method == 'upload':
            image_file = request.files.get('image_file')
            if image_file and image_file.filename != '':
                image_local_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
                image_file.save(image_local_path)
                print(f"Uploaded image file to: {image_local_path}")
            else:
                return jsonify({'error': 'No image file uploaded.'}), 400
        elif image_method == 'path' or image_method == 'url':
            image_identifier = request.form.get('image_identifier_text')
            if not image_identifier:
                return jsonify({'error': 'Image path/URL is missing.'}), 400
            image_local_path, error = get_file_from_path_or_url(image_identifier, app.config['UPLOAD_FOLDER'],
                                                                allowed_extensions=['.png', '.jpg', '.jpeg'])
            if error:
                return jsonify({'error': f'Image error: {error}'}), 400
        else:
            return jsonify({'error': 'Invalid image method chosen.'}), 400

        # --- Crucial File Existence Checks ---
        if not detector_local_path or not os.path.exists(detector_local_path):
            print(f"CRITICAL ERROR: Detector model path is invalid or does not exist: {detector_local_path}")
            return jsonify({'error': f'Detector model path is invalid or does not exist: {detector_local_path}'}), 400
        if not classifier_local_path or not os.path.exists(classifier_local_path):
            print(f"CRITICAL ERROR: Classifier model path is invalid or does not exist: {classifier_local_path}")
            return jsonify(
                {'error': f'Classifier model path is invalid or does not exist: {classifier_local_path}'}), 400
        if not image_local_path or not os.path.exists(image_local_path):
            print(f"CRITICAL ERROR: Image path is invalid or does not exist: {image_local_path}")
            return jsonify({'error': f'Image path is invalid or does not exist: {image_local_path}'}), 400

        # --- Load Models ---
        print(f"Loading detector model from: {detector_local_path}")
        detector = load_model(detector_local_path, compile=False)  # Use compile=False for inference
        print("Detector model loaded successfully.")

        print(f"Loading classifier model from: {classifier_local_path}")
        classifier = load_model(classifier_local_path, compile=False)  # Use compile=False for inference
        print("Classifier model loaded successfully.")

        # --- Load and Process Image ---
        print(f"Loading image from: {image_local_path}")
        img = load_img(image_local_path, target_size=(128, 128))
        print(f"Image loaded, original format: {img.format if hasattr(img, 'format') else 'N/A'}")

        img_arr = img_to_array(img) / 255.0
        print(f"Image array shape after img_to_array: {img_arr.shape}")

        # Ensure 3 channels (RGB)
        if img_arr.shape[-1] == 4:  # RGBA to RGB
            img_arr = img_arr[..., :3]
            print("Converted RGBA to RGB.")
        elif img_arr.ndim == 2 or img_arr.shape[-1] == 1:  # Grayscale to RGB
            img_arr = np.stack([img_arr] * 3, axis=-1)
            print("Converted grayscale/single channel to RGB.")

        # Ensure correct image array shape (batch, height, width, channels)
        if img_arr.ndim == 3:  # If already (height, width, channels)
            img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
        elif img_arr.ndim == 4 and img_arr.shape[0] != 1:  # If batch dim exists but is not 1
            # This scenario shouldn't happen with single image loading, but good to check
            print("Warning: Image array already has a batch dimension not equal to 1. Using as is.")

        print(f"Final image array shape before prediction: {img_arr.shape}")

        # --- Make Predictions ---
        print("Making detector prediction...")
        pred_detector = detector.predict(img_arr)
        pred_detector_class = np.argmax(pred_detector, axis=1)[0]
        print(f"Detector raw prediction: {pred_detector}, class: {pred_detector_class}")

        attack_labels = ['propre', 'FGSM', 'BIM', 'MIM', 'PGD', 'CW']
        detected_attack_type = attack_labels[pred_detector_class]
        print(f"Detected attack type: {detected_attack_type}")

        # --- Return Results ---
        # === Dans la route /predict ===

        # Remplacer la partie "Return Results" par :
        print(f"Detector raw prediction: {pred_detector}, class: {pred_detector_class}")
        print(f"Detected attack type: {detected_attack_type}")

        # Construire la réponse détaillée
        response = {
            'result': detected_attack_type,
            'detector_probs': pred_detector[0].tolist(),  # Convertir le array numpy en liste
            'detector_class': int(pred_detector_class),
            'classifier_probs': None,
            'classified_label': None
        }

        if detected_attack_type == 'propre':
            print("Detected as propre, making classifier prediction...")
            cls_pred = classifier.predict(img_arr)
            cls_class = np.argmax(cls_pred, axis=1)[0]
            print(f"Classifier raw prediction: {cls_pred}, class: {cls_class}")

            response['classifier_probs'] = cls_pred[0].tolist()
            response['classified_label'] = int(cls_class)

        return jsonify(response)

    except Exception as e:
        print(f"CRITICAL ERROR during prediction: {e}")
        traceback.print_exc()  # Print full traceback to the Flask console
        return jsonify({'error': f"Prediction failed due to server error: {str(e)}"}), 500

    finally:
        # --- Clean Up Temporary Files ---
        files_to_clean = [detector_local_path, classifier_local_path, image_local_path]
        for f_path in files_to_clean:
            if f_path and os.path.exists(f_path) and f_path.startswith(app.config['UPLOAD_FOLDER']):
                filename = os.path.basename(f_path)
                # Avoid deleting models/datasets that might have been generated by /train
                # and are meant to persist (e.g., the detector_model_*.h5 or adv_dataset_*.parquet)
                if not (filename.startswith("detector_model_") and filename.endswith(".h5")) and \
                        not (filename.startswith("adv_dataset_") and filename.endswith(".parquet")):
                    try:
                        os.remove(f_path)
                        print(f"Deleted temporary file: {f_path}")
                    except OSError as e:
                        print(f"Error deleting temporary file {f_path}: {e}")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify({"message": "Bonjour depuis Flask!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
