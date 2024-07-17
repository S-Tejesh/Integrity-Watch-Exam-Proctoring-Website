import base64
import json
from datetime import datetime, timedelta
import io
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import tensorflow as tf
from PIL import Image
from bson import ObjectId
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response
from pymongo import MongoClient
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from bson.binary import Binary
import cv2
import mediapipe as mp
import numpy as np

from ultralytics import YOLO
yolo_model = YOLO("yolov8m.pt")

app = Flask(__name__)
app.secret_key = 'something'
login_manager = LoginManager()
login_manager.init_app(app)
#client = MongoClient('mongodb+srv://admin:5R9g0GELT2C5REnA@cluster0.qc15eve.mongodb.net/')  # Update the connection string accordingly
client = MongoClient('mongodb://localhost:27017/')
db = client['Integrity-Watch']  # Replace 'mydatabase' with your database name
users_collection = db['users']
exams_collection = db['exams']
stop_proctoring = False
# Load the YOLO model

class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id
@login_manager.user_loader
def load_user(user_id):
    # You would retrieve the user from your database here
    return User(user_id)
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})

        if user:
            if user['password'] == password:
                session['user_id'] = str(user['_id'])
                user1=User(user['_id'])
                login_user(user1)
                return redirect(url_for('home'))
            else:
                error = 'Invalid password!'
        else:
            error = 'Username not found!'

        return render_template('test.html', error=error)
    return render_template('test.html')

@app.route('/admin-login', methods=['POST'])
def admin_login():
    admin_username = request.form.get('adminUsername')
    admin_password = request.form.get('adminPassword')

    # Check admin credentials from MongoDB
    admin = db["administrators"].find_one({'username': admin_username, 'password': admin_password})

    if admin:
        session['admin_id'] = str(admin['_id'])
        user1 = User(admin['_id'])
        login_user(user1)
        # Pass exam_history as a query parameter, not as part of the template rendering
        return redirect(url_for('admin_home'))

    return render_template('test.html', error="Invalid admin username or password.")
@app.route('/admin-home', methods=['GET'])
@login_required
def admin_home():
    admin_id = session.get('admin_id')
    if admin_id:
        return render_template('adminhome.html')
    else:
        # Redirect to the login page if admin_id is not found in session
        return redirect(url_for('login'))


@app.route('/exam-history', methods=['GET'])
@login_required
def exam_history():
    admin_id = session.get('admin_id')
    if admin_id:
        # Retrieve exam history for the admin
        exam_history = db['exams'].find({'admin_id': admin_id}).sort('created_at', -1)
        exam_history = list(exam_history)
        # Dictionary to store student count for each exam
        exam_student_counts = {}

        # Retrieve student count for each exam
        for exam in exam_history:
            student_count = db[f"exam_{exam['_id']}"].count_documents({})
            exam_student_counts[exam['_id']] = student_count
            exam['created_at'] = exam['created_at'].strftime('%Y-%m-%d')

        return render_template('examhistory.html', exam_history=exam_history, exam_student_counts=exam_student_counts)
    else:
        # Redirect to the login page if admin_id is not found in session
        return redirect(url_for('admin_home'))
@app.route('/admin/create-exam', methods=['GET', 'POST'])
@login_required
def create_exam():
    if request.method == 'POST':
        # Process form data for creating an exam
        exam_title = request.form.get('examTitle')
        exam_description = request.form.get('examDescription')
        exam_instructions = request.form.get('examInstructions')
        exam_duration = int(request.form.get('examDuration'))  # Convert to integer

        registration_date = request.form.get('registrationDate')
        exam_date = request.form.get('examDate')
        questions = request.form.getlist('question[]')  # Get list of questions
        options1 = request.form.getlist('option1[]')
        options2 = request.form.getlist('option2[]')
        options3 = request.form.getlist('option3[]')
        options4 = request.form.getlist('option4[]')
        answers = request.form.getlist('answer[]')

        # Combine data for each question
        exam_data = [{
            'question': question,
            'options': [options1[i], options2[i], options3[i], options4[i]],
            'answer': str(int(answers[i])-1)
        } for i, question in enumerate(questions)]

        admin_id = session.get('admin_id')
        current_date = datetime.now()

        # Convert date strings to datetime objects
        registration_date = datetime.strptime(registration_date, '%Y-%m-%d')
        exam_date = datetime.strptime(exam_date, '%Y-%m-%d')

        # Save the exam data to MongoDB with the date
        exams_collection.insert_one({
            'admin_id': admin_id,
            'title': exam_title,
            'description': exam_description,  # Added description field
            'instructions': exam_instructions,  # Added instructions field
            'duration': exam_duration,  # Added duration field
            'registration_date': registration_date,
            'exam_date': exam_date,
            'questions': exam_data,
            'created_at': current_date
        })

        return render_template('creation_loading.html')
    return render_template('create_exam.html')
@app.route('/display-images/<exam_id>')
@login_required
def display_images(exam_id):
    try:
        # Query the cheating_collection for all documents related to the specified exam
        documents = db[f'exam_{exam_id}'].find()

        # Dictionary to store images grouped by user ID along with user names
        user_images = {}
        conducted=False
        # Iterate over each document
        for document in documents:
            # Retrieve the user ID
            user_id = document.get('user_id')

            # Retrieve and decode the images
            images_data = document.get('cheat_imgs', [])
            # Convert BSON Binary objects to base64-encoded strings
            images_base64 = [base64.b64encode(image).decode('utf-8') for image in images_data]

            # Query the users collection to get the user name
            user = db['users'].find_one({'_id': ObjectId(user_id)})
            user_name = user.get('username') if user else 'Unknown'

            # Append user name and images to the dictionary, grouped by user ID
            if images_base64:
                if user_id in user_images:
                    user_images[user_id]['username'] = user_name
                    user_images[user_id]['images'].extend(images_base64)
                else:
                    user_images[user_id] = {'username': user_name, 'images': images_base64}
            exam_date=exams_collection.find_one({'_id':ObjectId(exam_id)}).get('exam_date');
            current_date = datetime.now()
            conducted = current_date > exam_date

        return render_template('view_details.html', user_images=user_images,conducted=conducted)
    except Exception as e:
        print('Error displaying images:', e)
        return 'Error displaying images. Please try again later.'

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['newUsername']
    password = request.form['newPassword']

    # Check if the username already exists
    if users_collection.find_one({'username': username}):
        error= 'Username already exists!'
        return render_template('test.html', error=error)
    # Insert new user into the database
    users_collection.insert_one({'username': username, 'password': password})
    error='Signup successful!'
    return render_template('test.html', error=error)

@app.route('/home', methods=['GET'])
@login_required
def home():
    return render_template('home.html')
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    response = make_response(redirect(url_for('login')))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return redirect(url_for('login'))

@app.route('/student/exam-details/<exam_id>', methods=['GET', 'POST'])
@login_required
def exam_details(exam_id):
    # Retrieve exam details from the database
    exam = exams_collection.find_one({'_id': ObjectId(exam_id)})

    return render_template('exam_details.html', exam=exam)

@app.route('/student/register/<exam_id>')
@login_required
def register_exam(exam_id):
    # Perform registration logic here (e.g., update user's registered exams in the database)
    
    # Redirect to the student home page or display a success message
    return redirect(url_for('exam_details',exam_id=exam_id))

@app.route('/exams')
@login_required
def exams():
    try:
        # Get the current date
        current_date = datetime.now()

        # Filter exams based on the current date being before or equal to the last date to register
        available_exams = []

        # Iterate over all exams
        for exam in exams_collection.find({'registration_date': {'$gte': current_date.replace(hour=0, minute=0, second=0, microsecond=0)}}):
            exam_id = exam['_id']

            # Check if there is no collection with the exam ID or no entry in the exam ID collection
            if not db[f"exam_{exam_id}"].find_one() or not db[f"exam_{exam_id}"].find_one({'user_id': session.get('user_id')}):
                available_exams.append(exam)

        return render_template('exams.html', exams=available_exams)
    except Exception as e:
        print('Error retrieving exams:', e)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Extract user ID, exam ID, and captured images from the form data
        user_id = request.form.get('user_id')
        exam_id = request.form.get('exam_id')
        captured_images_json = request.form.get('captured_images')

        # Check if the 'captured_image' data URL is present in the request
        if captured_images_json:
            # Handle captured image data URL
            captured_images = json.loads(captured_images_json)

            # Convert the base64 encoded image strings to OpenCV format
            images_data = [cv2.imdecode(np.frombuffer(base64.b64decode(image.split(',')[1]), np.uint8), cv2.IMREAD_COLOR) for image in captured_images]

            # Process and store only the faces
            faces_data = []

            for image_data in images_data:
                # Extract face region (adjust coordinates as needed)
                face_region = image_data[120:120 + 250, 200:200 + 250, :]

                # Convert the face region to base64 encoding
                _, encoded_face = cv2.imencode('.jpg', face_region)
                encoded_face_base64 = Binary(encoded_face.tobytes())

                # Store the encoded face in the faces_data list
                faces_data.append(encoded_face_base64)

            existing_document = db[f'exam_{exam_id}'].find_one({'user_id': user_id})

            # Insert the document only if the user_id does not exist
            if not existing_document:
                db[f'exam_{exam_id}'].insert_one({
                    'user_id': user_id,
                    'images': faces_data,
                    'wrote': False,
                    'cheat_imgs': [],
                    'exam_result':{},
                    'cheat_score':0
                })
            exam_date=exams_collection.find_one({'_id':ObjectId(exam_id)}).get('exam_date');
        return render_template('registration_loading.html',date=exam_date)
    except Exception as e:
        print('Error uploading images:', e)
        return jsonify({'success': False, 'message': 'Error uploading images'}), 400

@app.route('/view-images/<user_id>/<exam_id>')
def view_images(user_id, exam_id):
    # Query the registered_students collection
    document = db[f'exam_{exam_id}'].find_one({'user_id': user_id})

    if document:
        # Retrieve and decode the images
        images_data = document.get('images', [])

        # Convert BSON Binary objects to base64-encoded strings
        images_base64 = [base64.b64encode(image).decode('utf-8') for image in images_data]

        return render_template('view_images.html', images_base64=images_base64)
    else:
        return 'No images found for the given user and exam IDs.'

@app.route('/upcoming-exams')
@login_required
def upcoming_exams():
    try:
        # Retrieve the user ID from the session or authentication mechanism
        user_id = session.get('user_id')  # Change this line based on your authentication mechanism

        # Get the current date
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # List to store upcoming exams
        upcoming_exams = []

        # Iterate over each exam collection
        for exam_collection_name in db.collection_names():
            if exam_collection_name.startswith('exam_'):
                # Check if the user is registered for the exam and hasn't written it yet
                exam_id = exam_collection_name.split('_')[-1]
                registration_data = db[exam_collection_name].find_one({'user_id': user_id})
                if registration_data and (not registration_data.get('wrote')):
                    # Query the exams_collection for exam details
                    exam_det = exams_collection.find_one(
                        {'_id': ObjectId(exam_id), 'exam_date': {'$gte': current_date, '$lt': current_date + timedelta(days=1)}})
                    if exam_det:
                        upcoming_exams.append(exam_det)

        return render_template('upcoming_exams.html', exams=upcoming_exams)
    except Exception as e:
        print('Error retrieving upcoming exams:', e)


@app.route('/start-exam/<exam_id>', methods=['GET','POST'])
@login_required
def start_exam(exam_id):
    # Add logic to determine whether the user is eligible to take the exam
    # If eligible, redirect to the face authentication page
    return redirect(url_for('face_authentication', exam_id=exam_id))

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


model = tf.keras.models.load_model('siamesemodel.h5',
                                   custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': BinaryCrossentropy})


# Preprocess function (same as your provided code)
def preprocess_image(image):
    image = tf.image.resize(image, (100, 100))
    image = image / 255.0
    return image
@app.route('/face-authentication/<exam_id>', methods=['GET', 'POST'])
@login_required
def face_authentication(exam_id):
    if request.method == 'POST':
        try:
            # Get image data from the request
            image_data = request.get_data()

            # Convert the raw binary data to a NumPy array
            input_image = np.array(Image.frombytes('RGBA', (640, 480), image_data, 'raw'))
            input_image = input_image[120:120 + 250, 200:200 + 250, :]
            input_image = cv2.resize(input_image, (100, 100))

            # Extract RGB channels (assuming the original image has an alpha channel)
            input_image = input_image[:, :, :3]
            cv2.imwrite('input_image.jpg', cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

            # Fetch registered images from MongoDB for the given user and exam ID
            registered_images_document = db[f'exam_{exam_id}'].find_one({
                'user_id': session.get('user_id'),
            },{"images": 1})
            images_data = registered_images_document.get("images", [])

            registered_images=[]

            for i, image in enumerate(images_data):
                try:
                    # Convert binary data to numpy array using PIL
                    image =(Image.open(io.BytesIO(image)))

                    # Convert PIL Image to numpy array
                    image_array = np.array(image)
                    # Resize the cropped registered image to (100, 100)
                    resized_registered_image = cv2.resize(image_array, (100, 100))

                    # Extract RGB channels from registered image
                    rgb_registered_image = resized_registered_image[:, :, :3]
                    cv2.imwrite(f'registered_image_{i}.jpg', cv2.cvtColor(rgb_registered_image, cv2.COLOR_RGB2BGR))

                    registered_images.append(rgb_registered_image)
                except Exception as e:
                    print(f"Error decoding image: {str(e)}")
            #print("Input Image Shape:", input_image.shape)
            #print("Registered Image Shapes:", [img.shape for img in registered_images])

            # Verify the face authentication for each registered image
            results = []
            print(len(registered_images))
            for registered_img in registered_images:
                # Preprocess the registered image
                registered_img = preprocess_image(registered_img)

                if registered_img is not None:
                    registered_img = preprocess_image(registered_img)
                    # Make Predictions
                    print("Input Image Shape:", input_image.shape)
                    print("Registered Image Shape:", registered_img.shape)
                    result = model.predict(
                        [np.expand_dims(input_image, axis=0), np.expand_dims(registered_img, axis=0)])
                    results.append(result)
                else:
                    print("Error: Preprocessed image is None")

            # Detection Threshold: Metric above which a prediction is considered positive
            detection = np.sum(np.array(results) > 0.71)

            # Verification Threshold: Proportion of positive predictions / total positive samples
            verification = detection / len(registered_images)
            verified = verification > 0.5

            # Return verification results as JSON
            print(results,verified)
            return jsonify({'verified': str(verified)})
        except Exception as e:
            print(str(e))
            # Handle the exception and return an appropriate response (e.g., error message)
            return jsonify({'error': str(e)}), 500
    exam = exams_collection.find_one({'_id': ObjectId(exam_id)})
    return render_template('face_auth.html', exam=exam)
@app.route('/loading/<exam_id>')
@login_required
def loading_page(exam_id):
    return render_template('exam_loading.html', exam_id=exam_id)

@app.route('/proctoring_page/<exam_id>', methods=['GET'])
@login_required
def proctoring(exam_id):
    try:
        global stop_proctoring
        stop_proctoring = False
        exam_object_id = ObjectId(exam_id)
        exam_document = db['exams'].find_one({'_id': exam_object_id})
        db[f'exam_{exam_id}'].update_one(
            {'user_id': session.get('user_id')},
            {'$set': {'wrote': True}}
        )
        if exam_document is not None:
            return render_template('proctoring_page.html',exam=exam_document)
        else:
            return jsonify({'error': 'Exam not found'}), 404
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/trigger_proctoring/<exam_id>', methods=['POST'])
def trigger_proctoring(exam_id):
    # Load the YOLO model
    yolo_model = YOLO("yolov8m.pt")

    # Initialize MediaPipe Face Mesh and Face Detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Parameters for detecting cheating
    cheating_threshold = 30  # Number of consecutive frames indicating cheating
    reset_threshold = 10  # Number of consecutive frames to reset the cheating flag
    no_face_threshold = 60  # Number of consecutive frames indicating no face detection
    multiple_face_threshold = 60  # Number of consecutive frames indicating multiple face detections

    cheating_count = 0
    reset_count = 0
    no_face_count = 0
    multiple_face_count = 0
    cheating_flag = False
    image_saved_for_cheating = False  # Flag to track whether an image has been saved for the current cheating event
    score=0
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
            mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if stop_proctoring:
                db[f'exam_{exam_id}'].update_one(
                    {'user_id': session.get('user_id')},
                    {'$set': {'cheat_score': score}}
                )
                break
            # Object detection using YOLO
            results = yolo_model.predict(frame, classes=[67])

            # Perform object detection on the frame
            objects_detected = any(r.boxes.cls.tolist() for r in results)

            if objects_detected:
                # If objects are detected
                cheating_count = 0
                reset_count = 0
                no_face_count = 0
                multiple_face_count = 0
                cheating_flag = True
                if not image_saved_for_cheating:  # Save image only if not already saved for the current event
                    image_saved_for_cheating = True
                    # Save the image where cheating is detected
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    score+=4
                    # Save the image to MongoDB as BSON
                    bson_data = Binary(img_bytes)
                    db[f'exam_{exam_id}'].update_one(
                        { 'user_id': session.get('user_id')},
                        {'$push': {'cheat_imgs': bson_data}}
                    )

            # Cheating detection using MediaPipe and OpenCV
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Detect faces
            face_results = face_detection.process(image)

            # Convert image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.detections:
                if len(face_results.detections) == 1:
                    # Convert image to RGB and process
                    ih, iw, _ = image.shape
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            face_3d = []
                            face_2d = []
                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                    x, y = int(lm.x * iw), int(lm.y * ih)

                                    # Get the 2D Coordinates
                                    face_2d.append([x, y])

                                    # Get the 3D Coordinates
                                    face_3d.append([x, y, lm.z])

                            face_2d = np.array(face_2d, dtype=np.float64)
                            face_3d = np.array(face_3d, dtype=np.float64)

                            # The camera matrix
                            focal_length = 1 * iw
                            cam_matrix = np.array([[focal_length, 0, ih / 2],
                                                   [0, focal_length, iw / 2],
                                                   [0, 0, 1]])

                            # The distortion parameters
                            dist_matrix = np.zeros((4, 1), dtype=np.float64)

                            # Solve PnP
                            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                            # Get rotational matrix
                            rmat, jac = cv2.Rodrigues(rot_vec)

                            # Get angles
                            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                            # Get the y rotation degree
                            y = angles[1] * 360

                            # See where the user's head is looking
                            if y < -10:
                                cheating_count += 1
                                reset_count = 0  # Reset the reset counter
                                if cheating_count >= cheating_threshold and not image_saved_for_cheating:
                                    cheating_flag = True
                                    score+=1
                            elif y > 10:
                                cheating_count += 1
                                reset_count = 0  # Reset the reset counter
                                if cheating_count >= cheating_threshold and not image_saved_for_cheating:
                                    cheating_flag = True
                                    score += 1
                            else:
                                reset_count += 1
                                if reset_count >= reset_threshold:
                                    cheating_count = 0  # Reset the cheating counter
                                    cheating_flag = False
                                    image_saved_for_cheating = False  # Reset the flag
                else:
                    # Increment multiple face count and check for cheating
                    multiple_face_count += 1
                    if multiple_face_count >= multiple_face_threshold:
                        cheating_flag = True
                        multiple_face_count = 0
                        score += 2
                        # You might want to include additional actions here, such as logging or notifications
            else:
                # Increment no face count and check for cheating
                no_face_count += 1
                if no_face_count >= no_face_threshold:
                    cheating_flag = True
                    no_face_count = 0
                    score += 3
            if cheating_flag and not image_saved_for_cheating:
                _, img_encoded = cv2.imencode('.jpg', image)
                img_bytes = img_encoded.tobytes()

                # Save the image to MongoDB as BSON
                bson_data = Binary(img_bytes)
                db[f'exam_{exam_id}'].update_one(
                    {'user_id': session.get('user_id')},
                    {'$push': {'cheat_imgs': bson_data}}
                )
                image_saved_for_cheating = True  # Set the flag to indicate that the image has been saved
   # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

@app.route('/submit_exam/<exam_id>', methods=['POST'])
@login_required
def submit_exam(exam_id):
    global stop_proctoring
    stop_proctoring = True
    exam = exams_collection.find_one({'_id': ObjectId(exam_id)})

    # Get the submitted answers from the form
    submitted_answers = {}
    for question_number, option in request.form.items():
        print(option)
        if question_number.startswith('question'):
            question_number = int(question_number.replace('question', ''))
            submitted_answers[question_number] = option
    # Calculate the score
    score = 0
    for i, question in enumerate(exam['questions'], 1):
        correct_answer = question['options'][int(question['answer'])]
        submitted_answer = submitted_answers.get(i)
        print(correct_answer)
        print(submitted_answer)
        if submitted_answer == correct_answer:
            score += 1

    # Calculate total marks
    total_marks = len(exam['questions'])

    # Calculate percentage
    percentage = (score / total_marks) * 100

    # Prepare exam result data
    user_id = session['user_id']
    exam_result = {
        'score': score,
        'total_marks': total_marks,
        'percentage': percentage
    }
    db[f'exam_{exam_id}'].update_one({'user_id': user_id}, {'$set': {'exam_result': exam_result}})
    return redirect(url_for('home'))


@app.route('/exam_results/<exam_id>')
@login_required
def exam_results(exam_id):
    exam_collection = db[f'exam_{exam_id}']
    exam_results = list(exam_collection.find({},{'exam_result':1,'user_id':1,'cheat_score':1}).sort('exam_result.percentage',-1))
    exam_date = exams_collection.find_one({'_id': ObjectId(exam_id)}).get('exam_date');
    current_date = datetime.now()
    conducted = current_date > exam_date

    # Add rank to each result
    rank = 1
    prev_percentage = None
    for exam_result in exam_results:
        exam_result['name']=str(db['users'].find_one({'_id':ObjectId(exam_result['user_id'])})['username'])
        if prev_percentage is None or exam_result['exam_result']['percentage'] < prev_percentage:
            exam_result['rank'] = rank
        else:
            exam_result['rank'] = prev_rank
        prev_percentage = exam_result['exam_result']['percentage']
        prev_rank = rank
        rank += 1
    return render_template('exam_results.html', exam_results=exam_results,conducted=conducted)




if __name__ == '__main__':
    app.run(debug=True)
