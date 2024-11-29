import io
import zipfile
from flask import Flask, request
from minio import Minio
from werkzeug.utils import secure_filename
import json
import psycopg2
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)

DATASTORE_ACCESS_KEY="yZmhrURuUhaeVSUagMRa"
DATASTORE_SECRET_KEY="cnk0OxGuIgVx4La0prNaWUv7JpriCnxZfq2417ba"
DATASTORE_URI="localhost:9000"

# Initialize MinIO client
client = Minio(
    DATASTORE_URI,  # Replace with your MinIO endpoint
    access_key=DATASTORE_ACCESS_KEY,
    secret_key=DATASTORE_SECRET_KEY,
    secure=False
)
global bucket_name
bucket_name = "manageddataset"

if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

db_config = {
    "dbname": "datasetbase",
    "user": "pguser",
    "password": "pgpassword",
    "host": "localhost",  # Use the Docker host's IP if not running locally
    "port": "5432"
}
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Insert query
insert_query = """
INSERT INTO datapoints (x, y, metadata)
VALUES (%s, %s, %s)
RETURNING id;
"""

# Endpoint to handle file upload
@app.route('/upload-zip', methods=['POST'])
def upload_zip():
    global bucket_name
    # Check if a file was part of the request
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    print("check")
    print(f"File Content-Type: {file.content_type}")
    # Ensure the file is a ZIP
    # if file.content_type != 'application/zip':
    #     return "Invalid file type, must be a ZIP file", 400
    
    # Secure the filename
    zip_filename = secure_filename(file.filename)
    
    # Create a BytesIO buffer from the uploaded file data
    file_data = file.read()
    zip_buffer = io.BytesIO(file_data)

    uploaded_files = []
    # Process the ZIP file in memory
    with zipfile.ZipFile(zip_buffer, 'r') as zip_archive:
        for file_name in zip_archive.namelist():
            # Read each file within the ZIP archive
            with zip_archive.open(file_name) as extracted_file:
                # Upload each extracted file to MinIO
                object_name = secure_filename(file_name)  # Ensure safe file name
                
                file_content = extracted_file.read()
                file_stream = io.BytesIO(file_content)
                file_stream.seek(0)  # Ensure pointer is at the start

                # Upload the file to MinIO
                result = client.put_object(bucket_name, object_name, file_stream, length=len(file_content))

                ourl= f"https://{DATASTORE_URI}/{bucket_name}/{object_name}"
                # Collect file info to update the database later
                uploaded_files.append({
                    "object_name": object_name,
                    "bucket_name": bucket_name,
                    "object_url": ourl
                })
                # Data to insert
                data = (ourl, "labeltest", json.dumps(uploaded_files[-1]))

                # Execute the query
                cursor.execute(insert_query, data)
                
                conn.commit()

                print(f"Uploaded {object_name} to MinIO")


    select_query = "SELECT * FROM datapoints;"
    cursor.execute(select_query)

    # Fetch all rows
    rows = cursor.fetchall()

    # Print each row
    for row in rows:
        print(f"ID: {row[0]}, X: {row[1]}, Y: {row[2]}, Metadata: {row[3]}")


    retrieve_query = "SELECT metadata FROM datapoints WHERE id = %s;"
    file_id = 50  # Replace with the appropriate ID or query filter
    cursor.execute(retrieve_query, (file_id,))
    metadata = cursor.fetchone()

    if metadata:
        file_metadata = json.loads(metadata[0])  # Parse the JSON metadata
        bucket_name = file_metadata['bucket_name']
        object_name = file_metadata['object_name']
    else:
        print("No file found with the given ID.")

    response = client.get_object(bucket_name, object_name)
    # Read the binary data
    image_data = response.read()

    # Convert the binary data to a PIL image
    image = Image.open(BytesIO(image_data))

    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    print(image_array.shape)

    cursor.close()
    conn.close()

    return f"Files from {zip_filename} uploaded to MinIO", 200

if __name__ == '__main__':
    app.run(debug=True)
