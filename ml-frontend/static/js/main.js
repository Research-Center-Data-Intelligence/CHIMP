const EXPERIMENTATION_SERVER_URL = "http://localhost:5253";

const VID_WIDTH = 1280, VID_HEIGHT = 720;
const HIDDEN_CANVAS_WIDTH = 320, HIDDEN_CANVAS_HEIGHT = 180;
// const HIDDEN_CANVAS_WIDTH = 640, HIDDEN_CANVAS_HEIGHT = 360;

let id_calibrated_model = '', id_in_progress_calibration;

let sock;
let video_origin, canvas_origin;

let has_recently_updated_data = false;
let PREDICTION_TIMEOUT = 500;

function init() {
    // MEDIA WEBCAM CAPTURE
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
        alert("Your browser doesn't seem to support the use of a webcam. Please use a more modern browser.");
        return;
    }

    video_origin = document.createElement('video');
    video_origin.id = 'video_origin';
    video_origin.width = VID_WIDTH;
    video_origin.height = VID_HEIGHT;

    canvas_origin = document.createElement('canvas');
    canvas_origin.width = HIDDEN_CANVAS_WIDTH;
    canvas_origin.height = HIDDEN_CANVAS_HEIGHT;

    navigator.mediaDevices.getUserMedia({
            video: true
        })
        .then(stream => {
            video_origin.srcObject = stream;
            video_origin.onloadedmetadata = (e) => video_origin.play();
        })
        .catch(msg => console.log('Error: ' + msg));


    // SOCKET.IO
    sock = io.connect('http://' + document.domain + ':' + location.port);

    sock.on('connect',
        function() {
            console.log('Initialised SocketIO connection...');

            // START CAPTURE
            capture();
        });

    sock.on('disconnect',
        function() {
            console.log('Terminated SocketIO connection.');
        });

    sock.on('update-data', (data) =>
        {
            // console.log(`Updated data: ${data}`);

            if (!has_recently_updated_data) {
                has_recently_updated_data = true;

                data.forEach(face => {
                    // !! NOTE: Remove the line below to display the information of the last face instead.
                    // !! TODO: Create a multi-face information display
                    face = data[0]

                    face.forEach(emotion => {
                        prediction_text = (emotion[1]*100).toFixed(2) + '%';

                        switch (emotion[0]) {
                            case 'angry':
                                data_angry.innerText = prediction_text;
                                break;
                            case 'disgust':
                                data_disgust.innerText = prediction_text;
                                break;
                            case 'fear':
                                data_fear.innerText = prediction_text;
                                break;
                            case 'happy':
                                data_happy.innerText = prediction_text;
                                break;
                            case 'neutral':
                                data_neutral.innerText = prediction_text;
                                break;
                            case 'sad':
                                data_sad.innerText = prediction_text;
                                break;
                            case 'surprise':
                                data_surprise.innerText = prediction_text;
                                break;
                        }
                    });
                });

                setTimeout(function () {
                    has_recently_updated_data = false;
                }, PREDICTION_TIMEOUT)
            }
        });

    // Disconnect before closing the window
    window.onunload = function() {
        sock.disconnect()
    }
}

// CAPTURE AND MANIPULATE WEBCAM FEED
const capture = () => {
    canvas_origin.getContext('2d').drawImage(video_origin, 0, 0, canvas_origin.width, canvas_origin.height);
    canvas_origin.toBlob((blob) => {
        current_frame = blob;
        console.log("emitting process-image signal");

        sock.emit('process-image', {user_id: id_calibrated_model, image_blob: blob}, (data) => {
            let imgData = new Blob([data], {type: 'image/jpg'});
            let img = new Image();
            img.onload = () => preview.getContext('2d').drawImage(img, 0, 0, preview.width, preview.height);
            img.src = URL.createObjectURL(imgData);
            capture();
        });
    });
}

// MODEL TRAINING AND CALIBRATION
$(document).ready(function() {
    let btn = $('#btn_model_calibrate');
    let input = $('#input_data_upload');

    btn.attr('disabled', true);
    input.change(function() {
        if ($(this).val())
            btn.removeAttr('disabled');
        else
            btn.attr('disabled', true);
    });
});

function trainModel() {
    // Only run if button is not disabled, then disable training button, can only request one model training per time.
    // - Warning: should be given a time-out at server level in a production setting. Suffices for demonstration
    //              purposes with CHIMP.
    if (btn_model_training.getAttribute('disabled') !== null)
        return;

    btn_model_training.disabled = true;

    // Call training on experimentation server.
    $.ajax({
        url: '/train',
        type: 'POST',
        success: function(result) {
            console.log('Training initiated.');
            window.alert('A training procedure has been initiated. This will take more than an hour to complete. ' +
                'Do not await completion of model training. You can close this window at any time.');
        },
        error: function(error) {
            console.log(`Error in training call: ${error}`);
            window.alert('Could not call for a training procedure at this time. Please try again later.');
            btn_model_training.disabled = false;
        },
    });

    // Can implement a websocket system as to give feedback when training has completed, but model training takes longer
    //  than a user session would/should take.
}

function calibrateModel() {
    // Get socket id and prepare connection#
    id_in_progress_calibration = sock.id;
    let url = `/calibrate?user_id=${id_in_progress_calibration}`;
    console.log(`uploading for id: ${id_in_progress_calibration}`);

    // Get zip file and add it to the form data
    let formData = new FormData();
    formData.append('zipfile', input_data_upload.files[0], input_data_upload.files[0].name);

    // Send request
    $.ajax({
        url: url,
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(result) {
            console.log(`Successfully created a calibrated model.`);
            window.alert('Successfully created a personalised model. ' +
                'Now using your personalised model for predictions.');
            id_calibrated_model = id_in_progress_calibration;
            id_in_progress_calibration = '';
        },
        error: function(error) {
            // Notify the user of the error and re-enable the calibration button
            console.log(error);
            console.log(`Error in calibration call: ${error}`);
            window.alert('Could not call for calibration at this time. Please try again later.');
            btn_model_calibrate.disabled = false;
        },
    });

    // Notify user of the process and disable the calibration button
    console.log('Calibration initiated.');
    window.alert('A calibration procedure has been started. When the calibration process has finished, it will ' +
        'automatically be applied to this current session. You will be notified when the personalised model is ready.');
    btn_model_calibrate.disabled = true;
}
