// Event listener for when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Get references to the DOM elements
    const videoElement = document.getElementById('videoElement');
    const countdownOverlay = document.getElementById('countdownOverlay');
    const countdownText = document.getElementById('countdownText');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const pauseButton = document.getElementById('pauseButton');
    const resumeButton = document.getElementById('resumeButton');
    const saveButton = document.getElementById('saveButton');
    const emotionButtons = document.querySelectorAll('.emotionButton');
    const emotionOrderList = document.getElementById('emotionOrderList');

    
    let mediaRecorder; // MediaRecorder instance
    let currentEmotion = ''; // Current emotion being recorded
    let recordedBlob; // Blob of the recorded video
    let isRecording = false; // Flag to indicate if recording is in progress
    let isPaused = false; // Flag to indicate if recording is paused
    let emotionQueue = []; // Queue of emotions to be recorded
    let recordedSessions = []; // Array to store recorded sessions
    let isQueuePaused = false; // Flag to indicate if the recording queue is paused
    const socket = io(CONFIG.SOCKET_URL); // Socket connection for sending recorded data

    socket.on('connect',
        function() {
            console.log('Initialised SocketIO connection...');
        });

    socket.on('disconnect',
        function() {
            console.log('Terminated SocketIO connection.');
        });

    const recordedEmotions = new Set(); // Set to track recorded emotions

    // Function to set up the webcam
    async function setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            // Wait for the video metadata to load
            await new Promise((resolve) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play();
                    resolve();
                };
            });
        } catch (error) {
            console.error('Error accessing webcam:', error);
            alert('Failed to access webcam.');
        }
    }

    // Function to display a countdown before starting the recording
    async function startRecordingWithCountdown(emotion) {
        return new Promise((resolve) => {
            countdownOverlay.style.display = 'flex'; // Show countdown overlay
            const smiley = CONFIG.EMOTION_SMILEYS[emotion] || ''; // Get smiley for the emotion
            countdownText.textContent = `${CONFIG.COUNTDOWN_TIME} - ${smiley} ${emotion}`; // Set countdown text

            let countdown = CONFIG.COUNTDOWN_TIME; // Initialize countdown time
            const countdownInterval = setInterval(() => {
                countdown--;
                countdownText.textContent = `${countdown} - ${smiley} ${emotion}`; // Update countdown text
                if (countdown === 0) {
                    clearInterval(countdownInterval);
                    countdownOverlay.style.display = 'none'; // Hide countdown overlay
                    resolve(); // Resolve the promise after countdown ends
                }
            }, 1000);
        });
    }

    // Function to start recording a video for a given emotion
    async function startRecording(emotion = '') {
        if (isRecording) {
            console.warn('Already recording.');
            return;
        }

        currentEmotion = emotion;

        await startRecordingWithCountdown(currentEmotion); // Start countdown before recording

        const options = { mimeType: 'video/x-matroska;codecs=avc1' }; // Recording options
        mediaRecorder = new MediaRecorder(videoElement.srcObject, options);
        recordedBlob = null;

        // Event handler for when data is available from the media recorder
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedBlob = event.data; // Store the recorded blob
            }
        };

        // Event handler for when the recording stops
        mediaRecorder.onstop = () => {
            console.log(`Recording ${currentEmotion ? 'for ' + currentEmotion : ''} stopped.`);
            if (recordedBlob) {
                recordedSessions.push({ emotion: currentEmotion, blob: recordedBlob }); // Add recorded session to the array
                saveButton.disabled = false; // Enable the save button
                markEmotionAsRecorded(currentEmotion); // Mark emotion as recorded
            }
            isRecording = false;
            if (!isQueuePaused) {
                processEmotionQueue(); // Process next emotion in the queue
            }
        };

        mediaRecorder.start(); // Start recording
        console.log(`Recording ${emotion ? 'for ' + emotion : ''} started...`);
        isRecording = true;
        isPaused = false;

        // Stop recording after the maximum recording time
        setTimeout(() => {
            stopRecording();
        }, CONFIG.MAX_RECORDING_TIME * 1000);

        updateButtonState(); // Update the button states
    }

    // Function to stop the recording
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.requestData(); // Request the recorded data
            mediaRecorder.stop(); // Stop the media recorder
            isRecording = false;

            // Reset recorded emotions
            resetRecordedEmotions();

            updateButtonState(); // Update the button states
        }
    }

    // Function to pause the recording queue
    function pauseRecordingQueue() {
        if (isRecording) {
            stopRecording();
        }
        isQueuePaused = true; // Set queue paused flag to true
        console.log('Recording queue paused...');
        updateButtonState(); // Update the button states
    }

    // Function to resume the recording queue
    function resumeRecordingQueue() {
        isQueuePaused = false; // Set queue paused flag to false
        console.log('Recording queue resumed...');
        processEmotionQueue(); // Process next emotion in the queue
        updateButtonState(); // Update the button states
    }

    // Function to save the recorded sessions
    function saveRecording() {
        if (recordedSessions.length === 0) {
            console.warn('No recorded data to save.');
            return;
        }

        // Generate a timestamp for the recorded sessions
        const timestamp = new Date().toLocaleString('sv-SE', { timeZone: 'Europe/Amsterdam' }).replace(/[: ]/g, '-');
        const username = USERNAME;
        
        const blobs = recordedSessions.map(session => session.blob);
        const emotions = recordedSessions.map(session => session.emotion);
        const timestamps = recordedSessions.map((session, index) => `${timestamp}-${index}`);

        const payload = {
            user_id: '',
            username: username,
            image_blobs: blobs,
            emotions: emotions,
            timestamps: timestamps
        };
        socket.emit('upload_managed_calibration_data',payload)
        console.log(`Recording for ${emotions} saved.`);

        recordedSessions = []; // Clear the recorded sessions array
        saveButton.disabled = true; // Disable the save button

        updateButtonState(); // Update the button states
    }

    // Function to update the state of the buttons based on the current recording state
    function updateButtonState() {
        startButton.disabled = isRecording || emotionQueue.length > 0; // Disable start button if recording or queue is not empty
        stopButton.disabled = !isRecording && emotionQueue.length === 0; // Disable stop button if not recording and queue is empty
        pauseButton.disabled = !isRecording || isPaused || emotionQueue.length === 0; // Disable pause button if not recording, paused, or queue is empty
        resumeButton.disabled = !isQueuePaused || emotionQueue.length === 0; // Disable resume button if queue is not paused or empty
        emotionButtons.forEach(button => button.disabled = isRecording); // Disable emotion buttons if recording
    }

    // Function to process the next emotion in the queue
    function processEmotionQueue() {
        if (emotionQueue.length > 0 && !isQueuePaused) {
            const nextEmotion = emotionQueue.shift(); // Get next emotion from the queue
            startRecording(nextEmotion); // Start recording for the next emotion
        }
    }

    // Function to reset the recording state to initial values
    function resetRecordingState() {
        isRecording = false;
        isPaused = false;
        isQueuePaused = false;
        emotionQueue = [];

        updateButtonState(); // Update the button states
    }

    // Event listener for the start button to initiate recording for all emotions in the queue
    startButton.addEventListener('click', () => {
        for (const emotion in CONFIG.EMOTION_SMILEYS) {
            emotionQueue.push(emotion); // Add all emotions to the queue
        }

        processEmotionQueue(); // Process the emotion queue
    });

    // Event listeners for individual emotion buttons to start recording for a specific emotion
    emotionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const emotion = button.getAttribute('data-emotion').toLowerCase(); // Get the emotion from the button attribute

            if (!isRecording) {
                startRecording(emotion); // Start recording for the specific emotion
            } else {
                console.warn('Already recording. Please wait for the current recording to finish.');
            }
        });
    });

    // Event listener for the stop button to stop recording and clear the queue
    stopButton.addEventListener('click', () => {
        isQueuePaused = true; // Set queue paused flag to true
        emotionQueue = []; // Clear the emotion queue
        stopRecording(); // Stop the recording
        resetRecordingState(); // Reset the recording state
    });

    // Event listeners for the pause and resume buttons
    pauseButton.addEventListener('click', pauseRecordingQueue); // Pause the recording queue
    resumeButton.addEventListener('click', resumeRecordingQueue); // Resume the recording queue
    saveButton.addEventListener('click', saveRecording); // Save the recorded sessions

    // Initialize the webcam
    setupCamera().catch(error => {
        console.error('Failed to initialize webcam:', error);
    });

    // Function to mark an emotion as recorded in the UI
    function markEmotionAsRecorded(emotion) {
        recordedEmotions.add(emotion.toLowerCase()); // Add emotion to the recorded set
        // Find existing list item and add 'recorded' class
        const emotionListItem = document.querySelector(`#emotionOrderList li:nth-child(${Object.keys(CONFIG.EMOTION_SMILEYS).indexOf(emotion) + 1})`);
        if (emotionListItem) {
            emotionListItem.classList.add('recorded'); // Mark emotion as recorded in the UI
        }
    }

    // Function to reset the recorded emotions in the UI
    function resetRecordedEmotions() {
        recordedEmotions.clear(); // Clear the recorded emotions set
        // Remove 'recorded' class from all list items
        const listItems = emotionOrderList.getElementsByTagName('li');
        for (const listItem of listItems) {
            listItem.classList.remove('recorded'); // Reset recorded class for all list items
        }
    }

    // Populate emotion list with initial state
    for (const emotion in CONFIG.EMOTION_SMILEYS) {
        const listItem = document.createElement('li');
        listItem.textContent = `${CONFIG.EMOTION_SMILEYS[emotion]} ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`;

        // Check if emotion is recorded
        if (recordedEmotions.has(emotion)) {
            listItem.classList.add('recorded'); // Mark recorded emotions in the initial state
        }

        emotionOrderList.appendChild(listItem); // Append the list item to the emotion order list
    }
});


