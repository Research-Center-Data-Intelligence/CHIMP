let mediaRecorder;
let recordedChunks = [];
const videoElement = document.getElementById('videoElement');
const countdownText = document.getElementById('countdownText');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const pauseButton = document.getElementById('pauseButton');
const resumeButton = document.getElementById('resumeButton');
const saveButton = document.getElementById('saveButton');
const countdownOverlay = document.getElementById('countdownOverlay');

// Function to update button states
function updateButtonState(isRecording, isPaused) {
    startButton.disabled = isRecording;
    stopButton.disabled = !isRecording;
    pauseButton.disabled = !isRecording || isPaused;
    resumeButton.disabled = !isRecording || !isPaused;
    saveButton.disabled = recordedChunks.length === 0;
    
    // Visual feedback: Update button styles based on state
    startButton.classList.toggle('active', isRecording);
    pauseButton.classList.toggle('active', isPaused);
}

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Failed to access webcam.');
    }
}

function startRecordingWithCountdown() {
    return new Promise((resolve, reject) => {
        countdownOverlay.style.display = 'flex';
        countdownText.textContent = '3';

        let countdown = 3;
        const countdownInterval = setInterval(() => {
            countdown--;
            countdownText.textContent = countdown.toString();
            if (countdown === 0) {
                clearInterval(countdownInterval);
                countdownOverlay.style.display = 'none';
                resolve();
            }
        }, 1000);
    });
}

async function startRecording() {
    try {
        if (!videoElement.srcObject) {
            await setupCamera();
        }

        recordedChunks = []; // Clear previously recorded chunks

        await startRecordingWithCountdown();

        mediaRecorder = new MediaRecorder(videoElement.srcObject);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            updateButtonState(true, false); // Recording started
        };

        mediaRecorder.onpause = () => {
            updateButtonState(true, true); // Recording paused
        };

        mediaRecorder.onresume = () => {
            updateButtonState(true, false); // Recording resumed
        };

        mediaRecorder.onstop = () => {
            updateButtonState(false, false); // Recording stopped
        };

        mediaRecorder.start();
        console.log('Recording started...');
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Failed to start recording.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log('Recording stopped...');
    } else {
        console.error('MediaRecorder not initialized or not recording.');
    }
}

function pauseRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.pause();
        console.log('Recording paused...');
    } else {
        console.error('MediaRecorder not initialized or not recording.');
    }
}

function resumeRecording() {
    if (mediaRecorder && mediaRecorder.state === 'paused') {
        mediaRecorder.resume();
        console.log('Recording resumed...');
    } else {
        console.error('Recording is not paused.');
    }
}

function saveRecording() {
    if (recordedChunks.length === 0) {
        console.error('No recorded data available.');
        return;
    }

    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'recorded-video.webm';
    document.body.appendChild(a);
    a.click();

    // Clean up
    recordedChunks = [];
    URL.revokeObjectURL(url);
    document.body.removeChild(a);

    // Visual feedback: Reset button states after saving
    updateButtonState(false, false);
}

// Event listeners for buttons
startButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);
pauseButton.addEventListener('click', pauseRecording);
resumeButton.addEventListener('click', resumeRecording);
saveButton.addEventListener('click', saveRecording);

// Initialize webcam on page load
setupCamera().catch(error => {
    console.error('Failed to initialize webcam:', error);
});
