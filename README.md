# Beat-Synced-Squares.
Beat-Synced Squares
Turn any video file—or your live webcam feed—into a pulsing network of squares that dance to the beat of your music.

(Suggestion: Record a short clip of the effect and replace the image above to show it off!)

Features
Live Webcam Support: Runs in real-time using your computer's webcam and microphone.

Video File Processing: Overlay the effect on any existing video file and export the result.

Beat-Driven Spawning: Uses Librosa (for files) and Sounddevice (for mic) to detect audio onsets and spawn visuals in sync with the music.

Dynamic Visuals:

Squares are created at high-contrast feature points (ORB keypoints).

Smoothly tracks motion between frames using Lucas-Kanade optical flow.

Subtle jitter adds an organic, lively feel.

Ambient "noise" spawns ensure the screen never goes silent.

Edges connect each square to its nearest neighbors, creating a living graph aesthetic.

Requirements
Python 3.9+

FFmpeg

Must be installed and accessible in your system's PATH. You can check this by opening a terminal and running ffmpeg -version.

Installation & Setup
Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Create and activate a virtual environment:

# Create the environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate

Install the required packages:

pip install -r requirements.txt

Usage
You can run the effect in two modes:

1. Live Webcam Mode
To run the effect using your default webcam and microphone, simply run:

python main.py --input 0

A window will pop up with the live video feed.

Press 'q' on your keyboard with the window selected to quit.

If you have multiple webcams, you can try --input 1, --input 2, etc.

2. Video File Processing
To apply the effect to a video file and save the output:

python main.py --input path/to/your/video.mp4 --output cool_effect.mp4

Customization
You can tweak the visuals using these command-line arguments:

Argument

Default

Description

--life-frames

30

How many frames a square stays on screen.

--pts-per-beat

15

Max number of new squares to spawn on each beat.

--ambient-rate

2.0

Average number of random squares spawned per second in silence.

--jitter-px

0.5

How much each square "jitters" per frame for organic motion.

--neighbor-links

2

Number of lines connecting each square to its nearest neighbors.

--min-size / --max-size

20/50

The size range for the squares in pixels.

Example: For a more chaotic effect with short-lived squares and more links:

python main.py --input 0 --life-frames 15 --pts-per-beat 25 --neighbor-links 4

How It Works
The script's pipeline is as follows:

Audio Input:

For video files, the audio track is extracted. Librosa is used to detect the precise timestamps of musical onsets (beats).

For webcam mode, the script listens to the microphone in a separate thread using Sounddevice. It detects onsets in real-time.

Visual Processing (OpenCV):

Spawning: When a beat is detected, the ORB algorithm finds high-contrast keypoints in the frame. A random subset of these become new TrackedPoint squares.

Tracking: In each subsequent frame, the Lucas-Kanade optical flow algorithm calculates the new position of each active square.

Rendering: The squares, their labels, and the lines connecting them are drawn onto the final frame before it's displayed or saved.

License
This project is licensed under the MIT License. See the LICENSE file for details.
