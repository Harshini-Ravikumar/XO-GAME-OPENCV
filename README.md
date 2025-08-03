# 🤖✋ Gesture-Controlled Tic-Tac-Toe (XO Game)

Welcome to a whole new way of playing Tic-Tac-Toe — **no clicks, no taps, just gestures**!  
Use your index finger to draw, play, and reset the board — all through your webcam. It’s **intuitive, fast, and surprisingly competitive**.

---

## 🎮 How It Works

- **Hand Tracking** is powered by [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) and OpenCV.
- Hover your finger over a grid cell for a second to mark your **"X"**.
- The **smart bot ("O")** replies instantly with a surprisingly tough move!
- If you hover over the **RESET** button for a moment, the game resets.
- **Scoreboard** keeps track of all X wins, O wins, and draws.
- Each "X" and "O" is beautifully animated into place — and a smooth line slides across when someone wins.

> 🎁 Most games will end in a draw — but I've hidden an **easter egg**: there’s a way to **win every time**.  
Can you find the strategy that always beats the bot? 👀

---

## 🧠 Features

- 🖐️ Gesture-based input (finger hover to select)
- 🤖 Smart AI bot opponent using strategy heuristics
- ✨ Animated drawing for X, O, and winning lines
- 📊 Built-in scoreboard
- 🔁 Touch-free RESET button
- 🥚 Secret winning trick for the persistent ones

---

## 🧱 Libraries Used

- [**OpenCV** (`cv2`)](https://opencv.org/) — for webcam access, drawing shapes, animation
- [**MediaPipe**](https://mediapipe.dev/) — for real-time hand detection and tracking
- `numpy` — for array operations and grid setup
- `time`, `copy` — for animation timing and deep cloning
