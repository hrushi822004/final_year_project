import cv2

def webcam_module(camera_index=0):
    # Open the webcam
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows, safe on others too

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam is running... Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Warning: Failed to capture frame. Exiting...")
            break

        # Display the frame in a window
        cv2.imshow("Webcam Feed", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting webcam feed...")
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_module()
